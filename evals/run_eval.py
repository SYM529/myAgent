"""Agent 评测运行器。

用法：
    python -m evals.run_eval                      # 跑全量数据集
    python -m evals.run_eval --skip-network       # 跳过需要联网检索的用例
    python -m evals.run_eval --category math_direct math_multi_step
    python -m evals.run_eval --limit 10 --concurrency 2
    python -m evals.run_eval --tag baseline-v1    # 给 LangSmith trace 打标签

产出：
    evals/reports/report_<timestamp>.json   完整逐条结果
    evals/reports/report_<timestamp>.md     可直接贴进简历 / README 的汇总表
    evals/reports/latest.json / latest.md   最近一次结果，方便对比回归
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台默认 GBK，遇到非 GBK 字符会直接抛 UnicodeEncodeError，统一切到 UTF-8
for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

load_dotenv()  # 必须在导入读取环境变量的模块之前

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402  # pylint: disable=unused-import

from agent.llm import get_model  # noqa: E402
from agent.rag.context import RagRequestContext, reset_context, set_context  # noqa: E402
from agent.pricing import estimate_cost, sum_usage  # noqa: E402
from agent.service import agent, MODEL_NAME  # noqa: E402
from evals.dataset import DATASET_PATH, load_dataset  # noqa: E402,F401
from evals.judges import check_citation, check_tools, judge_answer  # noqa: E402

REPORT_DIR = Path(__file__).parent / "reports"

# 单条用例超时：既防死循环，也作为可靠性指标的一环
CASE_TIMEOUT = 120

# 价格表与 token 计量统一放在 agent.pricing：
# 评测和线上共用一份定义，避免两边算法漂移后无法归因成本变化。


# ==================== 数据集 ====================

# 多 Agent 图架构中"路由"对应的实际工具，用于把路由还原成等效工具调用，
# 使两种架构在同一套数据集上的工具调用指标可比
ROUTE_TO_TOOL = {
    "knowledge": "search_knowledge_base",
    "search": "web_search",
    "compute": "calculate",
    "time": "get_current_time",
}


def filter_cases(cases: list[dict], categories: list[str] | None, skip_network: bool, limit: int | None):
    if categories:
        cases = [c for c in cases if c.get("category") in categories]
    if skip_network:
        cases = [c for c in cases if not c.get("requires_network")]
    if limit:
        cases = cases[:limit]
    return cases


# ==================== 单条执行 ====================

_judge_llm = None


def get_judge_llm():
    """评测判定用的模型，temperature=0 保证评分可复现。

    必须与主模型走同一套 LLM_* 配置（agent.llm.get_model）：之前这里写死
    DEEPSEEK_BASE_URL + 全局 MODEL_NAME，服务商切到硅基流动后模型名与端点
    不匹配，judge 全量 400。get_model 还统一处理了混合推理模型的
    enable_thinking 关闭，避免 <think> 混进评分输出。
    """
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = get_model()
    return _judge_llm


def _final_answer(messages: list) -> str:
    """取最后一条不带 tool_calls 的 AI 消息作为最终答案。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content or ""
    return ""


def _collect_tools(messages: list) -> list[str]:
    """汇总链路中实际调用过的工具名（按调用顺序去重）。"""
    called = []
    seen = set()
    for msg in messages:
        names = []
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            names = [tc.get("name", "") for tc in tool_calls]
        elif getattr(msg, "type", "") == "tool":
            names = [getattr(msg, "name", "") or ""]
        for name in names:
            if name and name not in seen:
                seen.add(name)
                called.append(name)
    return called


async def run_case(case: dict, tag: str) -> dict:
    """执行单条用例：多轮依次提问，只对最后一轮做判定。"""
    judge_llm = get_judge_llm()
    turns = case.get("turns") or [case["input"]]

    messages: list = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    called_tools: list[str] = []
    started = time.perf_counter()
    error = None

    try:
        async with asyncio.timeout(CASE_TIMEOUT):
            for turn_index, question in enumerate(turns):
                messages.append(HumanMessage(content=question))
                # 与线上一致：把用户原问题放进上下文，供短 query 兜底
                set_context(RagRequestContext(user_message=question))
                try:
                    result = await agent.ainvoke(
                        {"messages": messages},
                        config={
                            "tags": ["eval", tag, case.get("category", "unknown")],
                            "metadata": {
                                "case_id": case["id"],
                                "turn": turn_index + 1,
                                "category": case.get("category"),
                            },
                        },
                    )
                finally:
                    reset_context()
                messages = result["messages"]

                for key, value in sum_usage(result["messages"]).items():
                    usage[key] += value
                for name in _collect_tools(result["messages"]):
                    if name not in called_tools:
                        called_tools.append(name)
                # 多 Agent 图架构由专家节点直接调用工具函数，不产生 tool 消息，
                # 因此从路由还原等效的工具调用，保证两种架构可比
                for route in (result.get("routes") or []):
                    tool = ROUTE_TO_TOOL.get(route)
                    if tool and tool not in called_tools:
                        called_tools.append(tool)
    except asyncio.TimeoutError:
        error = f"用例超时（>{CASE_TIMEOUT}s）"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    latency = time.perf_counter() - started
    final_answer = _final_answer(messages)
    last_question = turns[-1]

    if error:
        answer_pass, answer_reason = False, error
    elif not final_answer:
        answer_pass, answer_reason = False, "Agent 未产出最终答案"
    else:
        answer_pass, answer_reason = await judge_answer(
            case.get("judge", {}), judge_llm, last_question, final_answer
        )

    tool_pass, tool_reason = check_tools(
        called_tools,
        case.get("expect_tools", []),
        case.get("tool_match", "any"),
    )

    require_citation = bool(case.get("require_citation"))
    citation_pass, citation_reason = check_citation(final_answer, require_citation)

    return {
        "require_citation": require_citation,
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "question": last_question,
        "turns": len(turns),
        "answer": final_answer,
        "called_tools": called_tools,
        "expect_tools": case.get("expect_tools", []),
        "answer_pass": answer_pass,
        "answer_reason": answer_reason,
        "tool_pass": tool_pass,
        "tool_reason": tool_reason,
        "citation_pass": citation_pass,
        "citation_reason": citation_reason,
        "passed": bool(answer_pass and tool_pass and citation_pass),
        "latency_sec": round(latency, 3),
        "tokens": usage,
        "cost_cny": round(estimate_cost(usage, MODEL_NAME), 6),
        "notes": case.get("notes", ""),
    }


# ==================== 聚合与报告 ====================

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * p))
    return ordered[index]


def aggregate(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    latencies = [r["latency_sec"] for r in results]

    categories: dict[str, dict] = {}
    for r in results:
        bucket = categories.setdefault(r["category"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if r["passed"] else 0

    tool_cases = [r for r in results if r.get("expect_tools") or r.get("called_tools")]
    tool_passed = sum(1 for r in tool_cases if r["tool_pass"])

    citation_cases = [r for r in results if r.get("require_citation")]
    citation_passed = sum(1 for r in citation_cases if r.get("citation_pass"))

    usage = {
        "input_tokens": sum(r["tokens"]["input_tokens"] for r in results),
        "output_tokens": sum(r["tokens"]["output_tokens"] for r in results),
        "total_tokens": sum(r["tokens"]["total_tokens"] for r in results),
    }

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "answer_only_pass_rate": round(
            sum(1 for r in results if r["answer_pass"]) / total, 4
        ) if total else 0.0,
        "tool_call_accuracy": round(tool_passed / len(tool_cases), 4) if tool_cases else 0.0,
        "citation_accuracy": (
            round(citation_passed / len(citation_cases), 4) if citation_cases else None
        ),
        "citation_cases": len(citation_cases),
        "latency_sec": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(_pct(latencies, 0.5), 3),
            "p95": round(_pct(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "tokens": usage,
        "cost_cny": round(sum(r["cost_cny"] for r in results), 6),
        "categories": {
            name: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
            }
            for name, v in sorted(categories.items())
        },
    }


def render_markdown(summary: dict, tag: str, failures: list[dict]) -> str:
    lines = [
        f"# Agent 评测报告（{tag}）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　｜　模型：`{MODEL_NAME}`",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 用例总数 | {summary['total_cases']} |",
        f"| 通过数 | {summary['passed_cases']} |",
        f"| **综合通过率** | {summary['pass_rate'] * 100:.1f}% |",
        f"| 答案正确率（不含工具校验） | {summary['answer_only_pass_rate'] * 100:.1f}% |",
        f"| 工具调用准确率 | {summary['tool_call_accuracy'] * 100:.1f}% |",
        *([f"| 引用准确率 | {summary['citation_accuracy'] * 100:.1f}%"
           f"（{summary['citation_cases']} 条要求引用）|"]
          if summary.get("citation_accuracy") is not None else []),
        *([
            f"| 评测轮数 | {summary['repeat']} |",
            f"| **平均通过率** | {summary['mean_pass_rate'] * 100:.1f}% |",
            f"| 通过率波动区间 | {summary['pass_rate_min'] * 100:.1f}% ~ "
            f"{summary['pass_rate_max'] * 100:.1f}% |",
            f"| 稳定通过率（每轮都通过） | {summary['stable_pass_rate'] * 100:.1f}% |",
            f"| 抖动用例数 | {len(summary['flaky_cases'])} |",
        ] if summary.get("repeat", 1) > 1 else []),
        f"| 平均延迟 | {summary['latency_sec']['mean']}s |",
        f"| P50 延迟 | {summary['latency_sec']['p50']}s |",
        f"| P95 延迟 | {summary['latency_sec']['p95']}s |",
        f"| Token 总量 | {summary['tokens']['total_tokens']} |",
        f"| 估算成本 | CNY {summary['cost_cny']:.4f} |",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 通过 / 总数 | 通过率 |",
        "| --- | --- | --- |",
    ]
    for name, v in summary["categories"].items():
        lines.append(f"| {name} | {v['passed']} / {v['total']} | {v['pass_rate'] * 100:.1f}% |")

    if failures:
        lines += ["", "## 失败用例", "", "| ID | 类别 | 问题 | 失败原因 |", "| --- | --- | --- | --- |"]
        for f in failures[:30]:
            reason = f["answer_reason"] if not f["answer_pass"] else f["tool_reason"]
            reason = str(reason).replace("|", "\\|")[:80]
            question = str(f["question"]).replace("|", "\\|")[:40]
            lines.append(f"| {f['id']} | {f['category']} | {question} | {reason} |")

    lines.append("")
    return "\n".join(lines)


def print_console(summary: dict, failures: list[dict]):
    print("\n" + "=" * 62)
    if summary.get("repeat", 1) > 1:
        print(f"  平均通过率 {summary['mean_pass_rate'] * 100:.1f}%  "
              f"（波动 {summary['pass_rate_min'] * 100:.1f}% ~ "
              f"{summary['pass_rate_max'] * 100:.1f}%，{summary['repeat']} 轮）")
        print(f"  稳定通过率 {summary['stable_pass_rate'] * 100:.1f}%  "
              f"　抖动用例 {len(summary['flaky_cases'])} 条")
        if summary["flaky_cases"]:
            print(f"  抖动用例：{', '.join(summary['flaky_cases'])}")
    print(f"  综合通过率 {summary['pass_rate'] * 100:.1f}%  "
          f"({summary['passed_cases']}/{summary['total_cases']})")
    print(f"  工具调用准确率 {summary['tool_call_accuracy'] * 100:.1f}%   "
          f"答案正确率 {summary['answer_only_pass_rate'] * 100:.1f}%")
    if summary.get("citation_accuracy") is not None:
        print(f"  引用准确率 {summary['citation_accuracy'] * 100:.1f}%  "
              f"（{summary['citation_cases']} 条要求引用标注）")
    print(f"  延迟 mean {summary['latency_sec']['mean']}s / "
          f"P50 {summary['latency_sec']['p50']}s / P95 {summary['latency_sec']['p95']}s")
    print(f"  Token {summary['tokens']['total_tokens']}　成本 CNY {summary['cost_cny']:.4f}")
    print("=" * 62)
    print(f"  {'类别':<20}{'通过/总数':>12}{'通过率':>10}")
    print("-" * 62)
    for name, v in summary["categories"].items():
        print(f"  {name:<20}{str(v['passed']) + '/' + str(v['total']):>12}"
              f"{v['pass_rate'] * 100:>9.1f}%")
    print("=" * 62)
    if failures:
        print(f"\n  失败用例 {len(failures)} 条：")
        for f in failures[:10]:
            reason = f["answer_reason"] if not f["answer_pass"] else f["tool_reason"]
            print(f"   - [{f['id']}] {str(reason)[:70]}")
    print()


# ==================== 主流程 ====================

async def run_once(cases: list[dict], tag: str, concurrency: int,
                   round_index: int = 1) -> list[dict]:
    """执行一轮评测。多线程 tag 带上轮次，便于在 LangSmith 里区分。"""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    round_tag = tag if round_index <= 1 else f"{tag}-r{round_index}"

    async def worker(index: int, case: dict):
        async with semaphore:
            result = await run_case(case, round_tag)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{index:>3}/{len(cases)}] {status}  {result['id']:<16} "
                  f"{result['latency_sec']:>6.2f}s  {result['called_tools']}")
            results.append(result)

    await asyncio.gather(*(worker(i, c) for i, c in enumerate(cases, 1)))
    results.sort(key=lambda r: r["id"])
    return results


def aggregate_stability(runs: list[list[dict]]) -> dict:
    """多轮运行的稳定性聚合。

    LLM 采样有随机性，同一份代码不同轮次的失败用例可能完全不同。
    只看单轮数字，很容易把噪声当成优化效果（我们就踩过：改了 prompt 后
    通过率从 98% 掉到 94%，实际只是换了一批抖动的用例）。
    因此这里额外给出每轮通过率、稳定通过率和抖动用例清单。
    """
    flat = [r for run in runs for r in run]
    summary = aggregate(flat)

    per_case: dict[str, int] = {}
    for run in runs:
        for r in run:
            per_case[r["id"]] = per_case.get(r["id"], 0) + (1 if r["passed"] else 0)

    repeats = len(runs)
    total_cases = len(per_case)
    round_rates = [aggregate(run)["pass_rate"] for run in runs]

    always = [cid for cid, c in per_case.items() if c == repeats]
    never = [cid for cid, c in per_case.items() if c == 0]
    flaky = sorted(
        [cid for cid, c in per_case.items() if 0 < c < repeats],
        key=lambda cid: -per_case[cid],
    )

    summary.update({
        "repeat": repeats,
        "rounds_pass_rate": [round(r, 4) for r in round_rates],
        "mean_pass_rate": round(statistics.mean(round_rates), 4),
        "pass_rate_min": round(min(round_rates), 4),
        "pass_rate_max": round(max(round_rates), 4),
        "stable_pass_rate": round(len(always) / total_cases, 4) if total_cases else 0.0,
        "never_passed_count": len(never),
        "never_passed_cases": sorted(never),
        "flaky_cases": flaky,
        "per_case_pass_count": per_case,
    })
    return summary


async def run(tag: str, categories: list[str] | None, skip_network: bool,
              limit: int | None, concurrency: int, repeat: int = 1) -> dict:
    cases = filter_cases(load_dataset(), categories, skip_network, limit)
    if not cases:
        raise SystemExit("没有匹配的用例，请检查过滤条件")

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    print(f"\n开始评测：{len(cases)} 条用例 × {repeat} 轮，并发 {concurrency}，tag={tag}")
    print(f"LangSmith 追踪：{'已开启' if tracing else '未开启（设置 LANGCHAIN_TRACING_V2=true 可上传 trace）'}\n")

    runs: list[list[dict]] = []
    for round_index in range(1, repeat + 1):
        if repeat > 1:
            print(f"---- 第 {round_index}/{repeat} 轮 ----")
        runs.append(await run_once(cases, tag, concurrency, round_index))

    if repeat > 1:
        summary = aggregate_stability(runs)
    else:
        summary = aggregate(runs[0])
    failures = [r for r in runs[-1] if not r["passed"]]

    print_console(summary, failures)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tag": tag,
        "model": MODEL_NAME,
        "repeat": repeat,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "results": runs[-1],
    }
    json_path = REPORT_DIR / f"report_{stamp}.json"
    md_path = REPORT_DIR / f"report_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, tag, failures), encoding="utf-8")
    (REPORT_DIR / "latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "latest.md").write_text(
        render_markdown(summary, tag, failures), encoding="utf-8")

    print(f"报告已写入：\n  {json_path}\n  {md_path}\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description="运行 Agent 评测")
    parser.add_argument("--category", nargs="*", default=None, help="只跑指定类别")
    parser.add_argument("--skip-network", action="store_true", help="跳过需要联网检索的用例")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=4, help="并发数，默认 4")
    parser.add_argument("--tag", default="baseline", help="标记本次运行，便于对比与 trace 检索")
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复轮数，>1 时额外输出稳定性与抖动用例统计（建议 ≥3）")
    args = parser.parse_args()

    asyncio.run(run(args.tag, args.category, args.skip_network, args.limit,
                    args.concurrency, args.repeat))


if __name__ == "__main__":
    main()
