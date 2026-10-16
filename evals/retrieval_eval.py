"""检索质量评测：Recall@k、MRR、Precision@k、Rerank 增益、忠实度。

与 `run_eval.py` 的分工：

- `run_eval.py` 测**端到端答案**（用户拿到的回答对不对）
- 本模块只测**检索这一环**（该召回的内容有没有召回到）

分开度量的意义：端到端失败时，如果分不清是"检索没召回"还是"生成不忠实"，
只能靠猜去改。分开之后可以直接定位——Recall 低说明检索有问题，
Recall 高但答案错说明是生成环节的问题。

用法：
    python -m evals.retrieval_eval                    # 检索指标
    python -m evals.retrieval_eval --faithfulness     # 追加忠实度评测
    python -m evals.retrieval_eval --tag my-run
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from agent.rag import (  # noqa: E402
    all_documents_scope,
    get_retriever,
    get_store,
)
from agent.rag.reranker import Reranker  # noqa: E402
from agent.service import MODEL_NAME  # noqa: E402

DATASET_PATH = Path(__file__).parent / "retrieval_dataset.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"

# 参与评测的候选池：要大于最大的 k，否则 Recall@5 会被池子大小截断
CANDIDATE_POOL = 10
K_VALUES = [1, 3, 5]

GENERATE_SYSTEM = (
    "你是一个严格的问答助手。只根据给定的资料回答问题，"
    "绝对不要使用资料之外的任何知识或推测。"
    "如果资料不足以回答，就直接回答“资料中没有相关信息”。"
)

FAITHFULNESS_SYSTEM = (
    "你是 RAG 系统评测专家，负责判断回答是否忠实于资料。"
    "逐条检查回答中的事实陈述，确认每一条都能在资料中找到依据。"
    "只要出现资料中没有的信息（推测、补充、编造、张冠李戴），就判不通过。"
    "回答“资料中没有相关信息”属于忠实行为，应判通过。\n"
    "只输出一个 JSON 对象，不要输出其他内容：\n"
    '{"pass": true, "reason": "一句话说明"} 或 {"pass": false, "reason": "一句话说明"}'
)


@dataclass
class Case:
    id: str
    query: str
    gold_keywords: list[str]
    gold_source: str = ""
    notes: str = ""

    @property
    def parsed(self) -> bool:
        return bool(self.query and self.gold_keywords)


@dataclass
class CaseResult:
    id: str
    query: str
    ranks: dict = field(default_factory=dict)     # {k: recall 0/1}
    rr: float = 0.0                               # reciprocal rank
    precision: dict = field(default_factory=dict)
    found_rank_fused: int | None = None
    found_rank_reranked: int | None = None
    latency_sec: float = 0.0
    notes: str = ""


def load_cases(path: Path = DATASET_PATH) -> list[Case]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] 第 {line_no} 行解析失败：{e}", file=sys.stderr)
                continue
            case = Case(
                id=obj.get("id", f"line{line_no}"),
                query=obj.get("query", ""),
                gold_keywords=obj.get("gold_keywords", []),
                gold_source=obj.get("gold_source", ""),
                notes=obj.get("notes", ""),
            )
            if not case.parsed:
                print(f"[warn] 第 {line_no} 行缺少 query 或 gold_keywords，已跳过",
                      file=sys.stderr)
                continue
            cases.append(case)
    return cases


def is_relevant(text: str, keywords: list[str]) -> bool:
    """相关判定：chunk 必须包含全部黄金关键词。

    用"全部命中"而非"任一命中"，是为了避免只匹配到其中一个常见词就误判为相关
    （例如只含 8000 却不是讲端口的段落）。
    """
    lowered = text.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def first_relevant_rank(chunks, keywords: list[str]) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk.text, keywords):
            return index
    return None


# ==================== 检索指标 ====================

def measure_retrieval(cases: list[Case]) -> tuple[list[CaseResult], dict]:
    store = get_store()
    # 离线评测显式声明"我要全量文档"：不依赖任何隐式默认
    scope = all_documents_scope(store=store)
    reranker = Reranker()
    results: list[CaseResult] = []

    print(f"\n开始检索评测：{len(cases)} 条，候选池 {CANDIDATE_POOL}\n")

    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        hits = store.search(
            case.query, top_k=CANDIDATE_POOL, visible_doc_ids=scope
        )
        fused = [h.chunk for h in hits]
        ranked_pairs = reranker.rerank(case.query, fused, top_n=CANDIDATE_POOL) \
            if reranker.enabled else [(c, 0.0) for c in fused]
        reranked = [chunk for chunk, _ in ranked_pairs]
        latency = time.perf_counter() - started

        # 精排前后分别定位，用于对比 rerank 带来的增益
        rank_fused = first_relevant_rank(fused, case.gold_keywords)
        rank_reranked = first_relevant_rank(reranked, case.gold_keywords)

        result = CaseResult(
            id=case.id,
            query=case.query,
            found_rank_fused=rank_fused,
            found_rank_reranked=rank_reranked,
            latency_sec=round(latency, 3),
            notes=case.notes,
        )
        for k in K_VALUES:
            result.ranks[f"recall@{k}"] = 1 if (rank_fused and rank_fused <= k) else 0
            result.precision[f"precision@{k}"] = round(
                sum(1 for c in fused[:k] if is_relevant(c.text, case.gold_keywords)) / k, 4
            )
        result.rr = round(1.0 / rank_fused, 4) if rank_fused else 0.0

        results.append(result)
        print(f"  [{index:>2}/{len(cases)}] {case.id:<10} "
              f"融合命中@{rank_fused or '-':<4} 精排命中@{rank_reranked or '-':<4} "
              f"{latency:.2f}s")

    summary = aggregate_retrieval(results)
    return results, summary


def aggregate_retrieval(results: list[CaseResult]) -> dict:
    total = len(results)
    summary: dict = {"total_cases": total}

    for k in K_VALUES:
        key = f"recall@{k}"
        summary[f"fused_{key}"] = round(
            sum(r.ranks[key] for r in results) / total, 4) if total else 0.0
        summary[f"precision@{k}"] = round(
            statistics.mean(r.precision[f"precision@{k}"] for r in results), 4
        ) if total else 0.0

    # 精排后的 Recall@1：体现 cross-encoder 把正确结果提到首位的增益
    rerank_recall1 = sum(
        1 for r in results if r.found_rank_reranked and r.found_rank_reranked <= 1
    )
    rerank_recall3 = sum(
        1 for r in results if r.found_rank_reranked and r.found_rank_reranked <= 3
    )
    summary["reranked_recall@1"] = round(rerank_recall1 / total, 4) if total else 0.0
    summary["reranked_recall@3"] = round(rerank_recall3 / total, 4) if total else 0.0
    summary["mrr"] = round(statistics.mean(r.rr for r in results), 4) if total else 0.0
    summary["missed_cases"] = sorted(
        r.id for r in results if r.found_rank_fused is None
    )
    summary["latency_mean_sec"] = round(
        statistics.mean(r.latency_sec for r in results), 3) if total else 0.0
    return summary


# ==================== 忠实度 ====================

def _llm():
    import os

    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
    )


def measure_faithfulness(cases: list[Case]) -> tuple[list[dict], dict]:
    """忠实度：模型基于检索资料作答，检查是否编造了资料外的信息。"""
    retriever = get_retriever()
    scope = all_documents_scope(store=retriever.store)
    llm = _llm()
    rows: list[dict] = []

    print(f"\n开始忠实度评测：{len(cases)} 条\n")

    for index, case in enumerate(cases, 1):
        retrieval = retriever.retrieve(case.query, scope=scope)
        row = {
            "id": case.id,
            "query": case.query,
            "hit": retrieval.hit,
            "answer": "",
            "pass": False,
            "reason": "",
        }

        if not retrieval.hit:
            row["pass"] = True       # 未命中时允许如实说明
            row["reason"] = "未检索到资料，跳过生成"
            rows.append(row)
            print(f"  [{index:>2}/{len(cases)}] {case.id:<10} 未命中，跳过")
            continue

        try:
            answer = llm.invoke([
                SystemMessage(content=GENERATE_SYSTEM),
                HumanMessage(content=(
                    f"资料：\n{retrieval.context}\n\n问题：{case.query}"
                )),
            ]).content or ""
        except Exception as e:
            row["reason"] = f"生成失败：{type(e).__name__}: {e}"
            rows.append(row)
            continue

        row["answer"] = answer
        try:
            judge_raw = llm.invoke([
                SystemMessage(content=FAITHFULNESS_SYSTEM),
                HumanMessage(content=(
                    f"【资料】\n{retrieval.context}\n\n"
                    f"【问题】\n{case.query}\n\n"
                    f"【回答】\n{answer}"
                )),
            ]).content or ""
            cleaned = judge_raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            verdict = json.loads(cleaned)
            row["pass"] = bool(verdict.get("pass"))
            row["reason"] = str(verdict.get("reason", ""))
        except Exception as e:
            row["reason"] = f"判定失败：{type(e).__name__}: {e}"

        rows.append(row)
        print(f"  [{index:>2}/{len(cases)}] {case.id:<10} "
              f"{'PASS' if row['pass'] else 'FAIL'}  {row['reason'][:50]}")

    passed = sum(1 for r in rows if r["pass"])
    summary = {
        "total_cases": len(rows),
        "faithfulness": round(passed / len(rows), 4) if rows else 0.0,
        "failed_cases": sorted(r["id"] for r in rows if not r["pass"]),
    }
    return rows, summary


# ==================== 报告 ====================

def print_report(summary: dict, faith: dict | None):
    print("\n" + "=" * 60)
    print("  检索质量")
    print("=" * 60)
    print(f"  {'指标':<24}{'融合排序':>12}{'精排后':>12}")
    print("-" * 60)
    print(f"  {'Recall@1':<24}{summary['fused_recall@1'] * 100:>11.1f}%"
          f"{summary['reranked_recall@1'] * 100:>11.1f}%")
    print(f"  {'Recall@3':<24}{summary['fused_recall@3'] * 100:>11.1f}%"
          f"{summary['reranked_recall@3'] * 100:>11.1f}%")
    print(f"  {'Recall@5':<24}{summary['fused_recall@5'] * 100:>11.1f}%"
          f"{'—':>12}")
    print("-" * 60)
    print(f"  MRR（融合排序）        {summary['mrr']:.4f}")
    for k in K_VALUES:
        print(f"  Precision@{k}              {summary[f'precision@{k}'] * 100:.1f}%")
    print(f"  平均检索延迟           {summary['latency_mean_sec']}s")
    if summary["missed_cases"]:
        print(f"  完全未召回：{', '.join(summary['missed_cases'])}")
    else:
        print("  完全未召回：无")

    if faith:
        print("=" * 60)
        print(f"  忠实度                 {faith['faithfulness'] * 100:.1f}%")
        if faith["failed_cases"]:
            print(f"  不忠实用例：{', '.join(faith['failed_cases'])}")
    print("=" * 60 + "\n")


def render_markdown(summary: dict, faith: dict | None, tag: str) -> str:
    lines = [
        f"# 检索质量评测报告（{tag}）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 指标 | 融合排序 | 精排后 |",
        "| --- | --- | --- |",
        f"| Recall@1 | {summary['fused_recall@1'] * 100:.1f}% | "
        f"{summary['reranked_recall@1'] * 100:.1f}% |",
        f"| Recall@3 | {summary['fused_recall@3'] * 100:.1f}% | "
        f"{summary['reranked_recall@3'] * 100:.1f}% |",
        f"| Recall@5 | {summary['fused_recall@5'] * 100:.1f}% | — |",
        f"| MRR | {summary['mrr']:.4f} | — |",
        "| Precision@1 | "
        f"{summary['precision@1'] * 100:.1f}% | — |",
        f"| Precision@3 | {summary['precision@3'] * 100:.1f}% | — |",
        f"| Precision@5 | {summary['precision@5'] * 100:.1f}% | — |",
        f"| 平均检索延迟 | {summary['latency_mean_sec']}s | — |",
        "",
        f"完全未召回的用例：{', '.join(summary['missed_cases']) or '无'}",
        "",
    ]
    if faith:
        lines += [
            "## 忠实度",
            "",
            f"通过率 {faith['faithfulness'] * 100:.1f}%",
            "",
            f"不忠实用例：{', '.join(faith['failed_cases']) or '无'}",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--faithfulness", action="store_true", help="追加忠实度评测")
    parser.add_argument("--tag", default="retrieval", help="本次运行标记")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("没有可用用例")

    results, summary = measure_retrieval(cases)

    faith_rows = None
    faith_summary = None
    if args.faithfulness:
        faith_rows, faith_summary = measure_faithfulness(cases)

    print_report(summary, faith_summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tag": args.tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "faithfulness": faith_summary,
        "results": [
            {
                "id": r.id,
                "query": r.query,
                "found_rank_fused": r.found_rank_fused,
                "found_rank_reranked": r.found_rank_reranked,
                "recall": r.ranks,
                "precision": r.precision,
                "rr": r.rr,
                "latency_sec": r.latency_sec,
            }
            for r in results
        ],
        "faithfulness_details": faith_rows,
    }
    json_path = REPORT_DIR / f"retrieval_{stamp}.json"
    md_path = REPORT_DIR / f"retrieval_{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, faith_summary, args.tag), encoding="utf-8")
    (REPORT_DIR / "retrieval_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "retrieval_latest.md").write_text(
        render_markdown(summary, faith_summary, args.tag), encoding="utf-8")

    print(f"报告已写入：\n  {json_path}\n  {md_path}\n")


if __name__ == "__main__":
    main()
