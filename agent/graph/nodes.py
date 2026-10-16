"""多 Agent 图的节点实现。

设计动机：单 ReAct 循环里，工具越多、system prompt 越长，模型选错工具的
概率越高——本项目已经加到 4 个工具、6 条规则。这里改为"先分流、再专人专办"：

- classify：只判断问题类型，不做回答，prompt 极短
- 各专家：只挂载自己需要的上下文，互不干扰
- synthesize：汇总多路结果，统一处理引用

另外 knowledge 与 search 两路可以并行执行，这是单 ReAct 循环做不到的。
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph.state import (
    ROUTE_COMPUTE,
    ROUTE_DIRECT,
    ROUTE_KNOWLEDGE,
    ROUTE_SEARCH,
    ROUTE_TIME,
    ALL_ROUTES,
)
from agent.llm import get_model

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM = """你是任务分流器。判断用户的问题需要交给哪些专家处理，只输出路由，不要回答问题。

可选专家：
- knowledge：查询本地知识库（项目文档、配置参数、部署方式、内部资料）
- search：联网获取实时信息（天气、新闻、股价、赛事、当前事件）
- compute：得出精确数值（算术、单位换算、归一问题、"还剩多少"这类）
- time：当前时间或日期（"现在几点""今天几号"）
- direct：常识解释、翻译、创作、代码编写，不需要任何工具

判断规则（务必遵守）：
1. 只要问题要求得出一个具体数值，就必须选 compute，包括看起来简单的算术。
2. 问"现在几点/今天几号/当前时间"选 time，**不要**选 search——时间不是靠搜索得来的。
3. 涉及本项目的配置、参数、端口、部署方式、文档内容，一律选 knowledge，
   即使你觉得凭常识也能答——必须依据文档作答，不能绕过知识库。
4. 只有天气、新闻、股价这类真正依赖外部实时数据的，才选 search。
5. 一次可以选多个，例如既查知识库又要计算就同时选 knowledge 和 compute。
6. 拿不准就多选，宁可多查也不要漏查。

只输出 JSON，不要输出其他内容：
{"routes": ["knowledge"], "reason": "一句话理由"}"""

EXTRACT_EXPRESSION_SYSTEM = """把用户的数学问题转换成一个可被求值的数学表达式。

要求：
- 只输出表达式本身，不要解释、不要单位、不要代码块标记。
- 支持的写法：四则运算 + - * /、括号、**、常用函数（sqrt/log/sin/cos/abs/round/min/max/factorial）、
  常量 pi 与 e、以及 sum(range(a, b)) 形式的求和。
- 逗号、千分位、货币符号都要去掉。

示例：
用户：123 乘以 456
输出：123 * 456

用户：1 加到 100 的和
输出：sum(range(1, 101))

用户：半径为 5 的圆面积（圆周率取 3.14）
输出：3.14 * 5 ** 2"""

SYNTHESIZE_SYSTEM = """你是最终回答生成器。根据下面各路专家收集到的资料回答用户的问题。

严格要求：
1. 只使用资料中明确包含的信息，不要补充你自己的知识或推测。
2. 只有在引用【知识库资料】时才用 [编号] 标注，编号要与资料中的 [n] 一致。
   计算结果和联网搜索结果**不要**加 [编号]——没有来源却标引用等于伪造溯源。
3. 若某路资料检索失败或没有内容，忽略它即可，不要向用户报错。
4. 若所有资料都不足以回答，如实说明"没有找到相关信息"，不要编造。
5. 用中文回答。"""


def _history(state: dict) -> list:
    """取历史对话（不含本轮问题），让专家能理解"再把这个结果加上 350"这类指代。

    这里特意保留 SystemMessage：长对话被压缩后，早期内容会以一条 system 摘要的
    形式存在。若在此处把它过滤掉，graph 架构就看不到那部分上下文了。
    """
    messages = state.get("messages") or []
    prior = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage))
    ]
    return prior[:-1]


def _parse_routes(raw: str) -> tuple[list[str], str]:
    """解析分类结果，解析失败时回退到全查（宁可多查也不要漏查）。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        routes = [r for r in data.get("routes", []) if r in ALL_ROUTES]
        if routes:
            return routes, str(data.get("reason", ""))
    except Exception as e:
        logger.warning("路由解析失败，回退到全查：%s", e)
    return [ROUTE_KNOWLEDGE, ROUTE_SEARCH], "路由解析失败，回退"


def classify(state: dict) -> dict:
    """判断问题需要哪些专家。"""
    question = state.get("question") or ""
    if not question:
        messages = state.get("messages") or []
        question = messages[-1].content if messages else ""

    response = get_model().invoke([
        SystemMessage(content=CLASSIFY_SYSTEM),
        *_history(state),
        HumanMessage(content=question),
    ])
    routes, reason = _parse_routes(response.content or "")

    logger.info("路由结果：%s（%s）", routes, reason)
    return {"question": question, "routes": routes, "route_reason": reason}


def route_targets(state: dict) -> list[str]:
    """条件边：把路由映射成要执行的专家节点。"""
    routes = state.get("routes") or []
    if not routes or ROUTE_DIRECT in routes and len(routes) == 1:
        return ["synthesize"]
    experts = (ROUTE_KNOWLEDGE, ROUTE_SEARCH, ROUTE_COMPUTE, ROUTE_TIME)
    return [r for r in routes if r in experts] or ["synthesize"]


def knowledge_expert(state: dict) -> dict:
    """知识库专家：检索本地文档，产出带编号的上下文与引用。"""
    question = state.get("question") or ""
    try:
        from agent.rag import get_retriever, scope_for_request

        # 与 ReAct 工具走同一个 scope 入口（含同一个 store），权限规则只有一处实现
        retriever = get_retriever()
        scope = scope_for_request(store=retriever.store)
        result = retriever.retrieve(question, scope=scope)
    except Exception as e:
        logger.warning("知识库检索失败：%s", e)
        return {"errors": [f"知识库检索失败：{type(e).__name__}"]}

    if not result.hit:
        return {"knowledge_context": "", "citations": []}

    return {
        "knowledge_context": result.context,
        "citations": [
            {
                "index": c.index,
                "source": c.source,
                "heading": c.heading,
                "score": c.score,
                "text": c.text,
            }
            for c in result.citations
        ],
    }


def search_expert(state: dict) -> dict:
    """联网搜索专家。"""
    from agent.tools import web_search

    question = state.get("question") or ""
    try:
        return {"search_result": web_search.invoke({"query": question})}
    except Exception as e:
        logger.warning("联网搜索失败：%s", e)
        return {"errors": [f"联网搜索失败：{type(e).__name__}"]}


def compute_expert(state: dict) -> dict:
    """计算专家：先把自然语言转成表达式，再交给 AST 安全求值。

    拆成两步是为了让"表达式构造"和"求值"解耦——求值部分仍然走
    AST 白名单，绝不因为架构变化就放松安全约束。
    """
    from agent.tools import calculate

    question = state.get("question") or ""
    try:
        response = get_model().invoke([
            SystemMessage(content=EXTRACT_EXPRESSION_SYSTEM),
            *_history(state),
            HumanMessage(content=question),
        ])
        expression = (response.content or "").strip().strip("`").strip()
        if not expression:
            return {"errors": ["未能从问题中构造出数学表达式"]}
        result = calculate.invoke({"expression": expression})
        return {"compute_result": f"表达式 `{expression}` 的计算结果：{result}"}
    except Exception as e:
        logger.warning("计算失败：%s", e)
        return {"errors": [f"计算失败：{type(e).__name__}"]}


def time_expert(state: dict) -> dict:
    """时间专家：返回当前本地时间。

    独立成专家是为了避免"现在几点"被当成实时信息去联网搜索——
    时间应当由本地工具直接给出，搜索反而会拿到过期或错误的时间。
    """
    from agent.tools import get_current_time

    try:
        return {"time_result": get_current_time.invoke({})}
    except Exception as e:
        logger.warning("获取时间失败：%s", e)
        return {"errors": [f"获取时间失败：{type(e).__name__}"]}


def synthesize(state: dict) -> dict:
    """汇总各路结果，生成最终答案。"""
    question = state.get("question") or ""
    sections = []

    knowledge = state.get("knowledge_context") or ""
    if knowledge:
        sections.append(f"【知识库资料】\n{knowledge}")

    search = state.get("search_result") or ""
    if search:
        sections.append(f"【联网搜索结果】\n{search}")

    compute = state.get("compute_result") or ""
    if compute:
        sections.append(f"【计算结果】\n{compute}")

    now = state.get("time_result") or ""
    if now:
        sections.append(f"【当前时间】\n{now}")

    errors = state.get("errors") or []
    if errors:
        sections.append("【失败提示】\n" + "\n".join(f"- {e}" for e in errors))

    prior = _history(state)

    if not sections:
        # 没有走任何专家：直接回答
        response = get_model().invoke([
            SystemMessage(content="你是任务助手，请用中文简洁准确地回答用户的问题。"),
            *prior,
            HumanMessage(content=question),
        ])
        return {"messages": [AIMessage(content=response.content or "")]}

    prompt = (
        f"{'=' * 40}\n"
        f"{chr(10).join(sections)}\n"
        f"{'=' * 40}\n\n"
        f"用户问题：{question}"
    )
    response = get_model().invoke([
        SystemMessage(content=SYNTHESIZE_SYSTEM),
        *prior,
        HumanMessage(content=prompt),
    ])
    return {"messages": [AIMessage(content=response.content or "")]}
