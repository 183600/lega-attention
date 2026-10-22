"""坏案例分析 —— 把预测错误的样本打印出来并归因，这是新改进思路的最重要来源。

只看聚合分数（EM/PPL）会掩盖"错在哪里"；错误样本逐一落盘后，本模块按
可检验的规则把每个错误归入一个类别，回答三个问题：
  * 标注本身有错吗？   gold 答案不在 prompt 里 → 构造/标注错误，不是模型的错；
  * 样本极其模糊吗？   needle 埋在极深位置 / 超长上下文 → 难度边界样本；
  * 是模型的盲区吗？   其余错误按机制归类：
      - 误抄干扰值      输出的错误数字来自上下文（检索到错误位置——注意力
                        落在同类干扰行上，multikey/vt 的典型失败模式）；
      - 幻觉数字        输出了上下文中根本不存在的数字；
      - 无数值回答      没有给出可解析的数字（格式/停不下来）；
      - fwe：漏答高频词 / 多答无关词 / 无法解析。

覆盖两类预测错误：
  * RULER 检索类（ruler_*.json 的 badcases，见 categorize_case）；
  * 语言建模类（eval_*.json 的 worst_windows：按窗口末位交叉熵挑出的最差
    next-token 预测，见 categorize_window）——PPL 只反映平均，最差窗口
    才暴露"模型究竟错了哪句话"。

跨变体对照（cross_variant_analysis）：把各变体的错误样本按键对齐后区分
"全体变体共识失败"（更可能是数据/构造/内在难度问题）与"相对基线的特异性
新错误"（这才是该方法自身的盲区），避免把共有难度误归因给某一方法。

改进方向合成（synthesize_*_improvements）：分析不止于分类——主导错误模式
按确定性阈值转成"触发证据 → 建议 → 下一步实验"的可执行条目（含归因轴
数据/构造、难度边界、模型/方法），进入逐案例 md、badcase_summary.json 与
汇总页，让坏案例真正成为提出新改进思路的来源。

用法:
  python -m lega.badcase --results_dir results/tiny_smoke      # 处理全部 ruler_*.json
  python -m lega.badcase --results_dir results/small --ruler ruler_lega_seed1337.json
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .ruler import _NUM_RE

# 必须与 RULER 打分器使用同一“完整数值 token”定义：这里若退化成
# \d{4,}，带符号/小数/科学计数法或短数字会被截成不同字符串，误抄会被
# 错归因为幻觉（归因错误会直接误导后续改进方向）。
_WORD_RE = re.compile(r"[a-z]+")

CATEGORY_CN = {
    "label_error": "标注/构造错误（gold 不在 prompt 中，不是模型的错）",
    "label_unknown": "标注无法核对（缺少 answer_in_prompt，不计入模型失败）",
    "no_answer": "无数值回答（模型未输出可解析的数字，格式盲区）",
    "copy_distractor": "误抄干扰值（检索到错误位置：输出了上下文中的干扰数字）",
    "hallucinated": "幻觉数字（输出了上下文中不存在的数字）",
    "fwe_missing": "fwe 漏答高频词",
    "fwe_spurious": "fwe 多答无关词",
    "fwe_no_parse": "fwe 无法解析（未输出小写词）",
    "fwe_no_ref": "fwe 缺少 answer_words 词表基线（无法归因）",
}

# 错误归属轴：直接回答"标注有错？样本极难？模型盲区？"三问。难度边界不覆盖
# 具体机制（copy/hallucination 等仍保留在 category 里），只是给出统一的归因轴。
AXIS_CN = {"数据/构造": "数据/构造", "难度边界": "难度边界（深位）",
           "模型/方法": "模型/方法", "校准": "校准", "评测/导出": "评测/导出"}

# 改进方向合成的显式阈值（可审计，不靠主观读报告）。
DOMINANT_SHARE = 1.0 / 3.0     # 单一类别占错误总数该份额以上视为主导模式
DEEP_FAIL_TOL = 0.5            # 深位(>0.75)失败占 niah 错误该比例以上视为难度边界主导
MIN_PATTERN_CASES = 3          # 少于该样本数时模式不可信，只提示扩大评测
MAX_ADVICE = 8                 # 单份报告最多输出的改进方向条数


def categorize_case(rec: Dict) -> Dict:
    """把一条坏案例归入错误类别；返回附加 category/explanation/difficulty 的副本。

    归因顺序：先排除"不是模型的错"（标注/构造错误），再区分难度边界样本与
    模型盲区，最后按输出内容定位具体机制。
    """
    r = dict(rec) if isinstance(rec, dict) else {}
    task = r.get("task")
    # 0) 难度标注：只有 niah 家族的 depth 是 needle 位置比例（0~1）；vt 的
    # meta.depth 是链深度（整数），不能按位置解释。先算好该标志，保证
    # 下面各归因分支（fwe/no_answer 等提前返回）的返回值都带 hard_sample。
    # depth 归一化为数值或 None：外部/旧格式 JSON 里的字符串 depth 不能
    # 让整个归因流程崩溃（一个坏样本不应毁掉报告）。
    # 非有限 depth（NaN/Inf）不是位置比例：NaN 会破坏 JSON，Inf 会被
    # 误判成“极深位置”的难度边界样本；两者都按缺失处理。
    depth = _finite_float(r.get("depth"))
    r["depth"] = depth
    positional = depth is not None and str(task or "").startswith("niah")
    r["hard_sample"] = bool(positional and depth > 0.75)
    # 1) 标注/构造自检：gold 答案必须真的在 prompt 里，否则错在数据侧。
    # 外部/旧格式可能完全没有该字段：缺少证据时既不能断言标注错误，也不能
    # 把答案当成幻觉归因给模型——单列为无法核对，避免污染模型失败统计。
    if isinstance(rec, dict) and task != "fwe":
        aip = r.get("answer_in_prompt")
        # 只有显式布尔值才是可核对的证据：0/1、"false" 等非布尔值
        # 既不能当成 True，也不能当成 False（bool 是 int 子类）。
        if not isinstance(aip, bool):
            r["category"] = "label_unknown"
            r["explanation"] = ("样例缺少 answer_in_prompt 标注自检字段，"
                                "或该字段不是布尔值；无法确认 gold 是否在 "
                                "prompt 中，不计入模型失败")
            return r
        if aip is False:
            r["category"] = "label_error"
            r["explanation"] = "gold 答案未出现在 prompt 中：样例构造或标注有错，" \
                               "该错误样本不应计入模型失败"
            return r
    # 2) fwe：按词表覆盖情况归因
    if task == "fwe":
        # 外部/旧格式 JSON 的 pred 可能是数字等非字符串类型：显式 str()，
        # 一个坏样本不应让整份归因报告崩溃
        pred_words = set(_WORD_RE.findall(_pred_text(r).lower()))
        top = _word_set(r.get("answer_words"))
        if not pred_words:
            r["category"] = "fwe_no_parse"
            r["explanation"] = "模型输出中没有可解析的小写词"
        elif not top:
            # 外部/旧格式 JSON 缺少 answer_words：没有词表就无法判断漏答/多答，
            # 不能默认“全中”并把所有输出词记成无关词（归因到数据/导出侧）。
            r["category"] = "fwe_no_ref"
            r["explanation"] = ("样例缺少 answer_words 词表基线（外部/旧格式），"
                                "无法归因漏答/多答")
        else:
            missing = sorted(top - pred_words)
            spurious = sorted(pred_words - top)
            if missing:
                r["category"] = "fwe_missing"
                r["missing_words"] = missing
                r["explanation"] = f"漏答 {len(missing)} 个高频词: {' '.join(missing[:8])}"
                if spurious:
                    r["spurious_words"] = spurious
                    r["explanation"] += f"；同时多答无关词: {' '.join(spurious[:8])}"
            else:
                r["category"] = "fwe_spurious"
                r["spurious_words"] = spurious
                r["explanation"] = f"高频词全中但多答 {len(spurious)} 个无关词（严格 EM 记错）"
        return r
    # 3) 数值任务：按输出的第一个数字归因（与 EM 计分口径一致）；
    # pred 可能是数字等非字符串类型，显式 str() 后再解析
    nums = _NUM_RE.findall(_pred_text(r))
    if not nums:
        r["category"] = "no_answer"
        r["explanation"] = "模型没有给出可解析的数字回答（可能复读问题或提前停止）"
        return r
    first = nums[0]
    ctx_nums = _num_set(r.get("context_numbers"))
    if first in ctx_nums:
        r["category"] = "copy_distractor"
        r["explanation"] = (f"输出了 {first}——这是上下文中的干扰值（gold={r.get('answer')}），"
                            "注意力检索到了错误位置（同类干扰行/错误链）")
    else:
        r["category"] = "hallucinated"
        r["explanation"] = (f"输出了 {first}——上下文中并不存在该数字"
                            f"（gold={r.get('answer')}），属于幻觉")
    return r


def case_axis(c: Dict) -> str:
    """错误的粗归因轴：数据/构造（标注错）→ 校准（排位对但置信低）→
    难度边界（深位）→ 评测/导出（解析/导出）→ 模型/方法。

    与 category 的机制归因互补：机制回答"错在哪一步"（误抄/幻觉/格式），
    轴回答"这笔账算谁的"（数据、难度还是模型）。深位样本即使机制是误抄，
    也先归入难度边界——但 category 不变，改进方向合成同时参考两者。
    """
    if c.get("category") == "label_error":
        return "数据/构造"
    if c.get("category") == "label_unknown":
        # 缺自检元数据是导出/旧格式问题，不是模型能力问题。
        return "评测/导出"
    if c.get("category") == "top1_correct":
        # 排位正确、置信度低是输出校准问题，不是检索失败；即使位于深位，
        # 改结构/加数据也不对症。synthesize_window_improvements 给该类别的
        # 建议 axis 是"校准"，这里必须先于 hard_sample 判断，否则同一批
        # 案例在分布表与改进建议里会被算到两条不同的账上。
        return "校准"
    if c.get("hard_sample"):
        return "难度边界"
    if c.get("category") in ("unparsed", "fwe_no_ref", "fwe_no_parse"):
        # fwe_no_parse 与 unparsed/fwe_no_ref 同属评测/导出环节（解析规则或
        # 导出问题），必须与 synthesize_ruler_improvements 给出的 axis 一致，
        # 否则同一批案例在分布表与改进建议里被算到两条不同的账上。
        return "评测/导出"
    return "模型/方法"


def _share(count: int, total: int) -> float:
    """单一类别占错误总数的份额（total=0 时返回 0，避免除零）。"""
    return (count / total) if total else 0.0


def _report_total(reported: int, counts: List[tuple]) -> int:
    """报告分母至少覆盖显式总数与各类别计数之和，坏 JSON 也不出现 >100%。"""
    total = _count_int(reported) or 0
    category_total = 0
    for _, n in counts:
        ni = _count_int(n)
        if ni is not None and ni > 0:
            category_total += ni
    return max(total, category_total)


def _advice(axis: str, priority: str, trigger: str, suggestion: str,
            next_experiment: str) -> Dict:
    """一条可执行的改进方向（字段固定，便于汇总去重与结构化消费）。"""
    return {"axis": axis, "priority": priority, "trigger": trigger,
            "suggestion": suggestion, "next_experiment": next_experiment}


def synthesize_ruler_improvements(ana: Dict) -> List[Dict]:
    """把 RULER 错误归因分布合成成可检验的改进方向（坏案例→新思路的闭环）。

    规则确定性、显式阈值（DOMINANT_SHARE/DEEP_FAIL_TOL/MIN_PATTERN_CASES）：
    与"错误归因是提出新改进思路的最重要来源"对应，避免分析止步于分类表。
    输出按优先级排序，可直接进报告与改进 backlog。
    """
    if not isinstance(ana, dict):
        return []
    n = _count_int(ana.get("n_badcases")) or 0
    counts = ana.get("category_counts")
    counts = counts if isinstance(counts, dict) else {}
    if n == 0:
        return [_advice(
            "评测", "低", "当前评测没有错误样本",
            "零错误不等于无盲区（样本太少或难度不足都会得到满分），"
            "不能据此宣称方法更强",
            "扩大 RULER 的 n_samples 与 ctx 档位、加深 needle 深度后再比较")]
    recs = []
    le = _count_int(ana.get("label_errors")) or 0
    lu = _count_int(counts.get("label_unknown")) or 0
    # 标注错误与“缺少自检证据”的样本都不属于模型失败：类别主导度与深位
    # 难度的分母都要剔除。label_unknown 不是可归因错误，若留在分母里会
    # 稀释真实模型盲区，甚至把“无法核对”包装成模型方向。
    model_n = max(0, n - le - lu)
    if le:
        recs.append(_advice(
            "数据/构造", "高", f"{le}/{n} 条错误的 gold 不在 prompt 中",
            "先修评测构造/标注再谈模型：构造错误会同时污染 EM、机制分析与"
            "跨变体对照",
            "核查 RULER needle 插入与答案抽取逻辑，补 answer_in_prompt 断言后重跑"))
    if lu:
        recs.append(_advice(
            "评测/导出", "中",
            f"{lu}/{n} 条错误缺少可核对的 answer_in_prompt 自检证据",
            "先补齐/修复导出再归因：无法确认 gold 是否在 prompt 中时，"
            "这些样本不能计入模型失败",
            "在 RULER 结果中写入布尔 answer_in_prompt 并重跑归因"))
    if model_n < MIN_PATTERN_CASES:
        if model_n > 0:
            recs.append(_advice(
                "评测", "低",
                f"仅 {model_n} 条可归因于模型的错误样本，模式不具统计意义",
                "先扩大评测规模再按归因下结论（单条错误可能是偶然）",
                "把 RULER n_samples/ctx_lens 提高到每格 ≥20 个样本"))
        # 样本不足时任何类别占比/深位比例都只是 1~2 条的偶然结果；继续
        # 输出“高优先级模型改进方向”会把噪声包装成证据。
        return recs[:MAX_ADVICE]
    deep = _finite_float(ana.get("deep_position_fail_ratio"))
    if (deep is not None and 0.0 <= deep <= 1.0 and deep >= DEEP_FAIL_TOL
            and model_n >= MIN_PATTERN_CASES):
        recs.append(_advice(
            "难度边界", "中", f"niah 失败中 {deep:.0%} 位于 needle 深位 (>0.75)",
            "失败由长程检索边界主导：平均 EM 会掩盖深位退化，应按 depth 分层报告",
            "增加深位训练样本/更长训练上下文；对门控变体检查深位处门控分布，"
            "确认长度项 b·log n 校准是否正确"))
    cd = _count_int(counts.get("copy_distractor")) or 0
    if _share(cd, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "模型/方法", "高",
            f"误抄干扰值占模型错误 {cd}/{model_n}（检索到错误位置，不是没读到）",
            "定位失败：注意力落在同类干扰行/错误链上；先用机制分析确认"
            "（misleading_rate、P(hit|集中)）再改结构",
            "在失败样例上对比注意力落点；若属位置混淆，优先改位置编码或"
            "干扰项抑制，再考虑门控"))
    ha = _count_int(counts.get("hallucinated")) or 0
    if _share(ha, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "模型/方法", "高",
            f"幻觉数字占模型错误 {ha}/{model_n}（输出上下文中不存在的值）",
            "模型未 grounding 到上下文：检查训练/评测长度分布是否错配、"
            "是否存在长程复制的系统性衰减",
            "统计训练样本中长上下文与复制型任务占比；对门控变体比较 "
            "gate_mean 是否过低（该写入的信息被压制）"))
    na = _count_int(counts.get("no_answer")) or 0
    if _share(na, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "模型/方法", "中",
            f"无数值回答占模型错误 {na}/{model_n}（复读问题/提前停止）",
            "先排除解码/格式问题再改模型，否则真实能力会被低估",
            "检查 prompt 模板与 max_new_tokens/停止条件；贪心解码复测确认"
            "输出格式"))
    for cat, ax, hi, trig, sugg, nxt in (
        ("fwe_missing", "模型/方法", "中", "fwe 漏答高频词主导（召回不足）",
         "长上下文稀释导致高频词被漏掉：先确认不是分词/词形归一问题",
         "把 fwe 词出现次数 k 分层报告；核对 k≥3 高频词召回与上下文长度的关系"),
        ("fwe_spurious", "模型/方法", "中", "fwe 多答无关词主导（精确率/停不下来）",
         "严格 EM 下多答即错：先排除解码缺乏停止条件，再考虑集合口径并报",
         "检查解码 stop 条件与输出词表约束；同时给出集合评分对照 EM"),
        ("fwe_no_parse", "评测/导出", "中", "fwe 输出无法解析主导",
         "模型没输出小写词：可能是格式导出或解析规则问题而非能力问题",
         "把模型原文随结果落盘复核；核对解析正则与 prompt 模板"),
        ("fwe_no_ref", "评测/导出", "中", "fwe 缺少 answer_words 词表基线",
         "没有参考词表时漏答/多答无法归因；这是导出/旧格式问题，"
         "不能据此评价模型召回或精确率",
         "在 fwe 结果中补写 answer_words/参考词表并重跑归因，"
         "确认旧产物不是数据导出缺口"),
    ):
        c = _count_int(counts.get(cat)) or 0
        if _share(c, model_n) >= DOMINANT_SHARE:
            recs.append(_advice(ax, hi, f"{trig}（{c}/{model_n}）", sugg, nxt))
    return recs[:MAX_ADVICE]


def analyze_badcases(badcases: List[Dict], variant: str = None) -> Dict:
    """坏案例集合 → 归因分布 + 难度/上下文切片 + 结构化个案列表 + 改进方向。"""
    # 外部/旧格式 JSON 的 badcases 列表可能混入 null/标量；跳过坏样本，
    # 不让单条记录中断整份归因（与 process_* 的顶层类型校验同口径）。
    cases = [categorize_case(c) for c in (badcases if isinstance(badcases, list)
                                          else []) if isinstance(c, dict)]
    cat_counts = Counter(c["category"] for c in cases)
    axis_counts = Counter(case_axis(c) for c in cases)
    by_task_ctx: Dict[str, Dict] = {}
    for c in cases:
        # 记录可能缺少 task（外部/旧格式）：用 .get，一个坏样本不应中断聚合
        key = f"{c.get('task')}@{c.get('ctx')}"
        by_task_ctx.setdefault(key, Counter())[c["category"]] += 1
    hard = sum(1 for c in cases if c.get("hard_sample"))
    label_errors = cat_counts.get("label_error", 0)
    # 深度信息只在 niah 家族存在（vt meta.depth 是链深度而非位置）；给出
    # 深位（>0.75）失败占比用于判断"是样本极难，还是模型系统性盲区"
    with_depth = [c for c in cases if c.get("depth") is not None
                  and str(c.get("task") or "").startswith("niah")
                  and c.get("category") not in ("label_error", "label_unknown")]
    deep_fail = sum(1 for c in with_depth if c["depth"] > 0.75)
    out = {
        "variant": variant,
        "n_badcases": len(cases),
        "category_counts": dict(cat_counts),
        "category_cn": {k: CATEGORY_CN.get(k, k) for k in cat_counts},
        "axis_counts": dict(axis_counts),
        "axis_cn": {k: AXIS_CN.get(k, k) for k in axis_counts},
        "by_task_ctx": {k: dict(v) for k, v in by_task_ctx.items()},
        "hard_sample_ratio": (hard / len(cases)) if cases else None,
        "deep_position_fail_ratio": (deep_fail / len(with_depth)
                                      if with_depth else None),
        "label_errors": label_errors,
        "cases": cases,
    }
    # 坏案例分析不能止步于分类：同一份分布直接合成可检验的改进方向。
    out["recommendations"] = synthesize_ruler_improvements(out)
    return out


def format_case(idx: int, c: Dict) -> List[str]:
    """单个坏案例的 Markdown 行（问题/gold 摘录/预测/归因）。"""
    c = c if isinstance(c, dict) else {}
    depth = _finite_float(c.get("depth"))
    depth_s = f" depth={depth:.2f}" if depth is not None else ""
    lines = [f"### 案例 {idx}: {c.get('task')} @ctx={c.get('ctx')}"
             + depth_s
             + (" 【难度边界样本：needle 靠后】" if c.get("hard_sample") else "")]
    q = str(c.get("question") or "").strip().replace("\n", " ⏎ ")
    if len(q) > 220:
        q = q[:220] + "…"
    lines.append(f"- 问题: {q}")
    if c.get("needle_excerpt"):
        ne = str(c["needle_excerpt"]).replace("\n", " ⏎ ")
        lines.append(f"- gold 证据摘录: {ne}")
    pred = _pred_text(c).strip().replace("\n", " ⏎ ")
    if len(pred) > 160:
        pred = pred[:160] + "…"
    lines.append(f"- 模型输出: {pred!r}")
    lines.append(f"- gold 答案: {c.get('answer')!r}")
    cat = c.get("category")
    cat_s = CATEGORY_CN.get(cat, cat) if isinstance(cat, str) else "?"
    lines.append(f"- 归因: **{cat_s}** —— {c.get('explanation', '')}")
    return lines


def _render_recommendations(md: List[str], recs: List[Dict],
                            title: str = "## 改进方向（坏案例驱动）") -> None:
    """把改进方向渲染成 Markdown 小节（触发证据→建议→下一步实验）。"""
    if not isinstance(recs, list) or not recs:
        return
    md += ["", title, ""]
    for i, r in enumerate(recs, 1):
        if not isinstance(r, dict):
            continue
        md.append(f"{i}. **[{r.get('axis')}·{r.get('priority')}]** "
                  f"触发：{r.get('trigger')}")
        md.append(f"   - 建议：{r.get('suggestion')}")
        md.append(f"   - 下一步实验：{r.get('next_experiment')}")


def write_badcase_md(analysis: Dict, path: str, max_cases: int = 40) -> None:
    """写人类可读的 Markdown 报告（归因分布 + 改进方向 + 逐案例详情）。"""
    analysis = analysis if isinstance(analysis, dict) else {}
    md = [f"# 坏案例分析：{analysis.get('variant') or '-'}", ""]
    n_badcases = _count_int(analysis.get("n_badcases")) or 0
    counts = analysis.get("category_counts")
    counts = counts if isinstance(counts, dict) else {}
    clean = []
    for cat, n in counts.items():
        n = _count_int(n)
        if n is None:
            continue
        clean.append((cat, n))
    total = _report_total(n_badcases, clean)
    if not total:
        md.append("本变体无预测错误样本（或评测被跳过）。")
    else:
        md += [f"共 {total} 条错误样本；归因分布：", "",
               "| 类别 | 数量 | 占比 |", "|---|---|---|"]
        for cat, n in sorted(clean, key=lambda kv: -kv[1]):
            md.append(f"| {CATEGORY_CN.get(cat, cat)} | {n} | {n / total:.1%} |")
        axis = analysis.get("axis_counts")
        axis = axis if isinstance(axis, dict) else {}
        if axis:
            clean_axis = []
            for k, v in axis.items():
                n = _count_int(v)
                if n is not None:
                    clean_axis.append((k, n))
            axis_s = "；".join(f"{AXIS_CN.get(k, k)}×{v}" for k, v in
                              sorted(clean_axis, key=lambda kv: -kv[1]))
            md += ["", f"粗归因轴（数据/难度/模型）：{axis_s}。"]
        deep_ratio = _finite_float(analysis.get("deep_position_fail_ratio"))
        if deep_ratio is not None:
            md += ["",
                   f"needle 深位（>0.75）失败占 niah 类错误的 "
                   f"{deep_ratio:.0%}——"
                   "高占比说明是长程检索边界问题，低占比说明是系统性盲区。"]
        le_report = _count_int(analysis.get("label_errors")) or 0
        if le_report:
            md += ["", f"⚠ 有 {le_report} 条 gold 答案不在 prompt 中："
                        "样例构造或标注有错，须先修 RULER 构造再谈模型。"]
        md.append("")
        cases = analysis.get("cases")
        cases = cases if isinstance(cases, list) else []
        for i, c in enumerate(cases[:max_cases], 1):
            md += [""] + format_case(i, c)
        if total > max_cases:
            md.append(f"\n（仅展示前 {max_cases} 条，完整结构化数据见同目录 "
                      "badcase_summary.json）")
    # 改进方向对"零错误"同样给出（提示零错误的解释边界）
    _render_recommendations(md, analysis.get("recommendations") or [])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ---------------------------------------------------------------- 语言建模坏案例
# （PPL 最差窗口：next-token 预测错误的样本，字段来自 evaluate.collect_worst_windows）

WINDOW_CATEGORY_CN = {
    "repetition_attractor": "复读吸引子（低置信下首选续写退化为抄上下文已有串）",
    "confident_bias": "高置信错答（top1 概率高却错，系统性偏差=模型盲区）",
    "low_confidence": "低置信（top1 概率低：样本本身多解/噪声，或训练不足）",
    "top1_correct": "top1 正确但置信低（排位对、概率小，属校准问题）",
    "unparsed": "无预测分布（旧格式或导出失败，无法归因）",
    "label_unknown": "gold 缺失（旧格式/导出失败，无法判断预测正确性）",
}

# top1 概率 ≥ 阈值仍错 → 高置信错答（系统性偏差）；< 阈值 → 模型自己也不确定
TOP1_CONF_TOL = 0.5


def _token_text(value):
    """预测词项的文本形式；对象/容器/None 不是可解码 token，返回 None。"""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    return str(value)


def _window_top1(rec: Dict):
    """取出窗口 top1 预测 (文本, 概率)；缺失/格式异常返回 (None, None)。"""
    top5 = rec.get("pred_top5")
    if not isinstance(top5, (list, tuple)) or not top5:
        return None, None
    item = top5[0]
    if isinstance(item, (list, tuple)):
        if len(item) < 2:
            return None, None
        token = _token_text(item[0])
        if token is None:
            return None, None
        # 概率必须是有限数：inf 会被当成“高置信”伪造 confident_bias，
        # NaN 还会让报告 JSON 非法。
        return token, _finite_float(item[1])
    token = _token_text(item)
    return (token, None) if token is not None else (None, None)


def categorize_window(rec: Dict) -> Dict:
    """把一条语言建模坏案例（PPL 最差窗口）按可检验规则归因。

    规则（显式、顺序确定）：
      1. 无 top5 预测 → unparsed；
      2. gold 缺失/空串 → label_unknown（无真值则无从判断对错，不能把
         高置信预测误归因为模型盲区，也不能让它污染 gold 可见性比例）；
      3. top1 即 gold → top1_correct（排位正确但概率低，校准问题）；
      4. top1 概率 ≥ TOP1_CONF_TOL 且错 → confident_bias（高置信错答，
         最值得作为改进线索的系统性盲区）；若该 token 同时是上下文里已有的
         串，标记 repeats_context 并在解释中注明（复读是高置信错答的一种机制）；
      5. top1（非空白、≥2 字符）出现在 context_tail 中且不是 gold →
         repetition_attractor（低置信下的复读回环：模型退化为抄上下文）；
      6. 其余 → low_confidence（模型在多个候选间平均：样本可能本身多解/
         噪声，或训练不足）。
    附加布尔 gold_in_context / gold_in_top5 区分"上下文里出现过 gold 但
    模型没复制"（复制/检索失败）与"gold 从未出现"（真正的预测失败/长尾）。
    gold_in_context 优先使用采集时按完整窗口判定的布尔（evaluate 写入）；
    旧格式记录没有该字段时退化为 context_tail（只有窗口尾部 160 字符）检查，
    此时"未出现"只代表尾部未出现；两者都没有时记 None（未知），不伪造。
    """
    r = dict(rec) if isinstance(rec, dict) else {}
    top1, p1 = _window_top1(r)
    gold = r.get("gold_next")
    gold_txt = gold if isinstance(gold, str) else ("" if gold is None else str(gold))
    tail_raw = r.get("context_tail")
    tail = (tail_raw if isinstance(tail_raw, str)
            else ("" if tail_raw is None else str(tail_raw)))
    top5_txt = []
    top5_raw = r.get("pred_top5")
    for item in (top5_raw if isinstance(top5_raw, (list, tuple)) else []):
        if isinstance(item, (list, tuple)):
            token = _token_text(item[0]) if item else None
        else:
            token = _token_text(item)
        if token is not None:
            top5_txt.append(token)
    r["top1"] = top1
    r["top1_prob"] = p1
    gold_missing = not gold_txt
    recorded_ctx = r.get("gold_in_context")
    if gold_missing:
        # 没有 gold 就没有“可见性”：旧记录里即使带布尔也不能赋予意义，
        # 否则会以 unknown/present 的身份混进 gold_in_context_rate 分母。
        r["gold_in_context"] = None
    elif isinstance(recorded_ctx, bool):
        r["gold_in_context"] = recorded_ctx
    elif isinstance(tail_raw, str):
        # 旧格式没有完整窗口布尔：只有确实拿到尾部文本时才可退回尾部检查
        # （“尾部未出现”仍不等于“窗口内从未出现”）。
        r["gold_in_context"] = gold_txt in tail
    else:
        # 既无完整窗口布尔也无尾部文本：无法判断，不能伪造成“从未出现”。
        r["gold_in_context"] = None
    r["gold_in_top5"] = None if gold_missing else (gold_txt in top5_txt)
    if top1 is None:
        r["category"] = "unparsed"
        r["explanation"] = "窗口记录中没有可用的 top5 预测分布"
        return r
    if gold_missing:
        r["category"] = "label_unknown"
        r["explanation"] = ("窗口记录缺少可用的 gold_next：无法判断预测是否"
                            "正确，不能归因为模型高置信盲区或样本难度")
        return r
    pick = top1.strip()
    r["repeats_context"] = bool(pick and len(pick) >= 2 and pick in tail
                                and gold_txt != top1)
    if gold_txt and top1 == gold_txt:
        # gold 缺失/为空串时不能判"top1 正确"：空串相等会把无 gold 的记录
        # 误标为命中的校准样本（gold 不存在则无从谈"预测正确"）。
        r["category"] = "top1_correct"
        r["explanation"] = (f"top1 与 gold 一致但概率仅 {p1:.3f}（低置信正确），"
                            "属输出校准问题而非检索错误"
                            if p1 is not None else "top1 与 gold 一致")
    elif p1 is not None and p1 >= TOP1_CONF_TOL and gold_txt != top1:
        r["category"] = "confident_bias"
        r["explanation"] = (
            f"top1={top1!r} 概率 {p1:.2f}（高置信）却非 gold={gold_txt!r}；"
            + ("且 top1 是上下文里已出现的串（高置信复读）；"
               if r["repeats_context"] else "")
            + ("gold 在上下文中出现过但未被复制（检索/复制失败）"
               if r["gold_in_context"] is True else
               "gold 在上下文中未出现（真正的预测失败，长尾/分布外）"
               if r["gold_in_context"] is False else
               "gold 是否在上下文中出现未记录（无法区分检索失败与长尾）"))
    elif r["repeats_context"]:
        r["category"] = "repetition_attractor"
        r["explanation"] = (f"top1={top1!r} 是上下文里已出现的串（复读），"
                            f"而 gold 是 {gold_txt!r}——重复/回环失败模式")
    else:
        r["category"] = "low_confidence"
        r["explanation"] = (
            (f"top1={top1!r} 概率仅 {p1:.2f}（多候选平均）"
             if p1 is not None else f"top1={top1!r} 无概率信息")
            + ("，且 gold 在上下文中出现过（模型没抄到）"
               if r["gold_in_context"] is True else
               "，gold 未在上下文中出现（样本本身难/长尾）"
               if r["gold_in_context"] is False else
               "，gold 是否在上下文中出现未记录（无法判断样本难度）"))
    return r


def _word_set(raw) -> set:
    if isinstance(raw, str):
        return set(_WORD_RE.findall(raw.lower()))
    if not isinstance(raw, (list, tuple, set)):
        return set()
    out = set()
    for v in raw:
        if v is not None:
            out.update(_WORD_RE.findall(str(v).lower()))
    return out


def _num_set(raw) -> set:
    if isinstance(raw, str):
        return set(_NUM_RE.findall(raw))
    if not isinstance(raw, (list, tuple, set)):
        return set()
    out = set()
    for v in raw:
        if v is not None:
            out.update(_NUM_RE.findall(str(v)))
    return out


def _finite_float(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _pred_text(rec: Dict) -> str:
    """预测文本；数值 0 是合法输出，不能因 falsy 被当成空串。

    ``str(rec.get("pred") or "")`` 会把 0/0.0 变成空串，使模型真实输出的
    数字 0 被误归因为 no_answer。bool/容器/None 不是有效预测文本，按缺失
    处理（False 也不应被解析成单词 false）。
    """
    pred = rec.get("pred") if isinstance(rec, dict) else None
    if pred is None or isinstance(pred, (bool, np.bool_, dict, list, tuple, set)):
        return ""
    return str(pred)


def _safe_int(v):
    f = _finite_float(v)
    if f is None or not f.is_integer():
        return None
    try:
        return int(f)
    except (OverflowError, ValueError):
        return None


def _count_int(v) -> Optional[int]:
    """计数/分母 → 非负整数；bool、小数、负数与非有限值一律按无效处理。

    ``int(True) is 1``、``int(3.9) is 3`` 会把坏 JSON 里的布尔/小数计数
    悄悄变成错误样本数量，报告占比随之失真。整数值浮点/数字串仍接受
    （旧版 summary 可能把整数写成了 3.0 或 "3"）。
    """
    f = _finite_float(v)
    if f is None or f < 0 or not f.is_integer():
        return None
    try:
        return int(f)
    except (OverflowError, ValueError):
        return None


def _median(vals):
    out = []
    for v in vals:
        f = _finite_float(v)
        if f is not None:
            out.append(f)
    return statistics.median(out) if out else None


def _mean(vals):
    # 与 utils.mean_std 同口径：剔除 None、bool、非数值与非有限值（NaN/Inf）。
    # 发散 run 的窗口 loss 可能是 NaN，若混入均值会把 NaN 写进 JSON（非法）并
    # 污染报告；旧格式/坏数据里的 "n/a" 等字符串同样跳过而不是让整份报告崩掉。
    # bool 是 int 子类，float(True)=1.0，不拒绝会把 "loss": true 伪造成 1.0。
    out = []
    for v in vals:
        f = _finite_float(v)
        if f is not None:
            out.append(f)
    return float(sum(out) / len(out)) if out else None


def synthesize_window_improvements(ana: Dict) -> List[Dict]:
    """把最差窗口归因合成成可检验的改进方向（PPL 坏案例→新思路的闭环）。

    关键区分是"gold 在上下文中却没抄对"（复制/检索失败，模型信息可得）与
    "gold 从未出现"（真正的预测失败/长尾，通常只能靠数据/容量）。
    """
    if not isinstance(ana, dict):
        return []
    n = _count_int(ana.get("n_windows")) or 0
    counts = ana.get("category_counts")
    counts = counts if isinstance(counts, dict) else {}
    if n == 0:
        return [_advice(
            "评测", "低", "最差窗口记录为空",
            "PPL 平均分无法替代错误检查：没有 worst_n 转储时无法定位模型"
            "错在哪句话",
            "评测时开启 --worst_n（如 5）并复核导出格式")]
    recs = []
    up = _count_int(counts.get("unparsed")) or 0
    lu = _count_int(counts.get("label_unknown")) or 0
    # unparsed/label_unknown 是导出/真值缺失，不是模型失败；它们既不能
    # 进入模式占比的分母，也不能因总数够大就让模式结论通过统计门槛。
    model_n = max(0, n - up - lu)
    if up:
        recs.append(_advice(
            "评测/导出", "中", f"{up}/{n} 个窗口没有可用的 top5 预测",
            "先修评测导出再归因：旧格式/导出失败会让错误模式被静默丢弃",
            "核对 evaluate 的 worst_windows 字段与版本兼容；重跑评测导出"))
    if lu:
        recs.append(_advice(
            "评测/导出", "中", f"{lu}/{n} 个窗口缺少可用的 gold_next",
            "无真值无法判断预测对错：先修复/重导 gold 再归因，不能把这些"
            "窗口算成高置信错答或长尾失败",
            "核对 evaluate 的 worst_windows 是否完整写入 gold_next；"
            "用当前版本重跑评测并复核导出"))
    if model_n < MIN_PATTERN_CASES:
        if model_n > 0:
            recs.append(_advice(
                "评测", "低",
                f"仅 {model_n} 个可归因于模型的最差窗口，模式不具统计意义",
                "先扩大 worst_n/评测窗口数再按归因下结论",
                "把 --worst_n 提高到 ≥10 并跨更多窗口采样"))
        # 同 synthesize_ruler_improvements：窗口太少时不下高/中优先级结论。
        return recs[:MAX_ADVICE]
    cb = _count_int(counts.get("confident_bias")) or 0
    if _share(cb, model_n) >= DOMINANT_SHARE:
        # 分支判定只看 confident_bias 这批窗口：全局 gold_in_context_rate 会把
        # 其它类别（如 repetition_attractor）的可见性算进来，可能把"这批窗口
        # 的 gold 根本没见过"错判成"模型看得到却没抄对"。
        cb_gic = [bool(c.get("gold_in_context"))
                  for c in (ana.get("cases") or [])
                  if c.get("category") == "confident_bias"
                  and c.get("gold_in_context") is not None]
        if cb_gic:
            gic = sum(cb_gic) / len(cb_gic)
            scope = "其中"
        else:
            # 旧格式窗口缺 gold_in_context 字段：退回全局率但显式标注口径，
            # 不把它冒充成这批窗口自身的比例。
            gic = ana.get("gold_in_context_rate")
            scope = "全部最差窗口中"
        if gic is None:
            # 旧格式/导出缺失 gold_in_context：既不能断言“看得到没抄对”，
            # 也不能把未知当成“gold 从未出现”，先补齐证据再归因。
            recs.append(_advice(
                "评测/导出", "中",
                f"高置信错答占 {cb}/{model_n}，但缺少 gold_in_context 记录，"
                "无法判断是检索失败还是长尾预测失败",
                "先补齐窗口级 gold 可见性再决定改结构还是扩数据；把未知"
                "当成“从未出现”会低估复制/检索机制的问题",
                "用当前 evaluate 重跑 worst_windows（写入完整窗口的 "
                "gold_in_context），或从原窗口重新计算 gold 是否出现"))
        elif gic >= 0.5:
            recs.append(_advice(
                "模型/方法", "高",
                f"高置信错答占 {cb}/{model_n}，且{scope} {gic:.0%} 的 gold 在上下文中",
                "模型看得到却没抄对：优先检查注意力是否落在 gold 跨度"
                "（复制/检索失败而非知识缺失），这类失败最可能通过结构调整改善",
                "对这批窗口检查 top-1 注意力位置与 gold 跨度的重叠；区分"
                "位置编码混淆与干扰串抑制问题后再改注意力"))
        else:
            recs.append(_advice(
                "模型/方法", "高",
                f"高置信错答占 {cb}/{model_n}，且 gold 大多不在上下文中",
                "真正的预测失败/长尾：结构改进收益有限，优先扩大数据"
                "（或先按训练曲线判断是否过拟合再决定正则化）",
                "抽样人工复核 gold 可预测性；对照 curve_diagnosis.json 的"
                "状态决定是增数据还是加正则化"))
    rep = _count_int(counts.get("repetition_attractor")) or 0
    if _share(rep, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "模型/方法", "中",
            f"复读吸引子占 {rep}/{model_n}（低置信下退化为抄上下文）",
            "检查训练语料重复率与重复上下文下的写入行为，而不是只加解码惩罚",
            "核对 datacheck 报告的重复样本率；对门控变体比较重复上下文中的"
            "gate_mean（是否错误压制了写入）"))
    lc = _count_int(counts.get("low_confidence")) or 0
    if _share(lc, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "模型/方法", "中",
            f"低置信占 {lc}/{model_n}（模型在多个候选间平均）",
            "先按训练曲线判断是欠拟合（调学习率/数据管道）还是样本本身多解，"
            "再决定扩容或增加数据",
            "对照 curve_diagnosis.json：欠拟合→调学习率/数据；正常收敛→"
            "扩充数据或接受该样本的歧义性"))
    tc = _count_int(counts.get("top1_correct")) or 0
    if _share(tc, model_n) >= DOMINANT_SHARE:
        recs.append(_advice(
            "校准", "低",
            f"top1 正确但低置信占 {tc}/{model_n}",
            "排位正确、概率低属校准问题而非检索问题：温度缩放/top-k 校准"
            "优先于改结构",
            "在 held-out 窗口上做温度扫描，报告校准后的 NLL/ECE 变化"))
    return recs[:MAX_ADVICE]


def _known_bool_rate(cases: List[Dict], key: str) -> Optional[float]:
    """仅在字段被明确记录为布尔时计算比例；未知记录不进入分母。"""
    known = [c.get(key) for c in cases if isinstance(c.get(key), bool)]
    return (sum(1 for v in known if v) / len(known)) if known else None


def analyze_worst_windows(windows: List[Dict], variant: str = None) -> Dict:
    """最差窗口集合 → 归因分布 + 校准/复制统计 + 个案列表 + 改进方向。"""
    # 同 analyze_badcases：列表内的 null/数组等坏记录按损坏样本跳过。
    cases = [categorize_window(w) for w in (windows if isinstance(windows, list)
                                           else []) if isinstance(w, dict)]
    cat_counts = Counter(c["category"] for c in cases)
    axis_counts = Counter(case_axis(c) for c in cases)
    distances = [c.get("gold_distance_tokens") for c in cases
                 if isinstance(c.get("gold_distance_tokens"), (int, float))]
    out = {
        "variant": variant,
        "n_windows": len(cases),
        "category_counts": dict(cat_counts),
        "category_cn": {k: WINDOW_CATEGORY_CN.get(k, k) for k in cat_counts},
        "axis_counts": dict(axis_counts),
        "axis_cn": {k: AXIS_CN.get(k, k) for k in axis_counts},
        "mean_top1_prob": _mean([c.get("top1_prob") for c in cases]),
        "repeats_context_rate": (sum(1 for c in cases
                                     if c.get("repeats_context"))
                                 / len(cases)) if cases else None,
        # 与 gold_in_context 同口径：gold 缺失的窗口记为 None，不能把
        # “无真值”算成“gold 未进 top5”而拉低复制/检索成功率。
        "gold_in_top5_rate": _known_bool_rate(cases, "gold_in_top5"),
        "gold_in_context_rate": _known_bool_rate(cases, "gold_in_context"),
        # gold 距预测位的 token 距离（只在 gold 出现时定义）：区分“远处检索
        # 失败”与“近处复制失败”，后者更像注意力/复制机制问题。
        "median_gold_distance_tokens": _median(distances),
        "mean_loss": _mean([c.get("loss") for c in cases]),
        "cases": cases,
    }
    out["recommendations"] = synthesize_window_improvements(out)
    return out


def format_window_case(idx: int, c: Dict) -> List[str]:
    """单个最差窗口的 Markdown 行（上下文尾部/模型 top-1/真实续写/归因）。"""
    c = c if isinstance(c, dict) else {}
    p1 = c.get("top1_prob")
    where = (f"窗口 #{c.get('window')}" +
             (f"（{c['split']} 集）" if c.get("split") else ""))
    loss = c.get("loss")
    # loss 可能缺失、显式为 null、bool 或超出 float 范围（旧格式/外部 JSON）：
    # 一律显示 '-' 而不是崩溃，也不能把 True 显示成 1.000。
    loss_f = _finite_float(loss)
    loss_s = f"{loss_f:.3f}" if loss_f is not None else "-"
    lines = [f"### 案例 {idx}: {where}　loss={loss_s}"]
    tail = str(c.get("context_tail") or "").replace("\n", " ⏎ ")[-160:]
    lines.append(f"- 上下文尾部: …{tail}")
    top1 = c.get("top1")
    p1_f = _finite_float(p1)
    p1_s = f"（p={p1_f:.3f}）" if p1_f is not None else ""
    top_s = f"{top1!r}" + p1_s
    lines.append(f"- 模型 top-1: {top_s}")
    gic = c.get("gold_in_context")
    gic_note = ("　（gold 在上下文中出现过）" if gic is True else
                "　（gold 未在上下文中出现）" if gic is False else
                "　（gold 是否在上下文中出现未记录）")
    lines.append(f"- 真实续写(gold): {c.get('gold_next')!r}{gic_note}")
    pos = c.get("gold_last_pos")
    dist = c.get("gold_distance_tokens")
    pos_i = _safe_int(pos)
    dist_i = _safe_int(dist)
    if pos_i is not None and dist_i is not None:
        occ_i = _safe_int(c.get("gold_occurrences"))
        lines.append(f"- gold 位置: 窗口内第 {pos_i} 个 token，距预测位 "
                     f"{dist_i} tokens"
                     + (f"（共出现 {occ_i} 次，取最后一次）"
                        if occ_i is not None and occ_i > 1 else ""))
        snippet = c.get("gold_context_excerpt")
        if snippet:
            lines.append(f"- gold 附近上下文: …{snippet}…")
    cat = c.get("category")
    cat_s = WINDOW_CATEGORY_CN.get(cat, cat) if isinstance(cat, str) else "?"
    lines.append(f"- 归因: **{cat_s}** —— {c.get('explanation', '')}")
    return lines


def write_worstcase_md(analysis: Dict, path: str, max_cases: int = 10) -> None:
    """写语言建模坏案例的 Markdown 报告（含归因分布与逐案例详情）。"""
    analysis = analysis if isinstance(analysis, dict) else {}
    md = [f"# 语言建模最差窗口归因：{analysis.get('variant') or '-'}", ""]
    n_windows = _count_int(analysis.get("n_windows")) or 0
    counts = analysis.get("category_counts")
    counts = counts if isinstance(counts, dict) else {}
    clean = []
    for cat, n in counts.items():
        n = _count_int(n)
        if n is None:
            continue
        clean.append((cat, n))
    total = _report_total(n_windows, clean)
    if not total:
        md.append("本变体无最差窗口记录（评测被跳过或 --worst_n=0）。")
    else:
        md += [f"共 {total} 个最差窗口；归因分布：", "",
               "| 类别 | 数量 | 占比 |", "|---|---|---|"]
        for cat, n in sorted(clean, key=lambda kv: -kv[1]):
            md.append(f"| {WINDOW_CATEGORY_CN.get(cat, cat)} | {n} | {n / total:.1%} |")
        stats = []
        mtp = _finite_float(analysis.get("mean_top1_prob"))
        if mtp is not None:
            stats.append(f"平均 top-1 概率={mtp:.3f}")
        gic = _finite_float(analysis.get("gold_in_context_rate"))
        if gic is not None:
            stats.append(f"gold 出现在上下文中的比例={gic:.0%}")
        git5 = _finite_float(analysis.get("gold_in_top5_rate"))
        if git5 is not None:
            stats.append(f"gold 进入 top-5 的比例={git5:.0%}")
        dist_i = _safe_int(analysis.get("median_gold_distance_tokens"))
        if dist_i is not None:
            stats.append(f"gold 距预测位的中位距离={dist_i} tokens")
        if stats:
            md += ["", "；".join(stats) + "。"]
        cases = analysis.get("cases")
        cases = cases if isinstance(cases, list) else []
        not_ctx = sum(1 for c in cases
                      if isinstance(c, dict) and c.get("gold_in_context") is False)
        unknown_ctx = sum(1 for c in cases
                          if not (isinstance(c, dict)
                                  and isinstance(c.get("gold_in_context"), bool)))
        if not_ctx:
            md += ["", f"其中 {not_ctx} 个窗口的 gold 从未出现在上下文中——"
                       "这些更接近真正的预测失败/长尾，应与“抄不到”的复制失败分开看。"]
        if unknown_ctx:
            md += ["", f"另有 {unknown_ctx} 个窗口缺少 gold_in_context 记录——"
                       "无法判断 gold 是否在上下文中，不能计入“从未出现”。"]
        md.append("")
        for i, c in enumerate(cases[:max_cases], 1):
            md += [""] + format_window_case(i, c)
        if n_windows > max_cases:
            md.append(f"\n（仅展示 loss 最高的前 {max_cases} 个窗口，"
                      "完整数据见 eval_*.json 的 worst_windows）")
    _render_recommendations(md, analysis.get("recommendations") or [],
                            title="## 改进方向（最差窗口驱动）")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def process_eval_json(eval_path: str, out_dir: Optional[str] = None,
                      write_md: bool = True,
                      payload: Optional[Dict] = None) -> Dict:
    """处理一个 eval_*.json 的 worst_windows：归因分析 + 写 md（与源文件同目录）。

    payload 已由调用方解析时可直接传入；为 None 时自行读取。
    """
    if payload is None:
        with open(eval_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if not isinstance(payload, dict):
        # 顶层非对象（如坏文件里的 []）：抛 ValueError，调用方统一跳过该文件，
        # 不能让 AttributeError 中断整条流水线
        raise ValueError("JSON 顶层不是对象，无法解析 worst_windows")
    analysis = analyze_worst_windows(payload.get("worst_windows") or [],
                                     variant=payload.get("variant"))
    analysis["seed"] = payload.get("seed")
    analysis["split"] = payload.get("split")
    analysis["source"] = os.path.basename(eval_path)
    if write_md:
        out_dir = out_dir or os.path.dirname(os.path.abspath(eval_path))
        base = os.path.splitext(os.path.basename(eval_path))[0]
        # eval_<variant><suffix>.json -> worstcases_<variant><suffix>.md
        name = base[len("eval_"):] if base.startswith("eval_") else base
        write_worstcase_md(analysis, os.path.join(out_dir, f"worstcases_{name}.md"))
    return analysis


# ---------------------------------------------------------------- 跨变体错误对照

# 每个变体最多保留的"新增/改对"个案条数（只作报告展示，计数始终完整）
CROSS_LIST_LIMIT = 5


def case_key(rec: Dict) -> str:
    """错误样本的稳定标识：优先 RULER 构造期写入的 sample_id；语言建模窗口用
    split+window；旧格式回退到 task@ctx+问题文本（截断后仍需确定性一致）。"""
    if not isinstance(rec, dict):
        rec = {}
    sid = rec.get("sample_id")
    if sid:
        return str(sid)
    if rec.get("window") is not None:
        return f"lm@{rec.get('split') or '?'}#w{rec['window']}"
    q = rec.get("question") or rec.get("context_tail") or rec.get("answer") or ""
    # 旧格式没有 sample_id 时，同一问题在不同构造中可能有不同 gold（例如
    # multikey 的同一模板多次采样）；只按问题截断会把它们错误合并成一个
    # “共识失败”。答案必须进入回退键。
    ans = rec.get("answer")
    if ans is None:
        ans = rec.get("gold_next")
    return f"{rec.get('task')}@{rec.get('ctx')}|{str(q)[:120]}|{str(ans)[:80]}"


def _categorize_any(rec: Dict) -> Optional[Dict]:
    """对跨变体集合里的原始记录按类型归因（窗口 vs RULER）；失败返回 None。

    跨变体对照的输入是原始 badcases/worst_windows 记录（未归因），这里按
    记录类型分派到对应归因器，使"新增错误"可以按类别统计改进方向。
    """
    try:
        if rec.get("window") is not None or "pred_top5" in rec:
            return categorize_window(rec)
        return categorize_case(rec)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def _case_brief(rec: Dict) -> Dict:
    """跨变体报告中的个案摘要（简洁、可读，完整数据在逐 run 文件里）。"""
    pred = rec.get("pred")
    if pred is None:
        top1, p1 = _window_top1(rec)
        pred = (f"{top1!r} (p={p1:.3f})" if top1 is not None and p1 is not None
                else (repr(top1) if top1 is not None else None))
    excerpt = rec.get("question") or rec.get("context_tail") or ""
    cat = _categorize_any(rec) if isinstance(rec, dict) else None
    return {"key": case_key(rec), "task": rec.get("task"), "ctx": rec.get("ctx"),
            "window": rec.get("window"), "split": rec.get("split"),
            "category": (cat or {}).get("category"),
            "answer": rec.get("answer"), "gold_next": rec.get("gold_next"),
            "excerpt": str(excerpt)[-160:],
            "pred": (str(pred)[:80] if pred is not None else None)}


def synthesize_cross_improvements(cross: Dict) -> List[Dict]:
    """跨变体对照 → 改进方向：先判"账算数据还是算方法"，再定位主导错误类别。"""
    if not isinstance(cross, dict):
        return []
    recs = []
    variants = cross.get("variants") or []
    if len(variants) < 2:
        return recs
    n_unique = _count_int(cross.get("n_unique_failed_items")) or 0
    shared = _count_int(cross.get("n_shared_all_failures")) or 0
    if n_unique and _share(shared, n_unique) >= 0.5:
        recs.append(_advice(
            "数据/难度", "高",
            f"{shared}/{n_unique} 个失败样本为全体已评测变体共识失败",
            "失败以共识为主：优先怀疑数据/构造/内在难度，而不是方法本身；"
            "先修数据再比方法，否则结论会指向错误的对象",
            "人工复核共识失败样例的标注与可解性；修正后重跑评测再对比"))
    ref = cross.get("reference")
    for v in variants:
        if v == ref:
            continue
        pv = (cross.get("per_variant") or {}).get(v) or {}
        n_new = pv.get("n_new_vs_reference")
        if n_new:
            cats = pv.get("new_category_counts") or {}
            top = sorted(cats.items(), key=lambda kv: -kv[1])
            cat_s = (f"，主导类别 {CATEGORY_CN.get(top[0][0], top[0][0])}×{top[0][1]}"
                     if top else "")
            recs.append(_advice(
                "模型/方法", "高",
                f"{v} 相对 {ref} 新增 {n_new} 个错误{cat_s}",
                "特异性新增错误是该方法盲区最直接的线索，优先逐条分析这些样例，"
                "不要只看平均 EM",
                f"对 {v} 的 new_vs_reference 样例做注意力/门控可视化，定位与 "
                f"{ref} 的行为差异；若是难度边界样本占比高，先分层再归因"))
        # 只有“改对”而没有“新增错误”的变体同样要报告：净收益不能因新增
        # 为 0 就静默丢掉正收益证据。
        n_fixed = pv.get("n_fixed_vs_reference")
        if n_fixed:
            recs.append(_advice(
                "模型/方法", "低",
                f"{v} 相对 {ref} 改对 {n_fixed} 个样本",
                "改对与新增要并看：净收益 = 改对 − 新增，只有单侧数字会误导",
                f"结合新增错误做净收益统计与配对显著性检验（significance.json）"))
    return recs[:MAX_ADVICE]


def cross_variant_analysis(per_variant_cases: Dict[str, List[Dict]],
                           reference: str = "standard") -> Dict:
    """跨变体错误对照：把各变体的错误样本按键对齐，区分三类失败。

    输入 per_variant_cases: {variant: [错误样本记录]}（所有变体应使用同一批
    评测样本，逐变体错误列表来自同一份 ruler_*.json / eval_*.json 的 badcases
    或 worst_windows）。空列表表示"该变体已评测且零错误"，会作为通过者计入
    全体变体集合（否则一个全对的变体会被静默排除，让"全体共识失败"失真）。
    输出：
      * shared_all_failures —— 所有变体都错的样本：更可能是数据/构造错误或
        样本本身不可解（尤其当 standard 也错），不应算作某一方法的问题；
      * 相对参照（默认 standard）的 new_vs_reference —— 参照对而该方法错的
        样本：疑似该方法引入的特异性盲区，应优先逐条分析；
      * 相对参照的 fixed_vs_reference —— 参照错而该方法对的样本：该方法
        由错转对的证据，可与新增错误对照衡量净收益。
    n<2 个变体时返回空对照并注明。
    """
    # 保留空列表（已评测且零错误）：它们必须进入 all_set，否则任何一个全对
    # 的变体都会让"所有变体都错"的共识判定漏掉它，把方法特异性失败误报成
    # 数据/难度问题。None/非列表值（无此变体数据）才排除；顶层非对象按空对照。
    per_variant_cases = (per_variant_cases
                         if isinstance(per_variant_cases, dict) else {})
    variants = sorted((v for v, cs in per_variant_cases.items()
                       if isinstance(cs, list)), key=str)
    items: Dict[str, Dict] = {}
    for v in variants:
        for c in per_variant_cases[v]:
            # 外部/旧格式 JSON 的列表可能混入 null/标量：跳过坏记录，否则
            # case_key 的 .get 会中断整份跨变体对照（与 analyze_* 同口径）。
            if not isinstance(c, dict):
                continue
            # 每个变体对同一 key 的记录各不相同（预测/归因依赖该变体自身输出），
            # 必须逐变体保存，不能只留一个代表样本——否则相对参照的新增/改对
            # 会拿别的变体的记录去归因，类别统计与个案展示都会张冠李戴。
            it = items.setdefault(case_key(c),
                                  {"variants": set(), "example": c, "examples": {}})
            it["variants"].add(v)
            it["examples"][v] = c
    if len(variants) < 2:
        return {"variants": variants, "reference": reference,
                "n_unique_failed_items": len(items),
                "n_shared_all_failures": 0, "shared_failures": [],
                "per_variant": {}, "recommendations": [],
                "note": "少于 2 个变体的错误样本，无法做跨变体对照"}
    all_set = set(variants)
    shared = sorted(k for k, it in items.items() if it["variants"] == all_set)
    # 参照变体只有“未评测”（不在 variants 里）才无法计算新增/改对。
    # 若参照已评测且零错误（[]），按定义其余变体的每个失败都是“参照对而
    # 该方法错”，全部计入 new_vs_reference 是正确的，不需要兜底。
    has_ref = reference in variants
    ref_failed = ({k for k, it in items.items() if reference in it["variants"]}
                  if has_ref else set())
    per_v: Dict[str, Dict] = {}
    for v in variants:
        failed = {k for k, it in items.items() if v in it["variants"]}
        if has_ref and v != reference:
            new_keys = sorted(failed - ref_failed)
            fixed_keys = sorted(ref_failed - failed)
        else:
            new_keys, fixed_keys = [], []

        def _example_for(k):
            """该变体对 key 的原始记录（逐变体保存；缺失时回退代表样本）。"""
            return items[k]["examples"].get(v, items[k]["example"])

        def _cat_counts(keys):
            # 对全部新增/改对样本归因（不只展示的前 N 条），供改进方向统计；
            # 归因必须基于该变体自己的记录。
            cc = Counter()
            for k in keys:
                c = _categorize_any(_example_for(k))
                if c:
                    cc[c["category"]] += 1
            return dict(cc)

        per_v[v] = {
            "n_failed": len(failed),
            "n_shared_all": len(failed & set(shared)),
            "n_new_vs_reference": len(new_keys) if has_ref else None,
            "n_fixed_vs_reference": len(fixed_keys) if has_ref else None,
            "new_category_counts": _cat_counts(new_keys),
            "fixed_category_counts": _cat_counts(fixed_keys),
            "new_vs_reference": [_case_brief(_example_for(k))
                                 for k in new_keys[:CROSS_LIST_LIMIT]],
            "fixed_vs_reference": [_case_brief(_example_for(k))
                                   for k in fixed_keys[:CROSS_LIST_LIMIT]],
        }
    out = {
        "variants": variants, "reference": reference,
        "reference_available": has_ref,
        "n_unique_failed_items": len(items),
        "n_shared_all_failures": len(shared),
        "shared_failures": [_case_brief(items[k]["example"])
                            for k in shared[:CROSS_LIST_LIMIT]],
        "per_variant": per_v,
        "note": ("共识失败（所有变体都错，含标准基线）优先怀疑数据/构造/内在"
                 "难度；相对参照新增的错误才是该方法的特异性盲区，优先分析。"
                 + ("" if has_ref else f"⚠ 参照变体 {reference} 未评测或没有"
                                        "可用错误清单（不等于零错误），"
                                        "新增/改对计数不可得。")
                 + f"个案列表每类最多展示 {CROSS_LIST_LIMIT} 条，计数为完整值。"),
    }
    out["recommendations"] = synthesize_cross_improvements(out)
    return out


def process_ruler_json(ruler_path: str, out_dir: Optional[str] = None,
                       write_md: bool = True, payload: Optional[Dict] = None) -> Dict:
    """处理一个 ruler_*.json：归因分析 + 写 md（与源文件同目录）。

    payload 已由调用方解析时可直接传入；为 None 时自行读取。
    """
    if payload is None:
        with open(ruler_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层不是对象，无法解析 badcases")
    analysis = analyze_badcases(payload.get("badcases") or [],
                               variant=payload.get("variant"))
    analysis["seed"] = payload.get("seed")
    analysis["source"] = os.path.basename(ruler_path)
    if write_md:
        out_dir = out_dir or os.path.dirname(os.path.abspath(ruler_path))
        base = os.path.splitext(os.path.basename(ruler_path))[0]
        # ruler_<variant><suffix>.json -> badcases_<variant><suffix>.md
        name = base[len("ruler_"):] if base.startswith("ruler_") else base
        write_badcase_md(analysis, os.path.join(out_dir, f"badcases_{name}.md"))
    return analysis


def main():
    p = argparse.ArgumentParser(description="预测错误样本归因分析（RULER + 语言建模）")
    p.add_argument("--results_dir", required=True, help="含 ruler_*.json / eval_*.json 的结果目录")
    p.add_argument("--ruler", default=None,
                   help="只处理指定文件名（默认目录内全部 ruler_*.json）")
    p.add_argument("--no_md", action="store_true", help="不写 Markdown 报告")
    args = p.parse_args()

    if args.ruler:
        paths = [os.path.join(args.results_dir, args.ruler)]
    else:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "ruler_*.json")))
    if not paths:
        print(f"[badcase] {args.results_dir} 下没有 ruler_*.json，跳过")
    all_analyses = []
    # 归因与跨变体对照共用同一份 payload；解析成功后即登记。
    ruler_payloads = {}
    for fp in paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ana = process_ruler_json(fp, write_md=not args.no_md, payload=payload)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
        # 归因成功后才登记 payload（顶层非对象的坏文件已被上面的校验拒绝，
        # 不能进跨变体对照导致二次崩溃）
        ruler_payloads[fp] = payload
        all_analyses.append(ana)
        name = os.path.splitext(os.path.basename(fp))[0]
        if not ana["n_badcases"]:
            print(f"[badcase] {name}: 无错误样本")
            continue
        top = ", ".join(f"{CATEGORY_CN.get(k, k)}×{v}"
                        for k, v in sorted(ana["category_counts"].items(),
                                           key=lambda kv: -kv[1]))
        print(f"[badcase] {name}: {ana['n_badcases']} 条错误样本 —— {top}")
        # 直接在终端打印前 3 条个案（预测错误的样本要看得见，而非埋在文件里）
        for i, c in enumerate(ana["cases"][:3], 1):
            for line in format_case(i, c):
                print(f"[badcase]   {line}")

    # 语言建模最差窗口（eval_*.json 的 worst_windows）：同样逐条归因并打印
    window_analyses = []
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "eval_*.json"))):
        try:
            ana = process_eval_json(fp, write_md=not args.no_md)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
        if not ana["n_windows"]:
            continue
        window_analyses.append(ana)
        name = os.path.splitext(os.path.basename(fp))[0]
        top = ", ".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{v}"
                        for k, v in sorted(ana["category_counts"].items(),
                                           key=lambda kv: -kv[1]))
        print(f"[badcase] {name}: 最差窗口 {ana['n_windows']} 个 —— {top}")
        for i, c in enumerate(ana["cases"][:2], 1):
            for line in format_window_case(i, c):
                print(f"[badcase]   {line}")

    # 跨变体错误对照：共识失败 vs 相对 standard 的特异性新增错误
    per_variant_cases = {}
    for fp in paths:
        payload = ruler_payloads.get(fp)
        if payload is None:
            continue
        v = payload.get("variant")
        # "badcases": [] 表示"该变体已评测且零错误"，必须保留传入跨变体对照
        # （与 cross_variant_analysis 的语义一致）；只有缺失/非列表才视为无数据。
        if v and isinstance(payload.get("badcases"), list):
            per_variant_cases.setdefault(v, []).append(
                (payload.get("seed"), fp, payload["badcases"]))

    def _seed_key(item):
        raw = item[0]
        if isinstance(raw, (bool, np.bool_)):
            seed = None
        elif isinstance(raw, (int, np.integer)):
            seed = int(raw)
        elif isinstance(raw, str):
            # int("1_0")/float("1_0") 都会得到 10：下划线写法不能冒充种子 10。
            if "_" in raw:
                seed = None
            else:
                try:
                    seed = int(raw)
                except ValueError:
                    seed = _finite_float(raw)
        else:
            seed = _finite_float(raw)
        # 种子身份必须是整数：1.9 与 1.1 截断成同一个 1 会让跨变体
        # 对照静默选错 run；非整数/非有限/缺失一律排到最后。
        if seed is None or seed != math.floor(seed):
            return (1, 0, item[1])
        return (0, seed, item[1])

    # 每个变体取“最小种子”（与 run_all.cross_variant_badcase_map 同策略）：
    # paths 是文件名字典序（seed1000 排在 seed999 之前），取首个会选错种子。
    per_variant = {v: min(cands, key=_seed_key)[2]
                   for v, cands in per_variant_cases.items()}
    if all_analyses or window_analyses:
        merged = {
            "n_files": len(all_analyses),
            "variants": [a.get("variant") for a in all_analyses],
            "total_badcases": sum(a["n_badcases"] for a in all_analyses),
            "category_counts": dict(sum(
                (Counter(a["category_counts"]) for a in all_analyses),
                Counter())),
            "n_worstcase_files": len(window_analyses),
            "total_worst_windows": sum(a["n_windows"] for a in window_analyses),
            "worst_category_counts": dict(sum(
                (Counter(a["category_counts"]) for a in window_analyses),
                Counter())),
            "per_file": [{k: a.get(k) for k in
                          ("variant", "seed", "n_badcases", "category_counts",
                           "hard_sample_ratio", "deep_position_fail_ratio",
                           "label_errors", "source")}
                         for a in all_analyses],
            "per_worstcase_file": [{k: a.get(k) for k in
                                    ("variant", "seed", "split", "n_windows",
                                     "category_counts", "gold_in_context_rate",
                                     "mean_top1_prob", "mean_loss", "source")}
                                   for a in window_analyses],
        }
        # 改进方向：汇总每份分析的合成结果并标注来源，终端只打印高优先级条目
        # （分析不能止步于分类；错误模式必须转成下一步可执行的实验）。
        agg_recs = []
        for a in all_analyses:
            for r in a.get("recommendations") or []:
                agg_recs.append(dict(r, source=f"ruler:{a.get('variant')}"))
        for a in window_analyses:
            for r in a.get("recommendations") or []:
                agg_recs.append(dict(r, source=f"lm:{a.get('variant')}"))
        if len(per_variant) >= 2:
            cross = cross_variant_analysis(per_variant)
            merged["cross_variant"] = cross
            for r in cross.get("recommendations") or []:
                agg_recs.append(dict(r, source="cross_variant"))
        merged["recommendations"] = agg_recs
        out = os.path.join(args.results_dir, "badcase_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        if agg_recs:
            print(f"[badcase] 改进方向 {len(agg_recs)} 条（去重后取前 3 条）：")
            seen_a, shown = set(), 0
            for r in agg_recs:
                key = (r.get("axis"), r.get("suggestion"))
                if key in seen_a:
                    continue
                seen_a.add(key)
                shown += 1
                print(f"[badcase]   [{r.get('source')}][{r.get('axis')}"
                      f"·{r.get('priority')}] {r.get('suggestion')}")
                if shown >= 3:
                    break
        print(f"[badcase] 汇总 -> {out}")


if __name__ == "__main__":
    main()
