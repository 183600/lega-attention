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
from collections import Counter
from typing import Dict, List, Optional

_NUM_RE = re.compile(r"\d{4,}")
_WORD_RE = re.compile(r"[a-z]+")

CATEGORY_CN = {
    "label_error": "标注/构造错误（gold 不在 prompt 中，不是模型的错）",
    "no_answer": "无数值回答（模型未输出可解析的数字，格式盲区）",
    "copy_distractor": "误抄干扰值（检索到错误位置：输出了上下文中的干扰数字）",
    "hallucinated": "幻觉数字（输出了上下文中不存在的数字）",
    "fwe_missing": "fwe 漏答高频词",
    "fwe_spurious": "fwe 多答无关词",
    "fwe_no_parse": "fwe 无法解析（未输出小写词）",
}


def categorize_case(rec: Dict) -> Dict:
    """把一条坏案例归入错误类别；返回附加 category/explanation/difficulty 的副本。

    归因顺序：先排除"不是模型的错"（标注/构造错误），再区分难度边界样本与
    模型盲区，最后按输出内容定位具体机制。
    """
    r = dict(rec)
    task = r.get("task")
    # 0) 难度标注：只有 niah 家族的 depth 是 needle 位置比例（0~1）；vt 的
    # meta.depth 是链深度（整数），不能按位置解释。先算好该标志，保证
    # 下面各归因分支（fwe/no_answer 等提前返回）的返回值都带 hard_sample。
    depth = r.get("depth")
    positional = (depth is not None
                  and str(task or "").startswith("niah"))
    r["hard_sample"] = bool(positional and depth > 0.75)
    # 1) 标注/构造自检：gold 答案必须真的在 prompt 里，否则错在数据侧
    if task != "fwe" and r.get("answer_in_prompt") is False:
        r["category"] = "label_error"
        r["explanation"] = "gold 答案未出现在 prompt 中：样例构造或标注有错，" \
                           "该错误样本不应计入模型失败"
        return r
    # 2) fwe：按词表覆盖情况归因
    if task == "fwe":
        pred_words = set(_WORD_RE.findall((r.get("pred") or "").lower()))
        top = set(r.get("answer_words") or [])
        if not pred_words:
            r["category"] = "fwe_no_parse"
            r["explanation"] = "模型输出中没有可解析的小写词"
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
    # 3) 数值任务：按输出的第一个数字归因（与 EM 计分口径一致）
    nums = _NUM_RE.findall(r.get("pred") or "")
    if not nums:
        r["category"] = "no_answer"
        r["explanation"] = "模型没有给出可解析的数字回答（可能复读问题或提前停止）"
        return r
    first = nums[0]
    ctx_nums = set(r.get("context_numbers") or [])
    if first in ctx_nums:
        r["category"] = "copy_distractor"
        r["explanation"] = (f"输出了 {first}——这是上下文中的干扰值（gold={r.get('answer')}），"
                            "注意力检索到了错误位置（同类干扰行/错误链）")
    else:
        r["category"] = "hallucinated"
        r["explanation"] = (f"输出了 {first}——上下文中并不存在该数字"
                            f"（gold={r.get('answer')}），属于幻觉")
    return r


def analyze_badcases(badcases: List[Dict], variant: str = None) -> Dict:
    """坏案例集合 → 归因分布 + 难度/上下文切片 + 结构化个案列表。"""
    cases = [categorize_case(c) for c in badcases]
    cat_counts = Counter(c["category"] for c in cases)
    by_task_ctx: Dict[str, Dict] = {}
    for c in cases:
        key = f"{c['task']}@{c.get('ctx')}"
        by_task_ctx.setdefault(key, Counter())[c["category"]] += 1
    hard = sum(1 for c in cases if c.get("hard_sample"))
    label_errors = cat_counts.get("label_error", 0)
    # 深度信息只在 niah 家族存在（vt meta.depth 是链深度而非位置）；给出
    # 深位（>0.75）失败占比用于判断"是样本极难，还是模型系统性盲区"
    with_depth = [c for c in cases if c.get("depth") is not None
                  and str(c.get("task") or "").startswith("niah")]
    deep_fail = sum(1 for c in with_depth if c["depth"] > 0.75)
    return {
        "variant": variant,
        "n_badcases": len(cases),
        "category_counts": dict(cat_counts),
        "category_cn": {k: CATEGORY_CN.get(k, k) for k in cat_counts},
        "by_task_ctx": {k: dict(v) for k, v in by_task_ctx.items()},
        "hard_sample_ratio": (hard / len(cases)) if cases else None,
        "deep_position_fail_ratio": (deep_fail / len(with_depth)
                                      if with_depth else None),
        "label_errors": label_errors,
        "cases": cases,
    }


def format_case(idx: int, c: Dict) -> List[str]:
    """单个坏案例的 Markdown 行（问题/gold 摘录/预测/归因）。"""
    lines = [f"### 案例 {idx}: {c['task']} @ctx={c.get('ctx')}"
             + (f" depth={c['depth']:.2f}" if c.get("depth") is not None else "")
             + (" 【难度边界样本：needle 靠后】" if c.get("hard_sample") else "")]
    q = (c.get("question") or "").strip().replace("\n", " ⏎ ")
    if len(q) > 220:
        q = q[:220] + "…"
    lines.append(f"- 问题: {q}")
    if c.get("needle_excerpt"):
        ne = c["needle_excerpt"].replace("\n", " ⏎ ")
        lines.append(f"- gold 证据摘录: {ne}")
    pred = (c.get("pred") or "").strip().replace("\n", " ⏎ ")
    if len(pred) > 160:
        pred = pred[:160] + "…"
    lines.append(f"- 模型输出: {pred!r}")
    lines.append(f"- gold 答案: {c.get('answer')!r}")
    lines.append(f"- 归因: **{CATEGORY_CN.get(c['category'], c['category'])}**"
                 f" —— {c.get('explanation', '')}")
    return lines


def write_badcase_md(analysis: Dict, path: str, max_cases: int = 40) -> None:
    """写人类可读的 Markdown 报告（含归因分布表与逐案例详情）。"""
    md = [f"# 坏案例分析：{analysis.get('variant') or '-'}", ""]
    if not analysis["n_badcases"]:
        md.append("本变体无预测错误样本（或评测被跳过）。")
    else:
        md += [f"共 {analysis['n_badcases']} 条错误样本；归因分布：", "",
               "| 类别 | 数量 | 占比 |", "|---|---|---|"]
        total = analysis["n_badcases"]
        for cat, n in sorted(analysis["category_counts"].items(),
                             key=lambda kv: -kv[1]):
            md.append(f"| {CATEGORY_CN.get(cat, cat)} | {n} | {n / total:.1%} |")
        if analysis.get("deep_position_fail_ratio") is not None:
            md += ["",
                   f"needle 深位（>0.75）失败占 niah 类错误的 "
                   f"{analysis['deep_position_fail_ratio']:.0%}——"
                   "高占比说明是长程检索边界问题，低占比说明是系统性盲区。"]
        if analysis.get("label_errors"):
            md += ["", f"⚠ 有 {analysis['label_errors']} 条 gold 答案不在 prompt 中："
                        "样例构造或标注有错，须先修 RULER 构造再谈模型。"]
        md.append("")
        for i, c in enumerate(analysis["cases"][:max_cases], 1):
            md += [""] + format_case(i, c)
        if analysis["n_badcases"] > max_cases:
            md.append(f"\n（仅展示前 {max_cases} 条，完整结构化数据见同目录 "
                      "badcase_summary.json）")
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
}

# top1 概率 ≥ 阈值仍错 → 高置信错答（系统性偏差）；< 阈值 → 模型自己也不确定
TOP1_CONF_TOL = 0.5


def _window_top1(rec: Dict):
    """取出窗口 top1 预测 (文本, 概率)；缺失/格式异常返回 (None, None)。"""
    top5 = rec.get("pred_top5") or []
    if not top5:
        return None, None
    item = top5[0]
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return str(item[0]), float(item[1])
        except (TypeError, ValueError):
            return str(item[0]), None
    return str(item), None


def categorize_window(rec: Dict) -> Dict:
    """把一条语言建模坏案例（PPL 最差窗口）按可检验规则归因。

    规则（显式、顺序确定）：
      1. 无 top5 预测 → unparsed；
      2. top1 即 gold → top1_correct（排位正确但概率低，校准问题）；
      3. top1 概率 ≥ TOP1_CONF_TOL 且错 → confident_bias（高置信错答，
         最值得作为改进线索的系统性盲区）；若该 token 同时是上下文里已有的
         串，标记 repeats_context 并在解释中注明（复读是高置信错答的一种机制）；
      4. top1（非空白、≥2 字符）出现在 context_tail 中且不是 gold →
         repetition_attractor（低置信下的复读回环：模型退化为抄上下文）；
      5. 其余 → low_confidence（模型在多个候选间平均：样本可能本身多解/
         噪声，或训练不足）。
    附加布尔 gold_in_context / gold_in_top5 区分"上下文里出现过 gold 但
    模型没复制"（复制/检索失败）与"gold 从未出现"（真正的预测失败/长尾）。
    gold_in_context 优先使用采集时按完整窗口判定的布尔（evaluate 写入）；
    旧格式记录没有该字段时退化为 context_tail（只有窗口尾部 160 字符）检查，
    此时"未出现"只代表尾部未出现。
    """
    r = dict(rec)
    top1, p1 = _window_top1(r)
    gold = r.get("gold_next")
    gold_txt = gold if isinstance(gold, str) else ("" if gold is None else str(gold))
    tail = r.get("context_tail") or ""
    top5_txt = []
    for item in (r.get("pred_top5") or []):
        if isinstance(item, (list, tuple)) and item:
            top5_txt.append(str(item[0]))
        elif item is not None:
            top5_txt.append(str(item))
    r["top1"] = top1
    r["top1_prob"] = p1
    recorded_ctx = r.get("gold_in_context")
    r["gold_in_context"] = (bool(recorded_ctx)
                            if isinstance(recorded_ctx, bool)
                            else bool(gold_txt) and gold_txt in tail)
    r["gold_in_top5"] = bool(gold_txt) and gold_txt in top5_txt
    if top1 is None:
        r["category"] = "unparsed"
        r["explanation"] = "窗口记录中没有可用的 top5 预测分布"
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
               if r["gold_in_context"] else
               "gold 在上下文中未出现（真正的预测失败，长尾/分布外）"))
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
               if r["gold_in_context"] else
               "，gold 未在上下文中出现（样本本身难/长尾）"))
    return r


def _mean(vals):
    # 与 utils.mean_std 同口径：剔除 None 与非有限值（NaN/Inf）。发散 run 的
    # 窗口 loss 可能是 NaN，若混入均值会把 NaN 写进 JSON（非法）并污染报告。
    vals = [float(v) for v in vals
            if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def analyze_worst_windows(windows: List[Dict], variant: str = None) -> Dict:
    """最差窗口集合 → 归因分布 + 校准/复制统计 + 结构化个案列表。"""
    cases = [categorize_window(w) for w in (windows or [])]
    cat_counts = Counter(c["category"] for c in cases)
    return {
        "variant": variant,
        "n_windows": len(cases),
        "category_counts": dict(cat_counts),
        "category_cn": {k: WINDOW_CATEGORY_CN.get(k, k) for k in cat_counts},
        "mean_top1_prob": _mean([c.get("top1_prob") for c in cases]),
        "repeats_context_rate": (sum(1 for c in cases
                                     if c.get("repeats_context"))
                                 / len(cases)) if cases else None,
        "gold_in_top5_rate": (sum(1 for c in cases if c.get("gold_in_top5"))
                              / len(cases)) if cases else None,
        "gold_in_context_rate": (sum(1 for c in cases if c.get("gold_in_context"))
                                 / len(cases)) if cases else None,
        "mean_loss": _mean([c.get("loss") for c in cases]),
        "cases": cases,
    }


def format_window_case(idx: int, c: Dict) -> List[str]:
    """单个最差窗口的 Markdown 行（上下文尾部/模型 top-1/真实续写/归因）。"""
    where = (f"窗口 #{c.get('window')}" +
             (f"（{c['split']} 集）" if c.get("split") else ""))
    loss = c.get("loss")
    # loss 可能缺失或显式为 null（旧格式/外部 JSON）：显示 '-' 而不是崩溃
    loss_s = f"{float(loss):.3f}" if isinstance(loss, (int, float)) else "-"
    lines = [f"### 案例 {idx}: {where}　loss={loss_s}"]
    tail = (c.get("context_tail") or "").replace("\n", " ⏎ ")[-160:]
    lines.append(f"- 上下文尾部: …{tail}")
    top1 = c.get("top1")
    p1 = c.get("top1_prob")
    top_s = f"{top1!r}" + (f"（p={p1:.3f}）" if p1 is not None else "")
    lines.append(f"- 模型 top-1: {top_s}")
    lines.append(f"- 真实续写(gold): {c.get('gold_next')!r}"
                 + ("　（gold 在上下文中出现过）" if c.get("gold_in_context")
                    else "　（gold 未在上下文中出现）"))
    lines.append(f"- 归因: **{WINDOW_CATEGORY_CN.get(c['category'], c['category'])}**"
                 f" —— {c.get('explanation', '')}")
    return lines


def write_worstcase_md(analysis: Dict, path: str, max_cases: int = 10) -> None:
    """写语言建模坏案例的 Markdown 报告（含归因分布与逐案例详情）。"""
    md = [f"# 语言建模最差窗口归因：{analysis.get('variant') or '-'}", ""]
    if not analysis["n_windows"]:
        md.append("本变体无最差窗口记录（评测被跳过或 --worst_n=0）。")
    else:
        md += [f"共 {analysis['n_windows']} 个最差窗口；归因分布：", "",
               "| 类别 | 数量 | 占比 |", "|---|---|---|"]
        total = analysis["n_windows"]
        for cat, n in sorted(analysis["category_counts"].items(),
                             key=lambda kv: -kv[1]):
            md.append(f"| {WINDOW_CATEGORY_CN.get(cat, cat)} | {n} | {n / total:.1%} |")
        stats = []
        if analysis.get("mean_top1_prob") is not None:
            stats.append(f"平均 top-1 概率={analysis['mean_top1_prob']:.3f}")
        if analysis.get("gold_in_context_rate") is not None:
            stats.append(f"gold 出现在上下文中的比例="
                         f"{analysis['gold_in_context_rate']:.0%}")
        if analysis.get("gold_in_top5_rate") is not None:
            stats.append(f"gold 进入 top-5 的比例="
                         f"{analysis['gold_in_top5_rate']:.0%}")
        if stats:
            md += ["", "；".join(stats) + "。"]
        not_ctx = sum(1 for c in analysis["cases"] if not c.get("gold_in_context"))
        if not_ctx:
            md += ["", f"其中 {not_ctx} 个窗口的 gold 从未出现在上下文中——"
                       "这些更接近真正的预测失败/长尾，应与“抄不到”的复制失败分开看。"]
        md.append("")
        for i, c in enumerate(analysis["cases"][:max_cases], 1):
            md += [""] + format_window_case(i, c)
        if analysis["n_windows"] > max_cases:
            md.append(f"\n（仅展示 loss 最高的前 {max_cases} 个窗口，"
                      "完整数据见 eval_*.json 的 worst_windows）")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def process_eval_json(eval_path: str, out_dir: Optional[str] = None,
                      write_md: bool = True) -> Dict:
    """处理一个 eval_*.json 的 worst_windows：归因分析 + 写 md（与源文件同目录）。"""
    with open(eval_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
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
    sid = rec.get("sample_id")
    if sid:
        return str(sid)
    if rec.get("window") is not None:
        return f"lm@{rec.get('split') or '?'}#w{rec['window']}"
    q = rec.get("question") or rec.get("context_tail") or rec.get("answer") or ""
    return f"{rec.get('task')}@{rec.get('ctx')}|{str(q)[:120]}"


def _case_brief(rec: Dict) -> Dict:
    """跨变体报告中的个案摘要（简洁、可读，完整数据在逐 run 文件里）。"""
    pred = rec.get("pred")
    if pred is None:
        top1, p1 = _window_top1(rec)
        pred = (f"{top1!r} (p={p1:.3f})" if top1 is not None and p1 is not None
                else (repr(top1) if top1 is not None else None))
    excerpt = rec.get("question") or rec.get("context_tail") or ""
    return {"key": case_key(rec), "task": rec.get("task"), "ctx": rec.get("ctx"),
            "window": rec.get("window"), "split": rec.get("split"),
            "answer": rec.get("answer"), "gold_next": rec.get("gold_next"),
            "excerpt": str(excerpt)[-160:],
            "pred": (str(pred)[:80] if pred is not None else None)}


def cross_variant_analysis(per_variant_cases: Dict[str, List[Dict]],
                           reference: str = "standard") -> Dict:
    """跨变体错误对照：把各变体的错误样本按键对齐，区分三类失败。

    输入 per_variant_cases: {variant: [错误样本记录]}（所有变体应使用同一批
    评测样本，逐变体错误列表来自同一份 ruler_*.json / eval_*.json 的 badcases
    或 worst_windows）。空列表表示"该变体已评测且零错误"，会作为通过者计入
    全体变体集合（否则一个全对的变体会被静默排除，让"全体共识失败"失真）。
    输出：
      * shared_all_failures —— 所有变体都错的样本：更可能是数据/构造错误或
        样本本身不可解（尤其当 standard 也错），不应算作某一方法的缺陷；
      * 相对参照（默认 standard）的 new_vs_reference —— 参照对而该方法错的
        样本：疑似该方法引入的特异性盲区，应优先逐条分析；
      * 相对参照的 fixed_vs_reference —— 参照错而该方法对的样本：该方法
        由错转对的证据，可与新增错误对照衡量净收益。
    n<2 个变体时返回空对照并注明。
    """
    # 保留空列表（已评测且零错误）：它们必须进入 all_set，否则任何一个全对
    # 的变体都会让"所有变体都错"的共识判定漏掉它，把方法特异性失败误报成
    # 数据/难度问题。None/非列表值（无此变体数据）才排除。
    variants = sorted(v for v, cs in (per_variant_cases or {}).items()
                      if isinstance(cs, list))
    items: Dict[str, Dict] = {}
    for v in variants:
        for c in per_variant_cases[v]:
            items.setdefault(case_key(c), {"variants": set(), "example": c})["variants"].add(v)
    if len(variants) < 2:
        return {"variants": variants, "reference": reference,
                "n_unique_failed_items": len(items),
                "n_shared_all_failures": 0, "shared_failures": [],
                "per_variant": {},
                "note": "少于 2 个变体的错误样本，无法做跨变体对照"}
    all_set = set(variants)
    shared = sorted(k for k, it in items.items() if it["variants"] == all_set)
    # 参照变体（默认 standard）无任何错误时无法计算“新增/改对”：不套用
    # “全部失败都算新增”的兜底，否则每个变体都会被误报为引入大量新增错误。
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
        per_v[v] = {
            "n_failed": len(failed),
            "n_shared_all": len(failed & set(shared)),
            "n_new_vs_reference": len(new_keys) if has_ref else None,
            "n_fixed_vs_reference": len(fixed_keys) if has_ref else None,
            "new_vs_reference": [_case_brief(items[k]["example"])
                                 for k in new_keys[:CROSS_LIST_LIMIT]],
            "fixed_vs_reference": [_case_brief(items[k]["example"])
                                   for k in fixed_keys[:CROSS_LIST_LIMIT]],
        }
    return {
        "variants": variants, "reference": reference,
        "reference_available": has_ref,
        "n_unique_failed_items": len(items),
        "n_shared_all_failures": len(shared),
        "shared_failures": [_case_brief(items[k]["example"])
                            for k in shared[:CROSS_LIST_LIMIT]],
        "per_variant": per_v,
        "note": ("共识失败（所有变体都错，含标准基线）优先怀疑数据/构造/内在"
                 "难度；相对参照新增的错误才是该方法的特异性盲区，优先分析。"
                 + ("" if has_ref else f"⚠ 参照变体 {reference} 没有错误样本，"
                                        "新增/改对计数不可得。")
                 + f"个案列表每类最多展示 {CROSS_LIST_LIMIT} 条，计数为完整值。"),
    }


def process_ruler_json(ruler_path: str, out_dir: Optional[str] = None,
                       write_md: bool = True) -> Dict:
    """处理一个 ruler_*.json：归因分析 + 写 md（与源文件同目录）。"""
    with open(ruler_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
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
    for fp in paths:
        try:
            ana = process_ruler_json(fp, write_md=not args.no_md)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
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
    per_variant = {}
    for fp in paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        v = payload.get("variant")
        if v and v not in per_variant and payload.get("badcases"):
            per_variant[v] = payload["badcases"]
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
        if len(per_variant) >= 2:
            merged["cross_variant"] = cross_variant_analysis(per_variant)
        out = os.path.join(args.results_dir, "badcase_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[badcase] 汇总 -> {out}")


if __name__ == "__main__":
    main()
