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

用法:
  python -m lega.badcase --results_dir results/tiny_smoke      # 处理全部 ruler_*.json
  python -m lega.badcase --results_dir results/small --ruler ruler_lega_seed1337.json
"""
import argparse
import glob
import json
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
    p = argparse.ArgumentParser(description="RULER 坏案例归因分析")
    p.add_argument("--results_dir", required=True, help="含 ruler_*.json 的结果目录")
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
        return
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
    if all_analyses:
        merged = {
            "n_files": len(all_analyses),
            "variants": [a.get("variant") for a in all_analyses],
            "total_badcases": sum(a["n_badcases"] for a in all_analyses),
            "category_counts": dict(sum(
                (Counter(a["category_counts"]) for a in all_analyses),
                Counter())),
            "per_file": [{k: a.get(k) for k in
                          ("variant", "seed", "n_badcases", "category_counts",
                           "hard_sample_ratio", "deep_position_fail_ratio",
                           "label_errors", "source")}
                         for a in all_analyses],
        }
        out = os.path.join(args.results_dir, "badcase_summary.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[badcase] 汇总 -> {out}")


if __name__ == "__main__":
    main()
