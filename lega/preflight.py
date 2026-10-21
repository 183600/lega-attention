"""数据协议训练前闸门（run_all 与直接训练入口共用）。

同一份防泄漏自检结论必须在所有训练入口生效：如果只有 `run_all.py` 检查
`data_report.json`，那么 `python -m lega.train`、调优 trial 或 postrun
仍可直接拿一份“文档跨 split / 时序被打乱 / 跨 split n-gram 重叠”的数据
开训，报告里的告警就形同虚设。本模块把检查集中到一处，训练入口默认
强制，只有显式参数/环境变量才放行历史实验复现。
"""
import json
import os

from .data import CACHE_SCHEMA
from .datacheck import critical_split_alerts
from .repro import sha256_file
from .utils import strict_int

ALLOW_ENV = "LEGA_ALLOW_DATA_PROTOCOL_VIOLATIONS"

# 词表/特征拟合来源的可信值：字符级词表只能在 train 上拟合，GPT-2 BPE 是
# 公开预训练词表（与语料无关）。"all"/"val"/"test" 都意味着用了 eval 侧统计。
_SAFE_FIT_ON = ("train", "train_only", "gpt2_pretrained")


def data_report_protocol_issues(report) -> list:
    """直接核对报告内的协议声明，而不是只信 alerts 列表。

    critical_split_alerts 只看 alerts；一份旧报告/被手改报告只要把 alerts
    清空，就能把 protocol.shuffle_at_split_time=True、test_sealed=False 或
    split_protocol.documents_cross_cut=True 静默带进训练。这里把报告自身
    声明与 split_protocol 的显式违约独立复核一遍（缺失字段不误报）。
    """
    if not isinstance(report, dict):
        return []
    issues = []

    def add(msg):
        issues.append(msg)

    protocol = report.get("protocol")
    if isinstance(protocol, dict):
        for key, bad, msg in (
                ("split_first_then_preprocess", False,
                 "报告声明先预处理后划分（split_first_then_preprocess=False）"),
                ("shuffle_at_split_time", True,
                 "报告声明划分阶段随机打乱（shuffle_at_split_time=True），"
                 "时序/群组完整性不可保证"),
                ("test_sealed", False,
                 "报告声明测试集未尘封（test_sealed=False）")):
            value = protocol.get(key)
            if value is None:
                continue
            if not isinstance(value, bool):
                add(f"报告协议字段 {key}={value!r} 不是布尔值，无法确认协议")
            elif value is bad:
                add(msg)
        fit_on = protocol.get("fit_on")
        if fit_on is not None and str(fit_on) not in _SAFE_FIT_ON:
            add(f"报告 protocol.fit_on={fit_on!r} 不可信："
                "字符词表只能拟合 train，BPE 只能用公开预训练词表")
    tokenizer = report.get("tokenizer")
    if isinstance(tokenizer, dict):
        fit_on = tokenizer.get("fit_on")
        if fit_on is not None and str(fit_on) not in _SAFE_FIT_ON:
            add(f"报告 tokenizer.fit_on={fit_on!r} 不可信："
                "字符词表只能拟合 train，BPE 只能用公开预训练词表")
    prep_read = report.get("test_reads_at_prep")
    if isinstance(prep_read, dict) and "used_for_selection" in prep_read:
        used = prep_read.get("used_for_selection")
        if not isinstance(used, bool):
            add("报告 test_reads_at_prep.used_for_selection="
                f"{used!r} 不是布尔值，无法确认 test 未被用于选择")
        elif used:
            add("报告 test_reads_at_prep.used_for_selection=True："
                "准备阶段读取 test 被用于选择/拟合，测试集不再独立")
    split_protocol = report.get("split_protocol")
    if isinstance(split_protocol, dict):
        # 非布尔值不能被当成“未声明”：JSON 里的 1/"true"/"yes" 同样意味着
        # 上游在声明协议违约（或报告被手改），必须显式拒绝而不能静默通过。
        for key, msg in (
                ("shuffled", "报告 split_protocol.shuffled=True：时序/群组完整性不可保证"),
                ("fallback_char_cut",
                 "报告 split_protocol.fallback_char_cut=True：退化为字符切分，"
                 "同一文档可能跨 split"),
                ("documents_cross_cut",
                 "报告 split_protocol.documents_cross_cut=True：群组跨 split"),
                ("paragraphs_cross_cut",
                 "报告 split_protocol.paragraphs_cross_cut=True：段落群组跨 split")):
            value = split_protocol.get(key)
            if value is None:
                continue
            if not isinstance(value, bool):
                add(f"报告 split_protocol.{key}={value!r} 不是布尔值，无法确认协议")
            elif value:
                add(msg)
        verified = split_protocol.get("verified_disjoint")
        if verified is not None and not isinstance(verified, bool):
            add("报告 split_protocol.verified_disjoint="
                f"{verified!r} 不是布尔值，无法确认划分互斥性")
        overlaps = split_protocol.get("overlap_long_lines")
        bad_overlaps = {}
        if isinstance(overlaps, dict):
            for pair, val in overlaps.items():
                if not isinstance(val, dict):
                    continue
                exact = val.get("exact_lines") or 0
                norm = val.get("normalized_lines") or 0
                if exact or norm:
                    bad_overlaps[str(pair)] = {
                        "exact_lines": exact, "normalized_lines": norm}
        if bad_overlaps:
            prefix = ("报告声明 split_protocol.verified_disjoint=True，但"
                      if verified is True else "")
            add(f"{prefix}实测存在跨 split 共享长行 {bad_overlaps}："
                "声明与内容矛盾，不能按“已核对互斥”处理")
        missing = split_protocol.get("missing_files")
        if isinstance(missing, list) and missing:
            prefix = ("报告声明 split_protocol.verified_disjoint=True，但"
                      if verified is True else "")
            add(f"{prefix}缺少 split 文件 {missing}："
                "无法核对划分互斥性，缺失 split 的指标不可比较")
        if (verified is False and not bad_overlaps
                and not (isinstance(missing, list) and missing)):
            # verified_disjoint=False 本身就是“未验证通过”。没有重叠
            # 明细也没有缺失文件时不能当成通过：报告被手改/上游漏写
            # 证据，同样无法确认三个 split 互斥。
            add("报告声明 split_protocol.verified_disjoint=False，但未记录"
                "跨 split 长行重叠或缺失文件：无法确认划分互斥性")
    return issues


def allow_protocol_violations(cli_flag: bool = False) -> bool:
    """显式放行标志：CLI 参数或环境变量（供 run_all 向子进程透传）。"""
    if cli_flag:
        return True
    return str(os.environ.get(ALLOW_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


def load_data_report(data_dir: str):
    """读取 data_dir/data_report.json；缺失/损坏/非对象返回 None。"""
    path = os.path.join(data_dir, "data_report.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None


def data_report_binding_issue(data_dir: str):
    """检查 data_report.json 是否仍是 meta.json 记录的那一份。

    prepare_bins 把报告哈希写进 meta；若报告后来被手改/替换，缓存校验虽然
    能发现，但训练入口不会自动调用 prepare_bins。这里在训练前显式核对，
    避免汇总页把陈旧/被篡改的防泄漏诊断当作当前数据的事实。
    """
    report_path = os.path.join(data_dir, "data_report.json")
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return "meta.json 不可解析，无法核对 data_report 绑定"
    if not isinstance(meta, dict):
        return "meta.json 顶层不是对象，无法核对 data_report 绑定"
    recorded = meta.get("report_sha256")
    if not recorded:
        # 当前协议下缓存必须记录报告哈希。报告还在却没有哈希，说明 meta
        # 被手改/由旧流程拼接，无法证明报告对应当前数据，不能静默放行。
        if os.path.exists(report_path):
            return ("meta.json 未记录 data_report.json 哈希：报告无法与当前"
                    "数据绑定，防泄漏诊断不可信（须重建 bin/报告）")
        return None
    if not os.path.exists(report_path):
        return ("meta.json 记录了 data_report.json 哈希，但报告文件缺失："
                "缓存不完整或报告被删除，防泄漏诊断不可信")
    try:
        actual = sha256_file(report_path)
    except OSError as e:
        return f"无法读取 data_report.json: {e}"
    if actual != recorded:
        return ("data_report.json 与 meta.json 记录的哈希不一致"
                "（报告被改写或已过期），防泄漏诊断不可信")
    return None


def data_source_hash_issue(data_dir: str):
    """检查 bin 缓存记录的源文本哈希是否仍与当前 txt 一致。

    prepare_bins 命中缓存时会核对，但训练入口可以绕过它直接读 bin。若
    train/val/test.txt 在构建 bin 后被改动，训练/评测仍会静默使用旧缓存：
    此时 data_report.json 的统计与实际语料脱节，最坏情况是评测用了已被
    清洗/替换前的 test。只要 meta 记录过源哈希就强校验；旧缓存没记录时
    返回 None，交由 cache_schema 检查处理。
    """
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return "meta.json 不可解析，无法核对源文本哈希"
    if not isinstance(meta, dict):
        return "meta.json 顶层不是对象，无法核对源文本哈希"
    recorded = meta.get("source_sha256")
    if recorded is None:
        # 只要目录里仍有源文本，缺少源哈希就无法证明 bin 与 txt 一致；
        # “没记录”不等于“没变化”，按不可确认处理（纯 bin 目录才放行）。
        has_sources = any(
            os.path.exists(os.path.join(data_dir, f"{s}.txt"))
            for s in ("train", "val", "test"))
        if has_sources:
            return ("meta.json 未记录源文本哈希：无法确认 bin 由当前 txt 构建，"
                    "须重建数据缓存")
        return None
    if not isinstance(recorded, dict):
        return "meta.json 的 source_sha256 不是对象，无法核对源文本哈希"
    for split in ("train", "val", "test"):
        txt_path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(txt_path):
            continue
        expected = recorded.get(split)
        if not expected:
            return (f"{split}.txt 存在但 meta 未记录其源哈希：bin 可能是由"
                    "旧版/不同文本构建，须重建")
        try:
            actual = sha256_file(txt_path)
        except OSError as e:
            return f"无法读取 {split}.txt 核对源哈希: {e}"
        if actual != expected:
            return (f"{split}.txt 与构建 bin 时的源哈希不一致（源文本已改动，"
                    "缓存过期）：data_report 的诊断不再对应当前语料，须重建")
    return None


def data_bin_hash_issue(data_dir: str):
    """检查 bin 内容是否仍是 meta.json 记录的那一份。

    只核对 token 数量无法发现“同尺寸替换 train.bin（或把 test.bin 复制成
    train.bin）”：那样真正进入训练/评测的 token 流与 data_report.json 的
    去污染/重叠/漂移诊断脱节，最坏情况是 test 内容混入训练。prepare_bins
    命中缓存与 load_bin 都会核对；预检再独立核一遍，保证绕过 prepare_bins
    的训练入口也无法静默使用被替换的 bin。
    """
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return "meta.json 不可解析，无法核对 bin 内容哈希"
    if not isinstance(meta, dict):
        return "meta.json 顶层不是对象，无法核对 bin 内容哈希"
    recorded = meta.get("bin_sha256")
    has_bins = any(os.path.exists(os.path.join(data_dir, f"{s}.bin"))
                   for s in ("train", "val", "test"))
    if recorded is None:
        has_sources = any(os.path.exists(os.path.join(data_dir, f"{s}.txt"))
                          for s in ("train", "val", "test"))
        if has_bins and has_sources:
            return ("meta.json 未记录 bin 内容哈希：无法确认 bin 未被同尺寸"
                    "替换，须重建数据缓存")
        return None
    if not isinstance(recorded, dict):
        return "meta.json 的 bin_sha256 不是对象，无法核对 bin 内容哈希"
    for split in ("train", "val", "test"):
        bin_path = os.path.join(data_dir, f"{split}.bin")
        expected = recorded.get(split)
        if not expected:
            if has_bins and os.path.exists(bin_path):
                return (f"{split}.bin 存在但 meta 未记录其哈希：bin 可能是"
                        "旧版/不同流程产物，须重建")
            continue
        if not os.path.exists(bin_path):
            return f"meta 记录了 {split}.bin 哈希但文件缺失：缓存不完整，须重建"
        try:
            actual = sha256_file(bin_path)
        except OSError as e:
            return f"无法读取 {split}.bin 核对内容哈希: {e}"
        if actual != expected:
            return (f"{split}.bin 内容与 meta.json 记录的哈希不一致"
                    "（bin 被替换/篡改/损坏）：token 流与防泄漏诊断已脱节，"
                    "须重建数据缓存")
    return None


def data_cache_schema_issue(data_dir: str):
    """检查 bin 缓存是否由当前数据管线生成、词表拟合来源是否可信。"""
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return "meta.json 不可解析，无法确认数据缓存协议版本"
    if not isinstance(meta, dict):
        return "meta.json 顶层不是对象，无法确认数据缓存协议版本"
    try:
        schema = strict_int(meta.get("cache_schema", 0),
                            "meta.cache_schema")
    except (TypeError, ValueError, OverflowError):
        schema = -1
    if schema != CACHE_SCHEMA:
        return (f"数据缓存 cache_schema={schema} 与当前管线 "
                f"{CACHE_SCHEMA} 不一致：旧版清洗/去污染协议不可信，"
                "请运行 prepare_bins/prepare_data.py 重建 bin 后再训练")
    # schema 相同不代表分词器拟合来源可信：旧缓存/手改 meta 可能在全量
    # 语料（含 val/test）上拟合过词表。prepare_bins 命中缓存时会检查
    # fit_on，但直接训练入口绕过它，这里必须补上同一道检查。
    fit_on = meta.get("fit_on")
    if fit_on not in ("train", "gpt2_pretrained"):
        return (f"数据缓存 fit_on={fit_on!r} 不可信：字符词表必须在 train 上"
                "拟合（GPT-2 BPE 为公开预训练词表）；请重建 bin")
    return None


def data_protocol_preflight(data_dir: str, allow_override: bool = False) -> dict:
    """训练前的数据协议闸门：报告缺失只告警，严重违规默认中止。

    报告声明“打乱/字符切分/文档跨 split”、报告与 meta 绑定失效、或 bin
    缓存由旧版清洗/去污染协议生成时，已知/疑似泄露的数据不应进入全量训练。
    返回汇总（写入 repro），违规且未显式覆盖时抛 SystemExit；
    allow_override 仅用于复现历史实验。
    """
    report = load_data_report(data_dir)
    binding_issue = data_report_binding_issue(data_dir)
    schema_issue = data_cache_schema_issue(data_dir)
    source_hash_issue = data_source_hash_issue(data_dir)
    bin_hash_issue = data_bin_hash_issue(data_dir)
    critical = list(critical_split_alerts(report))
    for issue in data_report_protocol_issues(report):
        critical.append({"level": "warning", "code": "data_report_protocol",
                         "message": issue})
    if binding_issue:
        critical.append({"level": "warning", "code": "data_report_binding",
                         "message": binding_issue})
    if schema_issue:
        critical.append({"level": "warning", "code": "data_cache_schema",
                         "message": schema_issue})
    if source_hash_issue:
        critical.append({"level": "warning", "code": "data_source_hash",
                         "message": source_hash_issue})
    if bin_hash_issue:
        critical.append({"level": "warning", "code": "data_bin_hash",
                         "message": bin_hash_issue})
    info = {
        "report_present": report is not None,
        "binding_issue": binding_issue,
        "cache_schema_issue": schema_issue,
        "source_hash_issue": source_hash_issue,
        "bin_hash_issue": bin_hash_issue,
        "critical_violations": critical,
        "override": bool(allow_override),
    }
    if critical and not allow_override:
        detail = "；".join(
            f"[{a.get('code')}] {a.get('message')}" for a in critical[:5])
        raise SystemExit(
            f"数据协议前置检查失败（{len(critical)} 个严重问题）：{detail}。"
            "请先处理划分/清洗（必要时重建 data_report.json），确认无泄露后再"
            "训练；确需复现历史实验可显式加 --allow_data_protocol_violations "
            f"或设置 {ALLOW_ENV}=1。")
    return info
