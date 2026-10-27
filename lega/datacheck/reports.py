"""测试集访问账本与数据报告组装。"""
from ._shared import _install as _shared_install

_shared_install(globals())

def read_test_access_log(data_dir: str) -> list:
    path = os.path.join(data_dir, TEST_ACCESS_LOG)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            # 合法 JSON 但非对象的行（null/数组/标量）按坏行跳过；
            # summarize_test_access 需要按 dict 取字段。
            if isinstance(rec, dict):
                records.append(rec)
    return records


def append_test_access(data_dir: str, record: dict) -> dict:
    """追加一条 test 访问记录并返回审计摘要（原子写入 + 进程间文件锁）。

    读-改-写必须跨进程互斥：run_all 的评测阶段会并发评测多个变体，无锁时
    两个进程各自读旧账本再覆盖，会互相吞掉记录。锁加在独立的 .lock 文件上
    （对账本文件本身加锁不可行：os.replace 会更换 inode，后到者锁的是旧文件）；
    无 fcntl 的平台退化为原行为，不阻塞评测。
    """
    path = os.path.join(data_dir, TEST_ACCESS_LOG)
    lock_fp = None
    if fcntl is not None:
        lock_fp = open(path + ".lock", "a")
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
        records = read_test_access_log(data_dir)
        entry = dict(record)
        entry.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        records.append(entry)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return summarize_test_access(records)
    finally:
        if lock_fp is not None:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            lock_fp.close()


def summarize_test_access(records) -> dict:
    """测试集访问审计：评测次数、checkpoint 数、重复评测及其 test 数据版本。

    重复判定按 (checkpoint, test 数据指纹)：重建 test.bin 后，同一 checkpoint
    在新测试集上评测是合法的新实验，不应被旧账本误拒绝；反过来，同一版本
    上的重复评测必须能被识别。旧记录没有 test 指纹时按版本未知分组。
    """
    # 账本可能损坏成 bool/数字（JSON null/true）；不能对非可迭代对象做
    # 列表推导，否则一个坏文件会让评测入口在写访问账本时崩溃。
    records = ([r for r in records if isinstance(r, dict)]
               if isinstance(records, (list, tuple)) else [])
    counts = {}
    versions = set()
    for r in records:
        c = str(r.get("ckpt_sha256") or r.get("ckpt") or "")
        if not c:
            continue
        ver = r.get("test_data_fingerprint")
        ver = str(ver) if ver else None
        key = (c, ver)
        counts[key] = counts.get(key, 0) + 1
        if ver is not None:
            versions.add(ver)
    repeated = []
    repeated_details = []
    for (c, ver), n in counts.items():
        if n <= 1:
            continue
        prefix = c[:16]
        if prefix not in repeated:
            repeated.append(prefix)
        repeated_details.append({
            "ckpt_prefix": prefix,
            "test_data_fingerprint": ver,
            "evaluations": int(n),
        })
    ckpts = {c for c, _ in counts}
    return {
        "evaluations": int(len(records)),
        "distinct_checkpoints": int(len(ckpts)),
        "checkpoints_evaluated_multiple_times": repeated,
        # 精确到数据版本的重复项：调用方传当前 test 指纹时可区分“换数据重测”。
        "repeated_evaluations": repeated_details,
        "distinct_test_data_versions": int(len(versions)),
        "test_data_fingerprints": sorted(versions),
        "first_time": records[0].get("time") if records else None,
        "last_time": records[-1].get("time") if records else None,
    }


# ---------------------------------------------------------------- 报告组装

# 自检告警阈值：不是统计检验的显著性水平，而是“必须人工复核”的操作触发线，
# 与开放集/结构不同分布下的经验值一致（可通过 extra_report 覆盖记录）。


def leakage_alerts(tokenizer: dict, sanitization: dict, duplicates: dict,
                   contamination: dict, distribution: dict,
                   text_stats: dict = None, thresholds: dict = None) -> list:
    """把防泄漏/清洗诊断转成显式告警（供打印与汇总页展示）。

    返回 [{"level": "warning"|"info", "code": ..., "message": ...}]。
    没有告警时返回空列表——"无告警"与"没有自检"是两回事，后者由报告缺失表示。
    """
    def _num(v):
        if isinstance(v, (bool, np.bool_)):
            return None
        if isinstance(v, str) and "_" in v:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        return f if np.isfinite(f) else None

    def _count(v):
        # 计数必须是整数：bool/小数/坏字符串不能截断成看似合法的行数，
        # 否则报告会凭空写出“已去污染 N 行”或触发不存在的重叠告警。
        try:
            n = strict_int(0 if v is None else v, "计数")
        except ValueError:
            return 0
        return n if n >= 0 else 0

    th = dict(ALERT_THRESHOLDS)
    if isinstance(thresholds, dict):
        for k, v in thresholds.items():
            if k in th:
                f = _num(v)
                if f is not None:
                    th[k] = f
    alerts = []

    def add(level, code, message):
        alerts.append({"level": level, "code": code, "message": message})

    contamination = contamination if isinstance(contamination, dict) else {}
    for key, label in (("train_val", "train/val"), ("train_test", "train/test"),
                       ("val_test", "val/test")):
        c = contamination.get(key)
        c = c if isinstance(c, dict) else {}
        rate = _num(c.get("overlap_rate"))
        if rate is not None and rate > th["ngram_overlap_rate"]:
            add("warning", f"ngram_overlap_{key}",
                f"{label} 采样 {c.get('n', '?')}-gram 重叠率 {rate:.4%}"
                "（>0 表示疑似跨 split 语料复制，须人工核查）")
    distribution = distribution if isinstance(distribution, dict) else {}
    for key, label in (("val_vs_train", "val−train"),
                       ("test_vs_train", "test−train"),
                       ("test_vs_val", "test−val")):
        d = distribution.get(key)
        d = d if isinstance(d, dict) else {}
        tv, js = _num(d.get("tv")), _num(d.get("js_bits"))
        if tv is not None and tv > th["distribution_tv"]:
            add("warning", f"distribution_tv_{key}",
                f"{label} 一元 token TV 距离 {tv:.4f} > "
                f"{th['distribution_tv']}（分布漂移明显，结论外推需谨慎）")
        if js is not None and js > th["distribution_js_bits"]:
            add("warning", f"distribution_js_{key}",
                f"{label} JS 散度 {js:.4f} bit > {th['distribution_js_bits']}")
        oov = _num(d.get("eval_oov_token_rate"))
        if oov is not None and oov > th["eval_oov_token_rate"]:
            add("warning", f"eval_oov_{key}",
                f"{label} 方向 OOV token 质量占比 {oov:.4%} > "
                f"{th['eval_oov_token_rate']}")
    tokenizer = tokenizer if isinstance(tokenizer, dict) else {}
    unk_rate = tokenizer.get("unk_rate")
    for split, rate in (unk_rate if isinstance(unk_rate, dict) else {}).items():
        rate = _num(rate)
        if rate is not None and rate > th["eval_unk_char_rate"]:
            add("warning", f"unk_char_rate_{split}",
                f"{split} 字符级 OOV 率 {rate:.4%} > "
                f"{th['eval_unk_char_rate']}（该 split 有较多字符映射 <unk>）")
    sanitization = sanitization if isinstance(sanitization, dict) else {}
    for split, st in sanitization.items():
        st = st if isinstance(st, dict) else {}
        rate = _num(st.get("bad_char_line_rate"))
        if rate is not None and rate > th["bad_char_line_rate"]:
            add("warning", f"bad_char_lines_{split}",
                f"{split} 含坏字符行占比 {rate:.4%} > "
                f"{th['bad_char_line_rate']}（标签噪声偏高，建议回查语料）")
    dup = duplicates if isinstance(duplicates, dict) else {}
    n_exact = _count(dup.get("train_lines_dropped_as_duplicates_of_eval"))
    n_norm = _count(
        dup.get("train_lines_dropped_as_normalized_duplicates_of_eval"))
    if n_exact or n_norm:
        add("info", "decontamination_applied",
            f"已从 train 移除与 val/test 重复的长行：精确 {n_exact} 行、"
            f"空白归一化后额外 {n_norm} 行（val/test 未改写）")
    n_train_dup = _count(
        dup.get("train_intra_split_duplicate_lines_removed"))
    n_train_dup_norm = _count(
        dup.get("train_intra_split_normalized_duplicate_lines_removed"))
    if n_train_dup or n_train_dup_norm:
        add("info", "train_intra_duplicates_removed",
            f"已从 train 移除内部重复长行：精确 {n_train_dup} 行、"
            f"空白归一化后额外 {n_train_dup_norm} 行（保留首次出现）")
    n_short_tv = _count(dup.get("train_val_overlap_short_lines"))
    n_short_tt = _count(dup.get("train_test_overlap_short_lines"))
    n_short_vt = _count(dup.get("val_test_overlap_short_lines"))
    n_short_tv_norm = _count(
        dup.get("train_val_overlap_normalized_short_lines"))
    n_short_tt_norm = _count(
        dup.get("train_test_overlap_normalized_short_lines"))
    n_short_vt_norm = _count(
        dup.get("val_test_overlap_normalized_short_lines"))
    if n_short_tv or n_short_tt or n_short_tv_norm or n_short_tt_norm:
        add("warning", "short_line_cross_split_overlap",
            f"短行（<{dup.get('min_chars', '?')} 字符）跨 split 重复："
            f"train/val 精确 {n_short_tv} 行、train/test 精确 {n_short_tt} 行"
            f"（空白归一化后 {n_short_tv_norm}/{n_short_tt_norm}）；"
            "为避免误删标题/固定用语未从 train 移除，但重复样本会削弱 "
            "held-out 独立性，须人工核查")
    if n_short_vt or n_short_vt_norm:
        add("warning", "eval_short_lines_overlap",
            f"val/test 短行重复：精确 {n_short_vt} 行、"
            f"空白归一化后 {n_short_vt_norm} 行；val 选型会部分预演 test，"
            "test 不再完全独立，须核查来源或重建划分")
    doc = dup.get("document_level")
    doc = doc if isinstance(doc, dict) else {}
    n_doc_cross = _count(
        doc.get("train_blocks_dropped_as_duplicates_of_eval"))
    n_doc_intra = _count(
        doc.get("train_intra_split_duplicate_blocks_removed"))
    if n_doc_cross or n_doc_intra:
        add("info", "document_decontamination_applied",
            f"已从 train 移除多行块：与 val/test 重复 {n_doc_cross} 个、"
            f"train 内部重复 {n_doc_intra} 个"
            f"（共 {_count(doc.get('train_lines_dropped'))} 行；"
            "单行块交给行级去污染，val/test 未改写）")
    n_doc_eval_dup = _count(doc.get("val_test_overlap_blocks"))
    if n_doc_eval_dup:
        add("warning", "eval_splits_overlap_documents",
            f"val 与 test 有 {n_doc_eval_dup} 个多行块重复：两侧均保持尘封"
            "未改写，但 val 选型会部分预演 test，最终 test 指标的独立性下降，"
            "须核查/重建划分")
    near = dup.get("near_duplicate")
    near = near if isinstance(near, dict) else {}
    n_near_cross = _count(
        near.get("train_blocks_dropped_as_near_duplicates_of_eval"))
    n_near_intra = _count(
        near.get("train_intra_split_near_duplicates_removed"))
    if n_near_cross or n_near_intra:
        add("info", "near_duplicate_decontamination_applied",
            f"已从 train 移除近似重复块：与 val/test 近似 {n_near_cross} 个、"
            f"train 内部近似重复 {n_near_intra} 个"
            f"（共 {_count(near.get('train_lines_dropped'))} 行；"
            "val/test 未改写；阈值与三档敏感性见 duplicates.near_duplicate）")
    n_near_vt = _count(near.get("val_test_near_duplicate_pairs"))
    if n_near_vt:
        add("warning", "eval_splits_near_duplicates",
            f"val 与 test 有 {n_near_vt} 对近似重复块（阈值 "
            f"{near.get('threshold', '?')}）：两侧均保持尘封未改写，但 val 选型"
            "会部分预演 test，最终 test 指标的独立性下降，须核查/重建划分")
    if near.get("candidate_pairs_truncated") or _count(
            near.get("oversized_groups_truncated")):
        add("warning", "near_duplicate_scan_truncated",
            "近似重复扫描触发了候选簇/候选总数上限，可能有未核实的近似副本；"
            "报告中的删除量是下界，须降低语料规模或提高上限后重跑")
    frac = _num(near.get("removal_fraction_chars"))
    if frac is not None and frac > 0.25:
        add("warning", "near_duplicate_removal_high_fraction",
            f"近似去污染将移除 train 可用字符的 {frac:.1%}（>25%）：可能语料"
            "高度重复或阈值过宽，须人工核查 duplicates.near_duplicate 的"
            "阈值敏感性与示例后再决定是否沿用该预处理")
    if _count(near.get("train_blocks_skipped_too_large")):
        add("warning", "near_duplicate_scan_incomplete",
            f"有 {_count(near.get('train_blocks_skipped_too_large'))} 个 train 块"
            f"超过 {near.get('max_block_chars', NEAR_DUP_MAX_BLOCK_CHARS)} 字符"
            "未参与近似重复扫描：该块的近重复无法被排除，须人工核查")
    # 覆盖率：近重复结论只对“已扫描且达到 min_chars 的字符”成立。低于
    # 协议下限时必须显式说明剩余风险，不能把“未扫描”写成“无近重复”。
    coverage = _num(near.get("train_chars_coverage_fraction"))
    min_cov = _num(near.get("near_duplicate_min_coverage"))
    if min_cov is None:
        min_cov = float(NEAR_DUP_MIN_COVERAGE)
    if coverage is not None and coverage < min_cov:
        add("warning", "near_duplicate_low_coverage",
            f"近似重复扫描仅覆盖 train 全部字符的 {coverage:.1%}"
            f"（协议下限 {min_cov:.0%}；短块 "
            f"{_count(near.get('train_chars_skipped_short')):,} 字符、"
            f"少 shingle "
            f"{_count(near.get('train_chars_skipped_few_shingles')):,} 字符、"
            f"超大块 "
            f"{_count(near.get('train_chars_skipped_too_large')):,} 字符未扫描）："
            "近重复结论只对已覆盖部分成立，须人工核查未扫描语料")
    # 清洗后复核：证明“剩下的已干净”，而不只是“删过一些”。
    post = dup.get("post_cleanup")
    post = post if isinstance(post, dict) else {}
    if post.get("checked_pairs"):
        if post.get("train_eval_clean") is False:
            add("warning", "post_cleanup_residual_overlap",
                "清洗后复核发现 train 仍与 val/test 共享精确/归一化长行或多行块："
                f"长行 {post.get('overlap_long_lines', {})}、多行块 "
                f"{post.get('overlap_blocks', {})}。去污染并非闭合，"
                "须核查删除顺序/短行边界并重建数据")
        elif post.get("train_eval_clean") is True:
            add("info", "post_cleanup_verified",
                "清洗后复核：train 与 val/test 已无共享精确/归一化长行或多行块"
                "（短行与近似重复另见对应条目；val/test 保持尘封未改写）")
    n_eval_dup = _count(dup.get("val_test_overlap_long_lines"))
    n_eval_dup_norm = _count(
        dup.get("val_test_overlap_normalized_long_lines"))
    if n_eval_dup or n_eval_dup_norm:
        add("warning", "eval_splits_overlap",
            f"val 与 test 有长行重复：精确 {n_eval_dup} 条、空白归一化后共 "
            f"{n_eval_dup_norm} 条（含精确重复）：两侧均保持尘封未改写，但 "
            "val 选型会部分预演 test，最终 test 指标的独立性下降，"
            "须核查/重建划分")
    for split in ("val", "test"):
        n_dup = _count(dup.get(f"{split}_intra_split_duplicate_lines"))
        n_dup_norm = _count(
            dup.get(f"{split}_intra_split_normalized_duplicate_lines"))
        if n_dup or n_dup_norm:
            # 统计键覆盖所有非空行（含短行），且归一化计数包含精确重复；
            # 只读精确计数会漏掉“仅空白/NFKC 差异”的重复，文案也不能写成
            # “长行”（与 decontaminate_texts 的口径不一致）。
            if n_dup_norm:
                detail = (f"空白归一化后 {n_dup_norm} 条（其中精确 {n_dup} 条）")
            else:
                detail = f"精确 {n_dup} 条"
            add("info", f"{split}_intra_duplicates",
                f"{split} 内部有重复行：{detail}（未改写，避免破坏尘封；"
                "重复会高估评测有效样本量，解读置信区间时须注意）")
    text_stats = text_stats if isinstance(text_stats, dict) else {}
    for split, st in text_stats.items():
        st = st if isinstance(st, dict) else {}
        n_huge = _count(st.get("lines_gt_huge_threshold"))
        if n_huge:
            add("info", f"huge_lines_{split}",
                f"{split} 有 {n_huge} 行超过 "
                f"{st.get('huge_line_threshold_chars')} 字符"
                "（异常值已计数未删除，训练窗口可能被其主导）")
    return alerts


# 会让 train/val/test 独立性失效的协议违规：run_all 在训练前据此硬失败
# （可显式覆盖），避免“报告里有警告但全量训练照跑”。
# 划分协议违规（群组跨 split / 字符切分 / 打乱）之外，采样 64-gram 跨 split
# 重叠率 >0 是“语料在 split 间被复制”的直接证据——与分布漂移（temporal split
# 下属正常现象）不同，它必然导致评测读到训练见过的长串，必须训练前拦截。


def critical_split_alerts(data_report: dict) -> list:
    """从 data_report 中筛出必须在训练前处理的划分协议违规。"""
    if not isinstance(data_report, dict):
        return []
    alerts = data_report.get("alerts")
    if not isinstance(alerts, list):
        return []
    return [a for a in alerts
            if isinstance(a, dict)
            and a.get("code") in CRITICAL_SPLIT_ALERT_CODES]


def split_protocol_alerts(split_protocol: dict) -> list:
    """把上游划分协议的群组/时序风险转成显式告警（元数据不能“静默通过”）。

    split_protocol 来自 prepare_data 的 temporal_split_text info 或官方 split
    的静态描述。group_unit 定义的群组（如 heading_document）一旦跨 split，
    train/val/test 就不再独立——必须与 ngram 重叠同级进入 alerts，
    供汇总页与复核者看到，而不是只躺在报告元数据里。
    """
    alerts = []
    p = split_protocol if isinstance(split_protocol, dict) else {}
    if not p:
        return alerts

    def add(level, code, message):
        alerts.append({"level": level, "code": code, "message": message})

    if p.get("shuffled"):
        add("warning", "split_shuffled",
            "划分协议声明 shuffled=True：时序先后/群组完整性不可保证，须人工核查")
    dev = p.get("max_abs_fraction_deviation")
    tol = p.get("fraction_tolerance", SPLIT_FRACTION_TOLERANCE)
    if (isinstance(dev, (int, float)) and not isinstance(dev, bool)
            and isinstance(tol, (int, float)) and not isinstance(tol, bool)
            and dev > float(tol)):
        add("warning", "split_fraction_skew",
            f"实际 split 比例偏离目标 {dev:.1%}（> {float(tol):.1%}）："
            f"目标 {p.get('target_fractions', '?')}，实际 "
            f"{p.get('actual_fractions', '?')}；切点吸附到文档/段落边界会"
            "改变实际训练/评测规模，解读“80/10/10”与检验功效时须按实际值")
    if p.get("fallback_char_cut"):
        add("warning", "split_char_cut",
            "无可用换行或文档边界，退化为字符切分——同一文档可能跨 split"
            "（群组泄露），须人工核查")
    if p.get("documents_cross_cut"):
        add("warning", "split_documents_cross_cut",
            f"切点未落在文档边界：{p.get('group_unit', 'group')} 群组跨 split"
            f"（标题 {p.get('n_headings', '?')} 个、切点类型 "
            f"{p.get('cut_kinds', '?')}），违反“群组整体落在同一侧”，"
            "须调整划分比例或更换可整体划分的语料")
    if p.get("paragraphs_cross_cut"):
        add("warning", "split_paragraphs_cross_cut",
            f"切点未落在空行段落边界：{p.get('group_unit', 'group')} 群组跨 "
            f"split（段落边界 {p.get('n_paragraph_boundaries', '?')} 个、"
            f"切点类型 {p.get('cut_kinds', '?')}），违反“段落整体落在同一侧”，"
            "须调整划分比例或为语料补空行分隔")
    short = p.get("overlap_short_lines")
    if isinstance(short, dict):
        bad_short = {}
        for pair, val in short.items():
            if not isinstance(val, dict):
                continue
            exact = val.get("exact_lines") or 0
            norm = val.get("normalized_lines") or 0
            if exact or norm:
                bad_short[str(pair)] = {"exact_lines": exact,
                                        "normalized_lines": norm}
        if bad_short:
            add("warning", "split_overlap_short_lines",
                f"实际文本存在跨 split 共享短行 {bad_short}"
                f"（<{p.get('min_chars', '?')} 字符，多为标题/固定用语）；"
                "未自动删除，但重复样本会削弱 held-out 独立性，须人工核查")
    verified = p.get("verified_disjoint")
    overlaps = p.get("overlap_long_lines")
    bad = {}
    if isinstance(overlaps, dict):
        for pair, val in overlaps.items():
            if not isinstance(val, dict):
                continue
            exact = val.get("exact_lines") or 0
            norm = val.get("normalized_lines") or 0
            if exact or norm:
                bad[str(pair)] = {"exact_lines": exact,
                                  "normalized_lines": norm}
    if bad:
        if verified is True:
            add("warning", "split_overlap_long_lines",
                f"上游声明 verified_disjoint=True，但实际文本存在跨 split "
                f"共享长行 {bad}：声明与内容矛盾，不能按“已核对互斥”处理；"
                "去污染只能移除 train 侧副本，val/test 之间或清洗后仍重叠的"
                "样本会削弱 held-out 独立性，须核查来源或重建划分")
        else:
            add("warning", "split_overlap_long_lines",
                f"上游划分声明并非互斥：实际文本存在跨 split 共享长行 {bad}；"
                "去污染只能移除 train 侧副本，val/test 之间或清洗后仍重叠的"
                "样本会削弱 held-out 独立性，须核查来源或重建划分")
    missing = p.get("missing_files")
    if isinstance(missing, list) and missing:
        prefix = "上游声明 verified_disjoint=True，但" if verified is True else ""
        add("warning", "split_protocol_unverified",
            f"{prefix}缺少 split 文件 {missing}，无法核对“无共享样本”声明；"
            "缺失 split 的评测指标不可与其他 split 直接比较")
    if verified is False and not bad and not (
            isinstance(missing, list) and missing):
        # verified_disjoint=False 本身就是“未验证通过”。没有重叠明细也没有
        # 缺失文件时不能当成通过：报告被手改/上游漏写证据，同样无法确认
        # 三个 split 互斥。
        add("warning", "split_protocol_unverified",
            "上游声明 verified_disjoint=False，但未记录跨 split 长行重叠或"
            "缺失文件：无法确认划分互斥性")
    return alerts


def _text_line_digest_sets(text: str, min_chars: int):
    """返回一行文本的 (长行精确/归一化, 短行精确/归一化) 摘要集合。

    与 decontaminate_texts / verify_splits_disjoint 同口径：按归一化内容
    长度划分长短行，避免“仅空白差异”让同一行在两侧落入不同集合。
    """
    long_exact, long_norm = set(), set()
    short_exact, short_norm = set(), set()
    if not isinstance(text, str):
        return long_exact, long_norm, short_exact, short_norm
    for line in _iter_lines(text):
        s = line.strip()
        if not s:
            continue
        sn = _normalize_line(s)
        d = _line_digest(s)
        dn = d if sn == s else _line_digest(sn)
        if len(sn) < min_chars:
            short_exact.add(d)
            short_norm.add(dn)
        else:
            long_exact.add(d)
            long_norm.add(dn)
    return long_exact, long_norm, short_exact, short_norm


def post_cleanup_residual(texts: dict, min_chars: int = 40,
                          min_lines: int = 2) -> dict:
    """清洗后复核：最终 train 里是否仍残留与 val/test 相同的行/多行块。

    去污染函数只报告“删了多少”，这不能证明“剩下的已干净”。本函数对清洗
    后的文本再做一次独立核对（不重新删除、不改写 val/test），把“已应用
    去污染”升级为“已在当前文本上验证无精确/归一化重复”。它只覆盖精确
    行/块，不替代近似重复扫描；短行（标题/固定用语）单独计数，不因
    “已核对”而假装它们不存在。

    返回 JSON 原生类型；``train_eval_clean`` 仅在 train 与至少一个 eval
    split 都存在时给出布尔值，否则为 None（不可判定，不写成通过）。
    """
    min_chars = strict_int(min_chars, "min_chars")
    min_lines = strict_int(min_lines, "min_lines")
    if min_chars < 0:
        raise ValueError(f"min_chars 必须 >= 0，收到 {min_chars!r}")
    if min_lines < 1:
        raise ValueError(f"min_lines 必须 >= 1，收到 {min_lines!r}")
    texts = texts if isinstance(texts, dict) else {}
    present = [s for s in ("train", "val", "test")
               if isinstance(texts.get(s), str)]
    line_sets, block_sets = {}, {}
    for split in present:
        line_sets[split] = _text_line_digest_sets(texts[split], min_chars)
        block_sets[split] = {d for _, _, d, _ in
                             _iter_blocks(texts[split], min_lines)}

    def _pair(a, b):
        return f"{a}_{b}"

    pairs = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            pairs.append((a, b))
    overlap_long, overlap_short, overlap_blocks = {}, {}, {}
    for a, b in pairs:
        la, lan, sa, san = line_sets[a]
        lb, lbn, sb, sbn = line_sets[b]
        key = _pair(a, b)
        overlap_long[key] = {
            "exact_lines": int(len(la & lb)),
            "normalized_lines": int(len(lan & lbn))}
        overlap_short[key] = {
            "exact_lines": int(len(sa & sb)),
            "normalized_lines": int(len(san & sbn))}
        overlap_blocks[key] = int(len(block_sets[a] & block_sets[b]))
    train_pairs = [p for p in pairs if "train" in p]
    clean = None
    if train_pairs:
        clean = all(
            overlap_long[_pair(*p)]["exact_lines"] == 0
            and overlap_long[_pair(*p)]["normalized_lines"] == 0
            and overlap_blocks[_pair(*p)] == 0
            for p in train_pairs)
    vt_key = "val_test"
    return {
        "scope": ("exact/whitespace-normalized long lines and multi-line "
                  "blocks after all decontamination passes"),
        "min_chars": int(min_chars),
        "min_lines": int(min_lines),
        "checked_splits": present,
        "checked_pairs": [list(p) for p in pairs],
        "overlap_long_lines": overlap_long,
        "overlap_short_lines": overlap_short,
        "overlap_blocks": overlap_blocks,
        # 只对 train↔eval 下结论：val/test 之间的重复按协议保持尘封不改写，
        # 只能诊断并告警，不能算作“清洗失败”。
        "train_eval_clean": clean,
        "val_test_overlap_long_lines": (
            overlap_long.get(vt_key, {}).get("normalized_lines", 0)),
        "val_test_overlap_blocks": int(overlap_blocks.get(vt_key, 0)),
        "note": ("短行重复只计数；val/test 不被改写。本核对不覆盖低于 "
                 "min_chars 的短行或近似重复，那部分由 near_duplicate 报告"),
    }


def verify_splits_disjoint(txt_paths: dict, min_chars: int = 40,
                           required_splits=("train", "val", "test")) -> dict:
    """读回已写入的 train/val/test.txt，核对跨 split 无共享长行。

    split_protocol 里 "document_disjoint(no_shuffle)" / "无共享样本" 是上游
    来源的**声明**，不能只靠断言：这里对实际写出的文本做一次可复核的检查——
    若任意两个 split 存在相同长行（逐字符相同或 NFKC/空白归一化后相同），
    说明三个文件并非真正互斥，"无共享样本" 的协议被证伪。下游 prepare_bins
    也会做去污染诊断，这里只是把元数据声明与其实际内容绑定，避免报告
    "声明一套、数据一套"。

    文件缺失/路径为空时不能把"空集合交集"当成"已核对无重叠"：缺失的 split
    必须显式列入 missing_files 并令 verified_disjoint=False，否则报告会把
    "没有可核对的文件"伪装成"核对通过"。

    短行（按归一化内容长度 < min_chars）不计入 verified_disjoint（标题/固定
    用语的重复是正常现象），但会在 overlap_short_lines 中单独计数，避免短
    样本跨 split 复制成为既不删除也不报告的盲区。

    required_splits 默认要求 train/val/test 三个文件都存在；只有显式声明
    “本数据集没有 test（回退 val）”的调用方（如 dummy 生成器）才可传子集。
    """
    # 阈值本身必须是严格整数：min_chars=True 会让所有行都变成“短行”，
    # 长行重叠被跳过而 verified_disjoint 仍可能为 True，等于伪造核对通过。
    min_chars = strict_int(min_chars, "min_chars")
    if min_chars < 0:
        raise ValueError(f"min_chars 必须 >= 0，收到 {min_chars!r}")
    paths = txt_paths if isinstance(txt_paths, dict) else {}
    if isinstance(required_splits, str):
        raise ValueError("required_splits 需要字符串序列，不能是单个字符串")
    try:
        required = tuple(str(x) for x in required_splits)
    except TypeError as e:
        raise ValueError(
            f"required_splits 需要字符串序列，收到 {required_splits!r}") from e
    if not required:
        raise ValueError("required_splits 不能为空")
    missing = []
    sets = {}
    short_sets = {}
    for split, path in paths.items():
        if not path or not os.path.exists(path):
            missing.append(split)
            continue
        exact, norm = set(), set()
        short, short_norm = set(), set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                sn = _normalize_line(s)
                # 与 decontaminate_texts 同口径：按归一化内容长度分类，
                # 避免同一行因空白/NFKC 差异在两侧落入 long/short 不同集合。
                if len(sn) < min_chars:
                    short.add(_line_digest(s))
                    short_norm.add(_line_digest(sn))
                    continue
                exact.add(_line_digest(s))
                norm.add(_line_digest(sn))
        sets[split] = (exact, norm)
        short_sets[split] = (short, short_norm)
    # 必需 split 缺失时不能因“只比较了碰巧提供的文件”而声称已验证。
    for split in required:
        if split not in paths and split not in missing:
            missing.append(split)
    overlaps = {}
    overlaps_short = {}
    # 不能只比较硬编码的 train/val/test 三对：required_splits 允许调用方
    # 声明 dev 等额外 held-out。只比较固定三对会让 train 与 dev 的重复
    # 完全不被检查，却仍返回 verified_disjoint=True。
    ordered = ([s for s in ("train", "val", "test") if s in sets]
               + sorted(s for s in sets if s not in ("train", "val", "test")))
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            overlaps[f"{a}_{b}"] = {
                "exact_lines": len(sets[a][0] & sets[b][0]),
                "normalized_lines": len(sets[a][1] & sets[b][1])}
            overlaps_short[f"{a}_{b}"] = {
                "exact_lines": len(short_sets[a][0] & short_sets[b][0]),
                "normalized_lines": len(short_sets[a][1] & short_sets[b][1])}
    disjoint = (not missing and bool(overlaps)
                and all(v["exact_lines"] == 0
                        and v["normalized_lines"] == 0
                        for v in overlaps.values()))
    return {"verified_disjoint": bool(disjoint),
            "overlap_long_lines": overlaps,
            "overlap_short_lines": overlaps_short,
            "missing_files": missing,
            "min_chars": int(min_chars)}


def short_overlap_alerts(short_contamination: dict) -> list:
    """短 n-gram 重叠的告警（只警告、不进入硬闸门）。

    短 n（32）比 64-gram 更易被自然语言常见搭配命中，>0 不必然是泄露，但
    模板化文本的整句复用会在这里暴露；提示人工核查而非阻断训练。
    """
    alerts = []
    c = short_contamination if isinstance(short_contamination, dict) else {}
    for key, label in (("train_val", "train/val"), ("train_test", "train/test"),
                       ("val_test", "val/test")):
        d = c.get(key)
        d = d if isinstance(d, dict) else {}
        raw = d.get("overlap_rate")
        # bool 是 int 子类（True→1.0）：不能把 true 当成 100% 重叠率触发告警。
        if raw is None or isinstance(raw, (bool, np.bool_)):
            continue
        if isinstance(raw, str) and "_" in raw:
            continue
        try:
            rate = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(rate) or not 0.0 < rate <= 1.0:
            continue
        alerts.append({
            "level": "warning", "code": f"ngram_overlap_short_{key}",
            "message": (f"{label} 采样 {d.get('n', '?')}-gram 重叠率 "
                        f"{rate:.4%}（<64 token 的整句/模板复用；短 n 易有"
                        "自然搭配假阳性，须人工核查而非自动阻断）")})
    return alerts


def build_data_report(tokenizer: dict, splits: dict, sanitization: dict,
                      duplicates: dict, contamination: dict,
                      distribution: dict, text_stats: dict = None,
                      extra: dict = None,
                      alert_thresholds: dict = None) -> dict:
    """汇总为 data_report.json（只含 JSON 原生类型）。"""
    alert_thresholds = (alert_thresholds
                        if isinstance(alert_thresholds, dict) else {})
    extra = extra if isinstance(extra, dict) else None
    # “先划分后预处理”只能由上游划分协议证据支撑。缺少 split_protocol 时
    # 写 None（未知）而不是写死 True，避免报告做出无法证实的审计声明；
    # 同时产生告警，让“无法确认”与“已确认”在汇总页可区分。
    split_protocol = extra.get("split_protocol") if extra else None
    split_first_attested = (isinstance(split_protocol, dict)
                            and bool(split_protocol))
    # 划分是否打乱同样只能由上游协议声明：无声明时写 None（未知），
    # 不能把“没记录”当成“没打乱”。
    shuffle_at_split = None
    if isinstance(split_protocol, dict):
        value = split_protocol.get("shuffled")
        if isinstance(value, bool):
            shuffle_at_split = value
    alerts = leakage_alerts(tokenizer, sanitization, duplicates,
                            contamination, distribution, text_stats,
                            thresholds=alert_thresholds)
    if not split_first_attested:
        alerts.append({
            "level": "warning", "code": "split_first_unattested",
            "message": "报告缺少 split_protocol 证据：无法确认数据是“先划分"
                       "后预处理且划分未打乱”，只可确认分词/去污染在 train 上完成"})
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "split_first_then_preprocess": (True if split_first_attested
                                            else None),
            "split_first_then_preprocess_attested": bool(split_first_attested),
            # 必须如实记录分词器实际拟合来源：字符级词表在 train 上拟合，
            # 而 GPT-2 BPE 是公开预训练词表（与语料无关、不构成泄露）。统一
            # 写死 "train_only" 会与 tokenizer.fit_on 矛盾，使“无全量拟合”
            # 的审计声明失真。
            # 缺失时写 None（未知），不能默认成安全的 "train_only"：
            # 报告没有来源证据时，预检必须能区分“未知”与“已在 train 上拟合”。
            "fit_on": (tokenizer.get("fit_on")
                       if isinstance(tokenizer, dict) else None),
            "test_sealed": True,
            "shuffle_at_split_time": shuffle_at_split,
            "batch_shuffle_during_training_allowed": True,
        },
        "tokenizer": tokenizer,
        "splits": splits,
        "sanitization": sanitization,
        "text_stats": text_stats or {},
        "duplicates": duplicates,
        "contamination": contamination,
        "distribution_shift": distribution,
        "alert_thresholds": dict(ALERT_THRESHOLDS, **alert_thresholds),
        "alerts": alerts,
        "notes": [
            "所有拟合类统计（字符词表等）只在 train 上完成，val/test 只做 transform；",
            "去污染只改写 train：先删除与 val/test 重复的多行块及 train 内部"
            "重复块（单行块交给行级逻辑），再移除与 val/test 重复的长行并删除"
            "train 内部重复长行（保留首次出现）；val/test 不被改写；",
            "精确去重之后再做近似重复筛查：字符 5-gram MinHash + LSH 找候选，"
            "候选对精确计算 Jaccard/containment，跨 split 用 max(Jaccard, "
            "containment)、train 内部只用 Jaccard，阈值与三档敏感性（判定对数"
            "与真实删除量）见 duplicates.near_duplicate；只有 train 被改写；",
            "近重复扫描只覆盖达到 min_chars 且 shingle 数达标的字符："
            "duplicates.near_duplicate 同时报告覆盖率与被跳过字符数，"
            "低于协议下限会告警，不能把未扫描部分当作已验证；",
            "所有删除完成后，duplicates.post_cleanup 对最终文本再核对一次"
            "train 与 val/test 是否仍共享精确/归一化长行或多行块；"
            "train_eval_clean=False 进入 alerts 并作为训练前严重违规；",
            "accounting 逐 split 记录 tokens/字符/UTF-8 字节与 bytes/token："
            "token 级 PPL 的分母随分词器变化，规模比较以字节/字符核算为准；",
            "val 与 test 之间的重复/64-gram 重叠单独统计并告警：两侧均不改写，"
            "但重叠会削弱 final test 相对 val 选择集的独立性；",
            "64-gram 重叠为采样筛查下界，>0 时应人工核查对应语料；",
            "一元分布漂移默认全量精确计数（显式 max_samples 时才采样），"
            "用于提示 split 间分布差异而非统计检验；",
            "alerts 同时包含协议违规（群组/时序，见 split_protocol）与阈值触发的"
            "复核提示（见 alert_thresholds），不是显著性检验。",
        ],
    }
    # 语料规模核算：token 数之外同时记录字符数与 UTF-8 字节数。PPL 的
    # 分母是 token，字符级兜底与 BPE 的 token 粒度不同；bytes/token 与
    # 总字节数让读者能判断不同分词器/不同数据版本的结果是否可比，也便于
    # 发现“token 数正常但字节数异常”的分词/清洗事故。
    def _account_int(value):
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            n = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return n if n >= 0 else None

    accounting = {}
    splits_d = splits if isinstance(splits, dict) else {}
    text_stats_d = text_stats if isinstance(text_stats, dict) else {}
    for split in ("train", "val", "test"):
        st = splits_d.get(split)
        ts = text_stats_d.get(split)
        st = st if isinstance(st, dict) else {}
        ts = ts if isinstance(ts, dict) else {}
        tokens = _account_int(st.get("tokens"))
        chars = _account_int(ts.get("chars"))
        nbytes = _account_int(ts.get("utf8_bytes"))
        entry = {
            "tokens": tokens,
            "chars": chars,
            "utf8_bytes": nbytes,
            "chars_per_token": None,
            "bytes_per_token": None,
            "bytes_per_char": None,
        }
        if tokens and chars is not None:
            entry["chars_per_token"] = float(chars / tokens)
        if tokens and nbytes is not None:
            entry["bytes_per_token"] = float(nbytes / tokens)
        if chars and nbytes is not None:
            entry["bytes_per_char"] = float(nbytes / chars)
        accounting[split] = entry
    report["accounting"] = accounting
    if isinstance(extra, dict):
        # extra 只允许扩充报告，不能覆盖自检结论本身：否则上游元数据里的
        # 一个 "alerts": [] / "protocol": {...} 就能把防泄漏告警清空或把
        # 协议声明改成“安全”，而 meta/report 哈希仍然自洽。冲突键显式记录
        # 供审计，但不采纳其值。
        protected = set(report)
        ignored = sorted(str(k) for k in extra if k in protected)
        for key, value in extra.items():
            if key not in protected:
                report[key] = value
        if ignored:
            report["ignored_extra_keys"] = ignored
    # 协议违规（群组跨 split / 字符切分 / 打乱）置顶，确保汇总页的告警截断
    # 不会把更严重的问题挤出视野。
    protocol_alerts = split_protocol_alerts(report.get("split_protocol"))
    if protocol_alerts:
        report["alerts"] = protocol_alerts + report["alerts"]
    return report
