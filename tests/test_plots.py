"""可视化结果文件选择测试：同一 (变体, 种子) 不能重复计数。

在旧结果目录上以不同种子数重跑时，裸命名文件与 _seed<s> 文件可能同时
存在（并指向同一个种子）。图表的均值/误差棒必须按 repro.json 的命名
约定只取本次 run 的产物，否则同一次实验会被多算一次、过期结果混入均值。
"""
import json
import os
import shutil
import tempfile

import numpy as np

from lega.plots import (_HAS_MPL, _expected_tagged, _group_by_variant,
                        _group_items, _load_train_logs, _mean_by_step,
                        _ppl_split_label, _prefer_seeded, _read_records,
                        fig_auroc_heads, fig_latency, fig_misleading,
                        fig_ppl, fig_ruler)


def _w(rdir, name, obj):
    with open(os.path.join(rdir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_group_by_variant_deduplicates_seed_slots_by_convention():
    tmp = tempfile.mkdtemp()
    try:
        # 多种子约定：过期的裸命名文件不得与 _seed1 重复计入
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        _w(tmp, "eval_lega.json", {"variant": "lega", "seed": 1, "val_ppl": 999.0})
        _w(tmp, "eval_lega_seed1.json", {"variant": "lega", "seed": 1, "val_ppl": 10.0})
        _w(tmp, "eval_lega_seed2.json", {"variant": "lega", "seed": 2, "val_ppl": 11.0})
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g["lega"]) == [10.0, 11.0]
        # 单种子约定：当前裸命名文件胜出，过期的 _seed 文件不得混入
        _w(tmp, "repro.json", {"seeds": [1]})
        _w(tmp, "eval_lega.json", {"variant": "lega", "seed": 1, "val_ppl": 12.0})
        g2 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g2["lega"]) == [12.0]
        # 无 repro 信息：退回旧行为（_seed 优先，与 _seed1 冲突的裸文件丢弃）
        os.remove(os.path.join(tmp, "repro.json"))
        g3 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert sorted(d["val_ppl"] for d in g3["lega"]) == [10.0, 11.0]
        # 无任何种子信息才保留（旧格式单文件）
        for name in ("eval_lega_seed1.json", "eval_lega_seed2.json"):
            os.remove(os.path.join(tmp, name))
        _w(tmp, "eval_lega.json", {"variant": "lega", "val_ppl": 8.0})
        g4 = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert [d["val_ppl"] for d in g4["lega"]] == [8.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 变体分组：按 (变体, 种子) 去重且随命名约定择优")


def test_group_by_variant_keeps_bare_only_variant():
    """命名约定筛选必须逐变体进行：一个变体只有旧命名（无 _seed 后缀）的
    结果时，不能因为别的变体有 _seed 文件而被整体丢弃。"""
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        _w(tmp, "eval_lega_seed1.json",
           {"variant": "lega", "seed": 1, "val_ppl": 10.0})
        _w(tmp, "eval_lega_seed2.json",
           {"variant": "lega", "seed": 2, "val_ppl": 11.0})
        _w(tmp, "eval_standard.json", {"variant": "standard", "val_ppl": 8.0})
        _w(tmp, "eval_broken.json", [1, 2, 3])  # 合法 JSON 非对象：跳过不崩
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert set(g) == {"lega", "standard"}, g
        assert sorted(d["val_ppl"] for d in g["lega"]) == [10.0, 11.0]
        assert [d["val_ppl"] for d in g["standard"]] == [8.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 变体分组：仅有另一族命名的变体不被整体丢弃（坏文件跳过）")


def _w_log(root, variant, seed, loss):
    d = os.path.join(root, variant) if seed is None \
        else os.path.join(root, variant, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "train_log.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 1, "loss": loss}) + "\n")


def test_load_train_logs_respects_run_convention():
    """同一变体下旧布局（裸命名）与本次多种子布局的 train_log 并存时，
    按 repro.json 约定只取本次日志做均值；仅有另一族日志的变体仍保留。"""
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        _w_log(tmp, "lega", 1, 1.0)
        _w_log(tmp, "lega", 2, 3.0)
        _w_log(tmp, "lega", None, 9.0)       # 旧单种子布局的过期日志
        _w_log(tmp, "standard", None, 8.0)   # 该变体仅有旧布局日志
        series = _load_train_logs(tmp, rdir=tmp)
        xs, ys = _mean_by_step(series["lega"], "loss")
        # 只取本次的 seed1/seed2（均值 2.0），不得把过期的 9.0 混入（4.33）
        assert xs == [1] and ys == [2.0], (xs, ys)
        assert len(series["standard"]) == 1
        # 单种子约定：只取裸命名文件，过期的 _seed 日志不参与
        _w(tmp, "repro.json", {"seeds": [1]})
        series = _load_train_logs(tmp, rdir=tmp)
        _, ys = _mean_by_step(series["lega"], "loss")
        assert ys == [9.0], ys
        # 无 repro 信息：退回旧行为（两种布局都画，兼容手工目录）
        os.remove(os.path.join(tmp, "repro.json"))
        series = _load_train_logs(tmp, rdir=tmp)
        assert len(series["lega"]) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练曲线日志：按 repro.json 命名约定逐变体择优")


def test_run_scope_filters_stale_seed_and_variant_artifacts():
    """repro.json 的具体 seeds/variants 白名单必须生效：同后缀的旧种子
    （如 seeds 7,9 目录里的 _seed8）与已移除变体都不得混入本次图表。"""
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": [7, 9], "variants": ["lega"]})
        for s, ppl in ((7, 10.0), (8, 99.0), (9, 11.0)):
            _w(tmp, f"eval_lega_seed{s}.json",
               {"variant": "lega", "seed": s, "val_ppl": ppl})
        _w(tmp, "eval_standard_seed7.json",
           {"variant": "standard", "seed": 7, "val_ppl": 1.0})
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert set(g) == {"lega"}, g
        assert sorted(d["val_ppl"] for d in g["lega"]) == [10.0, 11.0]
        paths = [os.path.join(tmp, f"eval_lega_seed{s}.json") for s in (7, 8)]
        assert _prefer_seeded(paths, rdir=tmp) == [paths[0]]
        # 训练日志同样不得混入旧 seed8 / 已移除变体
        for s in (7, 8, 9):
            _w_log(tmp, "lega", s, float(s))
        _w_log(tmp, "standard", 7, 1.0)
        series = _load_train_logs(tmp, rdir=tmp)
        assert set(series) == {"lega"}, series
        _, ys = _mean_by_step(series["lega"], "loss")
        assert ys == [8.0], ys       # (7+9)/2，而不是 (7+8+9)/3
        # 无 repro 信息时保持旧行为（全量发现）
        os.remove(os.path.join(tmp, "repro.json"))
        series = _load_train_logs(tmp, rdir=tmp)
        assert set(series) == {"lega", "standard"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] repro 范围：旧种子/已移除变体不混入图表")


def test_empty_or_fractional_repro_seeds_do_not_scan_stale_artifacts():
    """seeds=[] 是显式“无种子”；小数种子不能被 int() 截断后混入整数范围。"""
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": []})
        _w(tmp, "eval_lega_seed1.json",
           {"variant": "lega", "seed": 1, "val_ppl": 10.0})
        _w(tmp, "analysis_lega_seed1.json",
           {"variant": "lega", "seed": 1, "n_records": 1})
        assert _expected_tagged(tmp) is False
        from lega.plots import _run_scope
        assert _run_scope(tmp)[1] == set()
        assert _group_by_variant(os.path.join(tmp, "eval_*.json")) == {}
        assert _load_train_logs(tmp, rdir=tmp) == {}
        # 小数种子不是整数身份：即使同目录白名单只有整数 1，也不相等
        os.remove(os.path.join(tmp, "eval_lega_seed1.json"))
        _w(tmp, "repro.json", {"seeds": [1]})
        _w(tmp, "eval_lega.json",
           {"variant": "lega", "seed": 1.9, "val_ppl": 99.0})
        from lega.plots import _seed_in_scope
        assert _seed_in_scope(1.9, {1}) is False
        assert _seed_in_scope("1", {1}) is True
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert g == {}, g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] repro 范围：空/小数种子不混入旧结果")


def test_bool_seeds_are_not_integer_seed_one():
    """``"seed": true`` / ``seeds: [true]`` 不得被当成种子 1。

    bool 是 int 的子类：``float(True) is 1.0`` 会让白名单 {1} 接受 True，
    dict 里 True 与 1 又是同一个键——两个不同的 run 会占用同一去重槽位，
    过期产物也会冒充本次 run 的结果。
    """
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": [True]})
        _w(tmp, "eval_lega_seed1.json",
           {"variant": "lega", "seed": 1, "val_ppl": 10.0})
        from lega.plots import _run_scope, _seed_in_scope, _seed_int
        # 白名单里的 bool 不是整数身份 → 视为显式无可用种子，不扫旧产物
        assert _run_scope(tmp)[1] == set()
        assert _group_by_variant(os.path.join(tmp, "eval_*.json")) == {}
        assert _seed_in_scope(True, {1}) is False
        assert _seed_in_scope(1, {1}) is True
        assert _seed_int(True) is None and _seed_int(False) is None
        assert _seed_int(1.0) == 1 and _seed_int("2") == 2
        assert _seed_int(1.5) is None and _seed_int([1]) is None
        # 单种子约定下，带 bool seed 的裸文件不能冒充 seed1 的结果
        _w(tmp, "repro.json", {"seeds": [1]})
        _w(tmp, "eval_lega.json",
           {"variant": "lega", "seed": True, "val_ppl": 99.0})
        g = _group_by_variant(os.path.join(tmp, "eval_*.json"))
        assert [d["val_ppl"] for d in g.get("lega", [])] == [10.0], g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] bool 种子既不等于 1，也不占用种子 1 的去重槽位")


def test_read_records_merges_ragged_headers_by_file_rows():
    """合并不同版本的逐记录 CSV 时，缺列必须按"该文件的行区间"补 NaN。

    旧版 CSV 可能缺少后来才加入的门控列（c/logit）：若统一在列尾补 NaN，
    后续文件的行会整体前移，isfinite 掩码按位置配对就会把 c 与错误的 g/hit
    配对，静默产出错误图表。这里直接用三份不同表头的文件锁定行对齐。
    """
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"seeds": [1, 2, 3]})
        rows_by_seed = {
            1: "g,hit\n0.1,1\n0.2,0\n",
            2: "g,hit,c\n0.3,1,30\n0.4,0,40\n",
            3: "g,hit,c,logit\n0.5,1,50,-5\n0.6,0,60,6\n",
        }
        for s, text in rows_by_seed.items():
            with open(os.path.join(tmp, f"analysis_records_lega_seed{s}.csv"),
                      "w", encoding="utf-8", newline="") as f:
                f.write(text)
        cols = _read_records(tmp, "lega")
        assert cols["g"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert cols["hit"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert np.isnan(cols["c"][:2]).all()
        assert cols["c"][2:].tolist() == [30.0, 40.0, 50.0, 60.0]
        assert np.isnan(cols["logit"][:4]).all()
        assert cols["logit"][4:].tolist() == [-5.0, 6.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 逐记录 CSV：不同表头按文件行区间对齐补 NaN")


def test_prefer_seeded_matches_run_convention():
    tmp = tempfile.mkdtemp()
    try:
        bare = os.path.join(tmp, "analysis_records_lega.csv")
        tagged = os.path.join(tmp, "analysis_records_lega_seed1.csv")
        for p in (bare, tagged):
            open(p, "w").close()
        _w(tmp, "repro.json", {"seeds": [1, 2]})
        assert _expected_tagged(tmp) is True
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
        _w(tmp, "repro.json", {"seeds": [1]})
        assert _expected_tagged(tmp) is False
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [bare]
        # 无 repro：_seed 优先（旧行为）
        os.remove(os.path.join(tmp, "repro.json"))
        assert _expected_tagged(tmp) is None
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
        # 约定族不存在时回退到另一族，而不是画不出图
        os.remove(bare)
        _w(tmp, "repro.json", {"seeds": [1]})
        assert _prefer_seeded([bare, tagged], rdir=tmp) == [tagged]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 记录文件选择：按 repro.json 命名约定取本次产物")


def test_train_logs_skip_non_dict_and_bad_steps():
    """日志中的 null/数组/坏 step 不能进入均值计算或让图崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "lega"))
        path = os.path.join(tmp, "lega", "train_log.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("null\n[1, 2]\n5\n")
            f.write('{"step": 1, "loss": 2.0, "val_ppl": 10.0}\n')
            f.write('{"step": "bad", "loss": 1.0}\n')
            f.write('{"step": 2, "loss": "bad"}\n')
            f.write('{"step": 3, "loss": 1.5, "val_ppl": 9.0}\n')
        series = _load_train_logs(tmp)
        assert list(series) == ["lega"]
        assert len(series["lega"][0]) == 4
        xs, ys = _mean_by_step(series["lega"], "loss")
        assert xs == [1, 3] and ys == [2.0, 1.5]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练日志：非 dict 行/坏 step/坏值安全跳过")


def test_mean_by_step_rejects_fractional_and_boolean_steps():
    """1.5/True 不能被 int() 截断后与 step=1 合并到同一 x 坐标；整数值
    浮点与整数字符串仍是合法 step，超大整数按坏点跳过。"""
    runs = [[{"step": 1, "loss": 2.0},
             {"step": 1.5, "loss": 99.0},
             {"step": True, "loss": 99.0},
             {"step": 2.0, "loss": 1.0},
             {"step": "3", "loss": 0.5},
             {"step": 10 ** 400, "loss": 0.0}]]
    xs, ys = _mean_by_step(runs, "loss")
    assert xs == [1, 2, 3], (xs, ys)
    assert ys == [2.0, 1.0, 0.5], ys
    print("  [ok] 训练曲线均值：小数/布尔 step 不截断合并")


def test_fig_latency_uses_each_variant_batch_when_no_common_batch():
    """没有共同 batch 档位时仍要为每个变体取自己的可用档位画线，
    不能拿第一个变体的 batch key 去查所有变体（否则整条线静默消失）。"""
    if not _HAS_MPL:
        print("  [跳过] 无 matplotlib，跳过 fig_latency 批次测试")
        return
    import matplotlib.axes as maxes
    from unittest import mock
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "latency.json", {"variants": {
            "standard": {"prefill_ms": {"256": {"8": 20.0}}},
            "qgate": {"prefill_ms": {"256": {"1": 12.0}},
                      "decode_ms_per_tok": {"256": 1.2}}}})
        labels = []
        orig = maxes.Axes.plot

        def spy(self, *a, **kw):
            if kw.get("marker") == "o":
                labels.append(kw.get("label"))
            return orig(self, *a, **kw)

        with mock.patch.object(maxes.Axes, "plot", spy):
            assert fig_latency(tmp) is True
        assert len(labels) == 2, labels
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] Prefill 图：无共同 batch 时按变体各自档位绘制")


def test_fig_latency_respects_variant_scope():
    """本次 run 范围外的变体不得出现在 decode 图上（旧结果可能仍留在目录）。"""
    if not _HAS_MPL:
        print("  [跳过] 无 matplotlib，跳过 fig_latency 范围测试")
        return
    import matplotlib.axes as maxes
    from unittest import mock
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "repro.json", {"variants": ["lega"], "seeds": [1]})
        _w(tmp, "latency.json", {"variants": {
            "lega": {"prefill_ms": {"256": {"1": 12.0}},
                     "decode_ms_per_tok": {"256": 1.2}},
            "stale": {"prefill_ms": {"256": {"1": 9.0}},
                      "decode_ms_per_tok": {"256": 0.9}}}})
        labels = []
        orig = maxes.Axes.plot

        def spy(self, *a, **kw):
            if kw.get("marker") in ("o", "s"):
                labels.append(kw.get("label"))
            return orig(self, *a, **kw)

        with mock.patch.object(maxes.Axes, "plot", spy):
            assert fig_latency(tmp) is True
        assert labels == ["LEGA(ours)", "LEGA(ours)"], labels
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 延迟图：范围外变体不进入 prefill/decode 曲线")


def test_ppl_split_label_reports_actual_split():
    """单一未知 split（如 train）必须显式标注，不能冒充验证集。"""
    assert _ppl_split_label({"val"}) == "验证集"
    assert _ppl_split_label({"test"}) == "测试集"
    assert _ppl_split_label({"train"}) == "train 集"
    assert _ppl_split_label({"val", "test"}) == "各评测集"
    assert _ppl_split_label({"train"}, invalid_split=True) == "各评测集"
    assert _ppl_split_label({None}) == "未知评测集"
    print("  [ok] PPL 图：未知 split 不冒充验证集")


def test_fig_ppl_falls_back_to_val_ppl_and_skips_bad_values():
    """ppl 为 null 时用 val_ppl，非数值 ppl 不能让画图崩溃。"""
    if not _HAS_MPL:
        print("  [跳过] 无 matplotlib，跳过 fig_ppl 渲染测试")
        return
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "eval_lega.json",
           {"variant": "lega", "ppl": None, "val_ppl": 10.0, "split": "val"})
        _w(tmp, "eval_bad.json",
           {"variant": "bad", "ppl": "oops", "val_ppl": None, "split": "val"})
        assert fig_ppl(tmp) is True
        assert os.path.exists(os.path.join(tmp, "fig_ppl.png"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] PPL 图：null ppl 回退 val_ppl，坏值不崩溃")


def test_group_items_treats_unhashable_seed_as_missing():
    """JSON seed 为数组/对象时按缺种子信息处理，不能作为字典键崩溃。"""
    items = [("eval_lega.json", {"variant": "lega", "seed": [1],
                                 "val_ppl": 9.0}),
             ("eval_standard.json", {"variant": "standard",
                                     "seed": {"n": 1}, "val_ppl": 10.0}),
             ("eval_bad.json", {"variant": ["bad"], "seed": 1}),
             ("eval_bad2.json", {"variant": {"x": 1}, "seed": 1})]
    groups = _group_items(items)
    assert len(groups["lega"]) == 1
    assert len(groups["standard"]) == 1
    assert set(groups) == {"lega", "standard"}
    print("  [ok] 分组：不可哈希 seed/变体按缺失处理")


def test_plots_tolerate_malformed_nested_fields():
    """畸形 split/seed/results/latency/per_head/CSV 单元格不能让画图崩溃。"""
    if not _HAS_MPL:
        print("  [跳过] 无 matplotlib，跳过畸形字段渲染测试")
        return
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "eval_lega_seed1.json", {
            "variant": "lega", "seed": 1, "val_ppl": 10.0,
            "split": ["val"]})
        _w(tmp, "eval_standard_seed1.json", {
            "variant": "standard", "seed": 1, "val_ppl": float("inf"),
            "split": "val"})
        assert fig_ppl(tmp) is True
        _w(tmp, "ruler_lega_seed1.json", {
            "variant": "lega", "seed": 1,
            "results": {"niah_synth": {"x": {"em": 0.5},
                                       "256": {"em": "bad"}},
                        "fwe": [1]}})
        assert fig_ruler(tmp) is True
        _w(tmp, "latency.json", {"variants": {
            "lega": [1],
            "standard": {"prefill_ms": {"256": {}},
                         "decode_ms_per_tok": {"256": "x"}},
            "qgate": {"prefill_ms": {"256": {"1": 12.0}},
                      "decode_ms_per_tok": {"256": 1.2},
                      "decode_prefill_ctx": {"256": "bad"}}}})
        assert fig_latency(tmp) is True
        _w(tmp, "analysis_lega_seed1.json", {
            "variant": "lega", "seed": 1, "n_records": 10,
            "p_hit_given_conc": "x", "p_hit_given_notconc": 0.2,
            "misleading_rate": float("nan"),
            "per_head": ["bad", {"layer": [1], "head": 0},
                         {"layer": 0, "head": 0,
                          "auroc_qterm": 0.6, "auroc_logit": 0.7}]})
        assert fig_misleading(tmp) is True
        assert fig_auroc_heads(tmp, variant="lega") is True
        with open(os.path.join(tmp, "analysis_records_lega.csv"), "w",
                  encoding="utf-8") as f:
            f.write("task,g,hit,c,log_n\nx,zzz,1,1,2\n")
        cols = _read_records(tmp, "lega")
        assert np.isnan(cols["g"][0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 图表：畸形嵌套字段/CSV 单元格安全跳过")


def test_fig_auroc_heads_tolerates_non_list_per_head():
    """analysis JSON 的 per_head 非列表/层头类型混杂时不崩溃。"""
    if not _HAS_MPL:
        print("  [跳过] 无 matplotlib，跳过 fig_auroc_heads 渲染测试")
        return
    tmp = tempfile.mkdtemp()
    try:
        _w(tmp, "analysis_lega.json", {"per_head": 5})
        assert fig_auroc_heads(tmp) is False
        _w(tmp, "analysis_lega.json", {"per_head": [
            {"layer": "x", "head": 0, "auroc_logit": 0.5,
             "auroc_qterm": 0.4},
            {"layer": 1, "head": 0, "auroc_logit": 0.6,
             "auroc_qterm": 0.5},
            {"layer": 10 ** 400, "head": 0, "auroc_logit": 0.7,
             "auroc_qterm": 0.6}]})
        assert fig_auroc_heads(tmp) is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 逐头 AUROC 图：per_head 非列表/类型混杂安全跳过")
