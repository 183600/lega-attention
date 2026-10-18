"""可视化结果文件选择测试：同一 (变体, 种子) 不能重复计数。

在旧结果目录上以不同种子数重跑时，裸命名文件与 _seed<s> 文件可能同时
存在（并指向同一个种子）。图表的均值/误差棒必须按 repro.json 的命名
约定只取本次 run 的产物，否则同一次实验会被多算一次、过期结果混入均值。
"""
import json
import os
import shutil
import tempfile

from lega.plots import (_expected_tagged, _group_by_variant, _load_train_logs,
                        _mean_by_step, _prefer_seeded)


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
