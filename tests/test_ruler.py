"""RULER 任务构造器测试：跨度有效性、答案一致性、打分函数。"""
import os
import random
import re
import shutil
import tempfile

from lega.data import CharTokenizer
from lega.ruler import (build_fwe, build_niah, build_vt, load_corpus_sents,
                        load_corpus_sents_with_source, score_prediction,
                        ContextTooSmall, TASKS, _stable_hash)

_TEXT = ("One of the special magic numbers for lyre is: 7021853.\n"
         "VAR A1 = 48213.\nVAR A2 = A1.\nVAR B1 = 99321.\nVAR B2 = B1.\n"
         "VAR C1 = 13579.\nVAR C2 = C1.\nVAR D1 = 24680.\nVAR D2 = D1.\n"
         "VAR E1 = 11235.\nVAR E2 = E1.\nVAR F1 = 81321.\nVAR F2 = F1.\n"
         "The grass is green. The sky is blue. What are the 10 most common words?\n"
         "ka ro mi ta lu sen dor vil pen tor nal bes qui mon fel gar hil jas ker lum 0123456789")


def _tok():
    return CharTokenizer(sorted(set(_TEXT)))


def _check_common(ex, ctx):
    assert len(ex["ids"]) <= ctx, f"样例超长: {len(ex['ids'])} > {ctx}"
    if ex["needle_span"] is not None:
        s, e = ex["needle_span"]
        assert 0 <= s < e <= len(ex["ids"])
    qs, qe = ex["query_span"]
    assert 0 <= qs < qe <= len(ex["ids"])
    assert ex["answer"]


def test_build_all_tasks():
    tok = _tok()
    for task in TASKS:
        # fwe 的词频表本身需要较大上下文。按任务给足上下文长度，
        # 任何 ContextTooSmall 都视为失败。
        ctx = 3072 if task == "fwe" else 400
        rng = random.Random(1234 + _stable_hash(task) % 1000)
        if task == "niah_synth":
            ex = build_niah(tok, ctx, rng, mode="synth")
        elif task == "niah_corpus":
            ex = build_niah(tok, ctx, rng, mode="corpus",
                            corpus_sents=["A quiet river runs through the old town.",
                                          "Some books are to be tasted, others swallowed."])
        elif task == "niah_multikey":
            ex = build_niah(tok, ctx, rng, mode="synth", multikey=True)
        elif task == "fwe":
            ex = build_fwe(tok, ctx, rng)
        elif task == "vt":
            ex = build_vt(tok, ctx, rng, n_chains=4, depth=4)
        _check_common(ex, ctx)
        # needle 跨度解码后应包含答案；value_span 必须精确解码为答案。
        if ex["needle_span"] is not None:
            s, e = ex["needle_span"]
            seg = tok.decode(ex["ids"][s:e])
            assert ex["answer"] in seg, f"{task}: needle 段不含答案"
        if ex.get("value_span"):
            s, e = ex["value_span"]
            assert tok.decode(ex["ids"][s:e]).strip() == ex["answer"]
        print(f"  [ok] {task}: n_tokens={ex['meta']['n_tokens']}")


def test_niah_needle_span_contains_answer():
    tok = _tok()
    for i in range(5):
        rng = random.Random(500 + i)
        ex = build_niah(tok, 512, rng, mode="synth")
        s, e = ex["needle_span"]
        assert ex["answer"] in tok.decode(ex["ids"][s:e])


def test_multikey_distractor_lines_stay_complete():
    """needle 插入位置不得截断干扰行：干扰值裂成两个数字会破坏任务构造。"""
    # _TEXT 未覆盖 x/z 等字母，未登录字符会静默解码成 id 0（'\\n'），
    # 只有补齐字母表才能用解码文本做行完整性断言。
    tok = CharTokenizer(sorted(set(_TEXT) | set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    pat = re.compile(r"One of the special magic numbers for \w+ is: \d{7}\.")
    built = 0
    for i in range(60):
        rng = random.Random(i)
        try:
            ex = build_niah(tok, 512, rng, mode="synth", multikey=True)
        except ContextTooSmall:
            continue
        built += 1
        text = tok.decode(ex["ids"])
        n_tmpl = text.count("One of the special magic numbers for")
        n_full = len(pat.findall(text))
        assert n_tmpl == n_full, (
            f"seed {i}: {n_tmpl} 处干扰/needle 模板但只有 {n_full} 条完整行"
            "（行被从中截断）")
        for m in re.finditer(r"\d+", text):
            assert len(m.group(0)) == 7, f"seed {i}: 残缺数值 {m.group(0)!r}"
    assert built > 0, "测试未能构造任何 multikey 样例"


def test_score_predictions():
    # NIAH：数值命中
    sc = score_prediction("niah_synth", "7021853", " The special magic number is 7021853.")
    assert sc["em"] == 1.0
    sc = score_prediction("niah_synth", "7021853", " 1234567")
    assert sc["em"] == 0.0
    # 先答错、后附正确答案不算命中
    assert score_prediction("niah_synth", "7021853",
                            " 1234567, or maybe 7021853")["em"] == 0.0
    # vt
    assert score_prediction("vt", "48213", " The value of VAR A4 is 48213.")["em"] == 1.0
    # fwe: 召回率
    top = ["ka", "ro", "mi", "ta"]
    sc = score_prediction("fwe", "ka ro mi ta", " ka ro mi", answer_words=top)
    assert sc["recall"] == 0.75 and sc["em"] == 0.0
    assert sc["precision"] == 1.0 and abs(sc["f1"] - 6 / 7) < 1e-9
    sc = score_prediction("fwe", "ka ro mi ta", " ta mi ka ro", answer_words=top)
    assert sc["recall"] == 1.0 and sc["em"] == 1.0
    assert sc["precision"] == 1.0 and sc["f1"] == 1.0
    # fwe 严格 EM：多答无关词不算命中；precision 惩罚“灌水”
    sc = score_prediction("fwe", "ka ro mi ta", " ta mi ka ro extra",
                          answer_words=top)
    assert sc["recall"] == 1.0 and sc["em"] == 0.0
    assert sc["precision"] == 0.8 and sc["f1"] < 1.0
    # 把上下文全部词都列出来：recall 仍满，但 precision 崩掉
    sc = score_prediction("fwe", "ka ro mi ta", " ka ro mi ta zz qq",
                          answer_words=top)
    assert sc["recall"] == 1.0 and sc["precision"] < 1.0
    # 空预测：precision 记为 0（无预测不能算“零误报”）
    sc = score_prediction("fwe", "ka ro mi ta", "", answer_words=top)
    assert sc["precision"] == 0.0 and sc["recall"] == 0.0 and sc["f1"] == 0.0
    # 非 fwe 任务 precision/f1 为 None（口径不适用）
    assert score_prediction("niah_synth", "7021853", "7021853")["precision"] is None


def test_run_ruler_dedupes_clamped_ctx():
    """多个 ctx_lens 被 clamp 到同一 room_ctx 时不应重复计算/相互覆盖。"""
    import torch

    from lega.model import GPT
    from lega.ruler import run_ruler

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(vocab_size=len(tok.stoi), block_size=320, d_model=32, n_layers=1,
                n_heads=2, ffn_mult=4, variant="standard")
    model.eval()
    max_new = 12
    room = 320 - max_new - 1
    res = run_ruler(model, tok, "cpu", ctx_lens=[room, room + 100, room + 200],
                    n_samples=1, max_new=max_new, fwe_max_new=max_new,
                    tasks=("niah_synth",),
                    verbose=False)
    assert list(res["niah_synth"].keys()) == [str(room)], \
        f"clamp 到同一 room_ctx 的 ctx 档应去重: {list(res['niah_synth'].keys())}"
    cell = res["niah_synth"][str(room)]
    assert 0.0 <= cell["em"] <= 1.0
    # 每格必须附带 Wilson 95% 区间：n=1 时 EM 只能是 0/1，点估计毫无
    # 精度可言，区间把抽样不确定性显式化（端点在 [0,1] 内且覆盖点估计）
    lo, hi = cell["em_ci95_wilson"]
    assert 0.0 <= lo <= cell["em"] <= hi <= 1.0
    print(f"  [ok] run_ruler 去重 + Wilson CI（em={cell['em']}, CI=[{lo:.3f}, {hi:.3f}]）")


def test_run_ruler_collects_badcases():
    """预测错误的样本必须连问题/gold 摘录/上下文数值表一起落盘（坏案例来源）。"""
    import torch

    from lega.model import GPT
    from lega.ruler import run_ruler

    tok = _tok()
    torch.manual_seed(0)
    # 随机初始化模型几乎必然答错 → badcases 非空；且 gold 必在 prompt 中
    model = GPT(vocab_size=len(tok.stoi), block_size=320, d_model=32, n_layers=1,
                n_heads=2, ffn_mult=4, variant="standard")
    model.eval()
    res = run_ruler(model, tok, "cpu", ctx_lens=[400], n_samples=2,
                    max_new=12, fwe_max_new=12, tasks=("niah_synth", "vt"),
                    verbose=False)
    bad = res.get("_badcases", [])
    assert bad, "随机模型应有错误样本被收集"
    for c in bad:
        assert c["task"] in ("niah_synth", "vt")
        assert c["answer"] and isinstance(c["pred"], str)
        assert c["question"] and "Answer:" in c["question"]
        assert c["needle_excerpt"] and c["answer"] in c["needle_excerpt"]
        # 标注自检：gold 必须真的在 prompt 里，否则是构造错误
        assert c["answer_in_prompt"] is True
        assert c["em"] == 0.0
        assert "context_numbers" in c
        # 坏案例 depth 字段统一语义为 needle 位置比例（0~1），仅 niah 家族有；
        # vt 的链深度不能混入该字段（链深度仍在 meta 中）
        if c["task"].startswith("niah"):
            assert c["depth"] is not None and 0.0 <= c["depth"] <= 1.0
        else:
            assert c["depth"] is None
            assert c["meta"].get("depth") is not None
    # 收集开关关闭时不产生 _badcases 键
    res2 = run_ruler(model, tok, "cpu", ctx_lens=[400], n_samples=1,
                     max_new=12, fwe_max_new=12, tasks=("niah_synth",),
                     verbose=False, collect_badcases=False)
    assert "_badcases" not in res2
    print(f"  [ok] run_ruler 坏案例收集（{len(bad)} 条，含问题/gold/数值表）")


def test_vt_rejects_invalid_depth():
    """depth<1 会生成“问题问 VAR A0、正文只有 VAR A1”的自相矛盾样例。"""
    tok = _tok()
    for depth in (0, -1):
        try:
            build_vt(tok, 512, random.Random(0), n_chains=2, depth=depth)
            raised = False
        except ValueError:
            raised = True
        assert raised, depth
    print("  [ok] VT 拒绝非法 depth")


def test_vt_rejects_fractional_and_boolean_chain_counts():
    """VT 的链数/深度必须按整数使用，不能截断或把 bool 当 1。"""
    tok = _tok()
    for kwargs in ({"n_chains": 2.5}, {"n_chains": True},
                   {"depth": 4.5}, {"depth": True}):
        try:
            build_vt(tok, 512, random.Random(0), **kwargs)
            raised = False
        except ValueError:
            raised = True
        assert raised, kwargs
    # 整数值浮点仍视为合法整数，避免破坏 JSON 数字的既有用法
    ex = build_vt(tok, 512, random.Random(0), n_chains=2.0, depth=4.0)
    assert ex["meta"]["n_chains"] == 2 and ex["meta"]["depth"] == 4
    print("  [ok] VT 拒绝小数/布尔链数与深度")


def test_context_too_small():
    tok = _tok()
    try:
        build_vt(tok, 40, random.Random(0), n_chains=4, depth=4)
        raised = False
    except ContextTooSmall:
        raised = True
    assert raised, "ctx 过小时应抛出 ContextTooSmall"


def test_vt_needle_span_is_value_definition_line():
    """VT 的 gold 跨度应精确落在含答案数值的定义行，而不是整条链的包围盒。"""
    tok = _tok()
    for i in range(10):
        rng = random.Random(900 + i)
        ex = build_vt(tok, 512, rng, n_chains=4, depth=4)
        s, e = ex["needle_span"]
        seg = tok.decode(ex["ids"][s:e])
        assert seg.startswith(f"VAR {ex['meta']['chain']}1 ="), f"跨度不是定义行: {seg!r}"
        assert ex["answer"] in seg
        assert f"VAR {ex['meta']['chain']}4" not in seg


def test_corpus_sents_prefer_heldout_over_train():
    """niah_corpus 的干扰语料必须优先 held-out（val → test → train）。

    取自 train.txt 会让评测上下文与训练语料重叠，长上下文检索分数偏高、
    机制分析建立在见过的文本上；只有完全没有 held-out 文本时才允许回退，
    且来源要能被调用方审计（payload 的 corpus_source）。
    """
    d = tempfile.mkdtemp()
    try:
        def _write(name, tag, n=5):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(" ".join(
                    f"{tag}{i} sentence has plenty of characters for the filter."
                    for i in range(n)))

        _write("train.txt", "T")
        _write("val.txt", "V")
        _write("test.txt", "S")
        sents, src = load_corpus_sents_with_source(d)
        assert src == "val" and all(s.startswith("V") for s in sents), (src, sents)
        os.remove(os.path.join(d, "val.txt"))
        sents, src = load_corpus_sents_with_source(d)
        assert src == "test" and all(s.startswith("S") for s in sents), (src, sents)
        os.remove(os.path.join(d, "test.txt"))
        sents, src = load_corpus_sents_with_source(d)
        assert src == "train" and all(s.startswith("T") for s in sents), (src, sents)
        # 兼容旧接口仍返回列表
        assert load_corpus_sents(d) == sents
        # 全部缺失/为空时必须显式返回 (空, None)，而不是静默给空列表
        os.remove(os.path.join(d, "train.txt"))
        assert load_corpus_sents_with_source(d) == ([], None)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] niah_corpus 干扰语料优先 held-out（val→test→train 仅回退）")
