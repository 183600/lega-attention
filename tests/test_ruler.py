"""RULER 任务构造器测试：跨度有效性、答案一致性、打分函数。"""
import os
import random
import re
import shutil
import tempfile

from lega.data import CACHE_SCHEMA, CharTokenizer
from lega.ruler import (DEPTH_BINS, EVIDENCE_TASKS, NEEDLE_TMPL,
                        build_evidence_condition, build_fwe, build_niah,
                        build_vt, depth_bin, load_corpus_sents,
                        load_corpus_sents_with_source, record_corpus_test_access,
                        score_prediction,
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


def test_char_tokenizer_missing_digits_skips_numeric_tasks():
    """字符词表没有数字时可跳过，不能让 gold 跨度定位抛 AssertionError。"""
    letters = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .\n"))
    tok = CharTokenizer(letters, add_unk=True)
    for fn in (lambda: build_niah(tok, 512, random.Random(1), mode="synth"),
               lambda: build_vt(tok, 512, random.Random(1))):
        try:
            fn()
        except ContextTooSmall:
            pass
        else:
            raise AssertionError("缺少数字的字符词表应可跳过数值任务")
    print("  [ok] 字符词表缺数字：niah/vt 以 ContextTooSmall 跳过")


def test_fwe_gold_uses_planned_words_not_unk_decoded_text():
    """字符词表 OOV 不能把 decode 出来的 unk/残词写成 fwe gold。"""
    chars = sorted(set("abcdefghijklmnoprstuvwyABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       " .\n?:0123456789"))
    tok = CharTokenizer(chars, add_unk=True)
    ex = build_fwe(tok, 3072, random.Random(0))
    assert "unk" not in ex["answer"].split()
    assert "unk" not in ex["answer_words"]
    assert ex["answer_words"] and len(ex["answer_words"]) == 10
    body = tok.decode(ex["ids"][:ex["query_span"][0]])
    for word in ex["answer_words"]:
        assert word in body, (word, body[:200])
    print("  [ok] fwe gold 来自计划写入的词，不受 <unk> 解码污染")


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


def test_multikey_never_silently_degrades_to_single_key():
    """multikey 至少要有干扰行；放不下时跳过，不能改名不换任务。"""
    tok = CharTokenizer(sorted(set(_TEXT) | set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")))
    pat = re.compile(r"One of the special magic numbers for \w+ is: \d{7}\.")
    built = 0
    for seed in range(100):
        try:
            ex = build_niah(tok, 280, random.Random(seed),
                            mode="synth", multikey=True)
        except ContextTooSmall:
            continue
        built += 1
        assert len(pat.findall(tok.decode(ex["ids"]))) >= 2, (
            f"seed={seed}: niah_multikey 没有干扰行")
    assert built > 0
    # room 连一条干扰行都放不下时必须跳过，而不是构造单键样例。
    for seed in range(20):
        try:
            build_niah(tok, 250, random.Random(seed),
                       mode="synth", multikey=True)
        except ContextTooSmall:
            pass
        else:
            raise AssertionError(f"seed={seed}: 无干扰行仍构造 multikey")
    print("  [ok] multikey 保证至少一条干扰行，否则 ContextTooSmall")


def test_score_predictions():
    # NIAH：数值命中
    sc = score_prediction("niah_synth", "7021853", " The special magic number is 7021853.")
    assert sc["em"] == 1.0
    sc = score_prediction("niah_synth", "7021853", " 1234567")
    assert sc["em"] == 0.0
    # 先答错、后附正确答案不算命中
    assert score_prediction("niah_synth", "7021853",
                            " 1234567, or maybe 7021853")["em"] == 0.0
    # 只截取数字子串会把负号/小数/字母后缀误当成正确答案
    for bad in (" -7021853.", " 7021853.9", " 7021853abc", "x7021853",
                " 7021853e2",
                # 表达式/比值中的前缀数字不是独立答案，不能判命中
                " 7021853/2", " 7021853 + 1", " 7021853-1",
                " 7021853:123", " 7021853*2", " 7021853%2"):
        assert score_prediction("niah_synth", "7021853", bad)["em"] == 0.0, bad
    # 右操作数同样不能成为“第一个数值回答”：否则 7021853/2 会在 gold=2
    # 时把表达式里的 2 误判为命中（表达式不是独立答案）。
    for bad, gold in ((" 7021853/2", "2"), (" 7021853 / 2", "2"),
                      (" 3*4", "4"), (" 10%3", "3"),
                      (" 123:456", "456"), (" 7021853 - 2", "2")):
        assert score_prediction("niah_synth", gold, bad)["em"] == 0.0, (bad, gold)
    # 标签后的数值仍是独立回答（冒号前是普通文本，不是范围/表达式）
    for ok in (" answer: 2", " answer:2", "The answer is: 2."):
        assert score_prediction("niah_synth", "2", ok)["em"] == 1.0, ok
    # 标点边界仍算命中
    for ok in ("7021853.", " 7021853,", "(7021853)"):
        assert score_prediction("niah_synth", "7021853", ok)["em"] == 1.0, ok
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


def test_run_ruler_rejects_fractional_and_bool_params():
    """run_ruler 入口参数必须是精确整数：截断会悄悄改写评测协议。

    n_samples=1.5 → 1、ctx_lens=[True] → 1（样例全过短被跳过、结果为空却
    看似成功）、vt_chains=4.5 → 4 会绕过 build_vt 的链数守卫。整数值浮点
    （JSON 数字的既有用法）仍视为合法整数。
    """
    import torch

    from lega.model import GPT
    from lega.ruler import run_ruler

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(vocab_size=len(tok.stoi), block_size=320, d_model=32, n_layers=1,
                n_heads=2, ffn_mult=4, variant="standard")
    model.eval()
    base = {"ctx_lens": [256], "n_samples": 1, "max_new": 12,
            "fwe_max_new": 12, "tasks": ("niah_synth",), "verbose": False}
    for key, bad in (("ctx_lens", [256.5]), ("ctx_lens", [True]),
                     ("ctx_lens", "256"), ("ctx_lens", b"256"),
                     ("n_samples", 2.5), ("n_samples", True),
                     ("seed", 1.5), ("seed", True),
                     ("max_new", 6.5), ("fwe_max_new", 9.5),
                     ("vt_chains", 4.5), ("vt_chains", True),
                     ("vt_depth", 3.5), ("gen_batch", 2.5),
                     ("n_samples", 0), ("ctx_lens", [0]),
                     ("ctx_lens", [-1]), ("ctx_lens", []),
                     ("max_new", 0), ("fwe_max_new", -1),
                     ("vt_chains", 0), ("vt_depth", 0),
                     ("gen_batch", 0), ("max_new", 319)):
        kwargs = dict(base, **{key: bad})
        try:
            run_ruler(model, tok, "cpu", **kwargs)
            raised = False
        except ValueError:
            raised = True
        assert raised, (key, bad)
    res = run_ruler(model, tok, "cpu", ctx_lens=[256.0], n_samples=1.0,
                    seed=7.0, max_new=12, fwe_max_new=12,
                    tasks=("niah_synth",), verbose=False)
    assert res["niah_synth"]
    print("  [ok] run_ruler 拒绝小数/布尔/越界参数，整数值浮点仍可用")


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


def test_vt_rejects_non_builtin_fractional_numbers():
    """np.float32/Decimal/Fraction 等非内建数值也不能被 int() 静默截断。"""
    import numpy as np
    from fractions import Fraction

    tok = _tok()
    for bad in (np.float32(4.5), np.bool_(True), Fraction(9, 2)):
        try:
            build_vt(tok, 512, random.Random(0), n_chains=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, (type(bad), bad)
    print("  [ok] VT 拒绝非内建小数/布尔数值的截断")


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


def test_corpus_sents_seal_test_and_prefer_heldout():
    """niah_corpus 的干扰语料必须优先 held-out，且 test 默认尘封。

    取自 train.txt 会让评测上下文与训练语料重叠，长上下文检索分数偏高、
    机制分析建立在见过的文本上；只有完全没有 held-out 文本时才允许回退，
    且来源要能被调用方审计（payload 的 corpus_source）。
    另外 test 是尘封 split：本加载器直读裸 .txt，绕过 load_bin 闸门，默认
    绝不能把 test 当候选（否则 val 缺失时无账本地读测试集）。
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
        # 默认：val 优先，test 不进候选
        sents, src = load_corpus_sents_with_source(d)
        assert src == "val" and all(s.startswith("V") for s in sents), (src, sents)
        # val 缺失时 test 默认被尘封，回退 train（而不是静默读 test）
        os.remove(os.path.join(d, "val.txt"))
        sents, src = load_corpus_sents_with_source(d)
        assert src == "train" and all(s.startswith("T") for s in sents), (src, sents)
        # 非布尔真值不能放行 test
        for bad in (1, "false", "1"):
            sents, src = load_corpus_sents_with_source(d, allow_test=bad)
            assert src == "train" and all(s.startswith("T") for s in sents), \
                (bad, src, sents)
        # split 别名/绝对路径不能绕过 test 尘封（"./test" 会解析到 test.txt）
        for alias in ("./test", os.path.join(d, "test"), "../test"):
            try:
                load_corpus_sents_with_source(d, splits=(alias,))
                raise AssertionError(f"splits={alias!r} 绕过了 test 尘封闸门")
            except ValueError:
                pass
        # 显式放行才允许 test
        sents, src = load_corpus_sents_with_source(d, allow_test=True)
        assert src == "test" and all(s.startswith("S") for s in sents), (src, sents)
        # 兼容旧接口仍返回列表（同样不放行 test，此时 val 已删 → 回退 train）
        compat = load_corpus_sents(d)
        assert all(s.startswith("T") for s in compat) and compat, compat
        # 全部缺失/为空时必须显式返回 (空, None)，而不是静默给空列表
        os.remove(os.path.join(d, "train.txt"))
        os.remove(os.path.join(d, "test.txt"))
        assert load_corpus_sents_with_source(d) == ([], None)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] niah_corpus 干扰语料优先 held-out、test 默认尘封（显式 allow_test 才放行）")


def test_corpus_test_access_recorded():
    """显式放行 test 语料时，必须写入 test 访问账本（账本旁路留痕）。"""
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(
                f"S{i} sentence has plenty of characters for the filter."
                for i in range(5)))
        ok = record_corpus_test_access(d, "test", stage="ruler")
        assert ok is True
        from lega.datacheck import read_test_access_log
        from lega.repro import sha256_file
        records = read_test_access_log(d)
        assert len(records) == 1
        assert records[0]["reason"] == "ruler_niah_corpus_distractors"
        assert records[0]["corpus_source_sha256"] == \
            sha256_file(os.path.join(d, "test.txt"))
        # 非 test 来源不写账本
        assert record_corpus_test_access(d, "val") is False
        assert len(read_test_access_log(d)) == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] test 语料放行写入访问账本，val 来源不写")


def test_run_ruler_records_per_sample_outcomes():
    """每格必须落盘逐样例 EM（sample_id 跨变体稳定），供样例级配对检验。"""
    import torch

    from lega.model import GPT
    from lega.ruler import run_ruler

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(vocab_size=len(tok.stoi), block_size=320, d_model=32, n_layers=1,
                n_heads=2, ffn_mult=4, variant="standard")
    model.eval()
    res = run_ruler(model, tok, "cpu", ctx_lens=[320], n_samples=3,
                    max_new=12, fwe_max_new=12, tasks=("niah_synth",),
                    verbose=False)
    cell = next(iter(res["niah_synth"].values()))
    samples = cell["samples"]
    assert len(samples) == cell["n"] == 3
    ids = [s["sample_id"] for s in samples]
    assert len(set(ids)) == len(ids), ids
    assert all(isinstance(s["sample_id"], str) and s["sample_id"] for s in samples)
    assert all(s["em"] in (0.0, 1.0) for s in samples)
    assert sum(s["em"] for s in samples) / len(samples) == cell["em"]
    # NIAH 逐样例带 needle 位置与分箱；分层汇总必须与样本逐条一致。
    assert all(s["depth_bin"] in DEPTH_BINS for s in samples), samples
    assert all(isinstance(s["depth"], float) and 0.0 <= s["depth"] <= 1.0
               for s in samples), samples
    assert set(cell["em_by_depth"]) == {s["depth_bin"] for s in samples}
    for b, rec in cell["em_by_depth"].items():
        rows = [s for s in samples if s["depth_bin"] == b]
        assert rec["n"] == len(rows)
        assert abs(rec["em"] - sum(s["em"] for s in rows) / len(rows)) < 1e-12
        lo, hi = rec["em_ci95_wilson"]
        assert 0.0 <= lo <= rec["em"] <= hi <= 1.0
    print("  [ok] run_ruler 逐样例 EM 落盘（sample_id 唯一、与聚合 EM 一致）")


def test_depth_bin_boundaries_and_invalid_values():
    """位置分箱必须半开且覆盖端点：1/3→middle、2/3→late、1.0→late。"""
    assert depth_bin(0.0) == "early"
    assert depth_bin(0.3333) == "early"
    assert depth_bin(1.0 / 3.0) == "middle"
    assert depth_bin(0.5) == "middle"
    assert depth_bin(2.0 / 3.0) == "late"
    assert depth_bin(1.0) == "late"
    for bad in (None, True, False, -0.01, 1.01, float("nan"),
                float("inf"), "0.5", object()):
        assert depth_bin(bad) is None, bad
    print("  [ok] depth_bin（半开边界、末期含 1.0、非法值返回 None）")


def test_ruler_payload_binds_checkpoint_hash(monkeypatch):
    """RULER 结果必须绑定被评测 checkpoint 的路径与内容指纹。

    只写目录名/seed 不够：同一路径的权重被覆盖后，旧 RULER 分数仍会被当作
    当前模型结果引用。这里跑一次最小 CLI（任务因 ctx 过小被跳过，但仍落盘
    payload），核对 ckpt_sha256 与文件内容一致。
    """
    import json
    import sys

    import numpy as np
    import torch

    from lega.model import GPT, save_checkpoint
    from lega.repro import sha256_file

    d = tempfile.mkdtemp()
    try:
        data = os.path.join(d, "data")
        os.makedirs(data)
        text = "The quick brown fox jumps over the lazy dog. 0123456789"
        itos = sorted(set(text))
        with open(os.path.join(data, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "itos": itos, "char_unk": False,
                       "vocab_size": len(itos), "cache_schema": CACHE_SCHEMA,
                       "fit_on": "train"}, f)
        arr = np.array([itos.index(c) for c in text], dtype=np.uint16)
        arr.tofile(os.path.join(data, "train.bin"))
        arr.tofile(os.path.join(data, "val.bin"))
        ckpt = os.path.join(d, "model.pt")
        torch.manual_seed(0)
        model = GPT(vocab_size=len(itos), block_size=128, d_model=8,
                    n_layers=1, n_heads=1, ffn_mult=2, variant="lega",
                    dropout=0.0, attn_dropout=0.0, gate_bias_init=2.0)
        save_checkpoint(ckpt, model, extra={"variant": "lega", "seed": 5})
        out = os.path.join(d, "ruler.json")
        monkeypatch.setattr(sys, "argv", [
            "ruler", "--ckpt", ckpt, "--data_dir", data, "--out", out,
            "--ctx_lens", "64", "--n_samples", "1", "--device", "cpu"])
        from lega.ruler import main as ruler_main
        ruler_main()
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["ckpt"] == os.path.abspath(ckpt)
        assert payload["ckpt_sha256"] == sha256_file(ckpt)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] RULER payload 绑定 checkpoint 路径与内容哈希")


def test_evidence_absent_wrong_conflict_builders_and_scoring():
    """无证据/错误证据/冲突证据：答案固定 NONE，且只在 abstain 时给 EM。"""
    tok = _tok()
    for i, task in enumerate(EVIDENCE_TASKS):
        ex = build_evidence_condition(tok, 512, random.Random(100 + i), task)
        assert ex["task"] == task and ex["answer"] == "NONE"
        assert len(ex["ids"]) <= 512
        qs, qe = ex["query_span"]
        assert 0 <= qs < qe <= len(ex["ids"])
        inserted = ex["meta"]["inserted"]
        if task == "evidence_absent":
            assert inserted == []
        elif task == "evidence_wrong":
            assert len(inserted) == 1
            assert inserted[0]["key"] != ex["meta"]["target_key"]
        else:
            assert len(inserted) == 2
            assert inserted[0]["key"] == inserted[1]["key"]
            assert inserted[0]["value"] != inserted[1]["value"]
    assert score_prediction("evidence_absent", "NONE", " NONE")["em"] == 1.0
    assert score_prediction("evidence_absent", "NONE", " none.")["em"] == 1.0
    assert score_prediction("evidence_absent", "NONE", " 7021853")["em"] == 0.0
    assert score_prediction("evidence_conflict", "NONE", " NONE 123")["em"] == 1.0


def test_evidence_wrong_conflict_do_not_degrade_to_absent():
    """room 放不下必需证据时必须跳过，不能构造出没有证据的 wrong/conflict。"""
    tok = _tok()
    for condition in ("evidence_wrong", "evidence_conflict"):
        for seed in range(20):
            try:
                build_evidence_condition(
                    tok, 120, random.Random(seed), condition)
            except ContextTooSmall:
                pass
            else:
                raise AssertionError(
                    f"{condition} seed={seed}: 证据放不下却仍构造了样例")
    print("  [ok] wrong/conflict 证据放不下时按 ContextTooSmall 跳过")


def test_evidence_conflict_keeps_both_evidence_lines_intact():
    """冲突条件的两条证据都必须完整落在问题之前，不能被 room 截断。"""
    tok = _tok()
    for seed in range(200):
        ex = build_evidence_condition(
            tok, 512, random.Random(seed), "evidence_conflict")
        qs, _ = ex["query_span"]
        assert len(ex["meta"]["inserted"]) == 2
        for rec in ex["meta"]["inserted"]:
            line = tok.encode(
                "\n" + NEEDLE_TMPL.format(key=rec["key"], value=rec["value"]))
            at = rec["pos"]
            assert ex["ids"][at:at + len(line)] == line, (
                f"seed={seed}: 冲突证据在 pos={at} 被截断")
            assert at + len(line) <= qs, (
                f"seed={seed}: 冲突证据越过问题起点")
