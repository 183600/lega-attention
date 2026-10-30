"""数据卫生与防泄漏检查（拆分模块，对外接口保持不变）。"""
"""数据卫生与防泄漏检查：划分协议、清洗、去污染、漂移筛查与测试集访问账本。

实验协议（本模块是实现而不是口号）:
  * 先划分 train/val/test，再做任何预处理；字符级词表只在 train 上拟合，
    val/test 只做 transform，未见字符统一映射到 <unk>（见 data.py）；
  * 连续文本（--local_txt）按时间先后切分：最早 80% 训练、其后 10% 验证、
    最新 10% 测试；切点对齐文档/段落/行边界，同一文档绝不跨 split；
  * 重复样本：与 val/test 精确重复的长行从 train 移除（去污染），val/test
    保持尘封不动；另用采样与全量精确 64-gram 扫描筛查跨 split 的 token 级
    重叠（全量结果逐字节复核，见 data_report.contamination_exact）；
  * token 流硬校验：整数 dtype、有限值、id ∈ [0, vocab)、uint16 可容纳；
  * 分布漂移：train 与 val/test 的一元 token 分布 TV/JS 散度与 OOV
    （默认全量精确计数，显式 max_samples 时才采样）；
  * 文本清洗：U+FFFD 替换字符、NUL 与非法控制符（标签噪声来源）；
  * 测试集访问账本：每次 test 评测都追加记录到 data_dir/.test_access_log.jsonl，
    使“反复在测试集上评测/调参”的行为可审计。

所有函数只返回 JSON 原生类型（int/float/str/bool/None），可直接写入报告。
"""
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import numpy as np
from ..utils import strict_int

try:
    import fcntl
except ImportError:  # Windows 等无 flock 平台：退化为原行为（单进程安全）
    fcntl = None

UNK_TOKEN = "<unk>"

TEST_ACCESS_LOG = ".test_access_log.jsonl"

# 非法控制符（保留 \t \n \r）；U+FFFD 替换为空格（长度不变，不跨词粘连）
_SANITIZE_TABLE = {i: None for i in range(32) if i not in (9, 10, 13)}

_SANITIZE_TABLE[0x7f] = None

_SANITIZE_TABLE[ord("\ufffd")] = " "

# 行级标签噪声（语言建模标签即下一 token，来自同一份清洗前文本）：
# 含 U+FFFD/非法控制符的行比例反映“标签质量”。
# 只匹配坏字符本身：单字符类扫描是线性的；若用 `[^\n]*坏字符[^\n]*` 按行
# 匹配，正则引擎在无坏字符的行上会从每个起点回溯整行（O(行长²/行)），
# 长行大语料下不可取。行号由坏字符位置之间的换行增量换算
# （见 sanitize_text），整体单趟线性。
_BAD_CHAR_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# WikiText 风格的一级/多级标题行：作为“文档边界”的首选切点。
# 兼容 `= Title =`、`== Section ==`、`=== Sub ===` 及 WikiText-103 原始
# 语料的分隔写法 `= = Section = =`、`= = = Sub = = =`。
_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:=[ \t]*)+[^=\s\r\n][^\r\n]*?(?:[ \t]*=)+[ \t]*\r?$")

# ---------------------------------------------------------------- 近似重复
#
# 精确行/块去重只能抓住逐字节（或仅空白/NFKC 差异）的复制。真实语料里更常见
# 的是“改了几个词/删了一段”的近似复制：它同样会让 train 记住 val/test 的
# 内容，也会在 train 内部放大同一段文本的权重。这里用字符 5-gram 的
# MinHash + LSH 做近似重复筛查，再对候选对精确计算 Jaccard/containment。
#
# 口径：
#   * 只从 train 删除；val/test 逐字节不动（与精确去污染一致）；
#   * 跨 split 用 max(Jaccard, containment)：val 的一小段被整段抄进更长的
#     train 文档时，Jaccard 会被长度稀释，containment 才能发现；
#   * train 内部只用 Jaccard，避免“短段落被长文档包含”导致过度删除；
#   * 阈值是协议参数而不是从数据估计出来的；报告同时给出 0.70/阈值/0.90
#     三档敏感性，读者能看出删除量对阈值的依赖。
NEAR_DUP_SHINGLE = 5

NEAR_DUP_THRESHOLD = 0.80

NEAR_DUP_MIN_CHARS = 200

NEAR_DUP_MIN_SHINGLES = 20

NEAR_DUP_BANDS = 6

NEAR_DUP_ROWS = 3

NEAR_DUP_HASHES = NEAR_DUP_BANDS * NEAR_DUP_ROWS

NEAR_DUP_MAX_BLOCK_CHARS = 2_000_000

NEAR_DUP_MAX_GROUP = 64

NEAR_DUP_MAX_CANDIDATES = 1_000_000

NEAR_DUP_CACHE_BLOCKS = 10_000

# 近似重复扫描的“可审计覆盖率”下限：低于该比例时，报告不能暗示全语料都已
# 做过近重复筛查。短块（标题/固定用语）与超大块被跳过是协议选择，但跳过了
# 多少字符必须显式可见；覆盖率不足只告警、不自动改阈值，避免按结果挑参数。
NEAR_DUP_MIN_COVERAGE = 0.80

# 包含关系（containment）候选的确定性锚点数。MinHash/LSH 估计的是
# Jaccard：短 val 段被整段抄进长 train 文档时 Jaccard 很低，band 不会碰撞，
# 但 containment=1。这里从每个块的 shingle 集合中均匀取少量真实 shingle
# 作为锚点；扫描另一侧时统计命中的锚点数，达到阈值比例即进入候选，再由
# 精确 Jaccard/containment 裁决。候选阶段只需高召回，精确阶段保证不误删。
NEAR_DUP_ANCHORS = 8

_U64 = np.uint64

_U64_GOLDEN = _U64(0x9E3779B97F4A7C15)

_U64_MIX1 = _U64(0xBF58476D1CE4E5B9)

_U64_MIX2 = _U64(0x94D049BB133111EB)

_U64_BASE = _U64(1099511628211)

# 短 n-gram 筛查的默认长度：64-gram（ngram_overlap）是“必须拦截”的强证据，
# 但短于 64 token 的复制（整句/段落级复用，尤其模板化文本）会完全漏检。
# 短 n 更易被自然语言常见搭配命中，因此只做**告警**、不参与硬闸门。
SHORT_NGRAM_N = 32

_NEWLINE_RE = re.compile("\n")

# 空行：两个换行之间只允许空白（兼容 CRLF 与“有空格的行”）。
_BLANK_LINE_RE = re.compile(r"(?<=\n)[ \t\r]*\n")

# 时序划分的“实际比例偏离目标”告警线（绝对百分点）。切点对齐文档/段落
# 边界可能把 80/10/10 拉成 70/12/18；这不是错误，但必须显式报告，否则
# “train 占 80%”只是目标值，读者会误以为实际语料也按此规模训练。
SPLIT_FRACTION_TOLERANCE = 0.05

# 自检告警阈值：不是统计检验的显著性水平，而是“必须人工复核”的操作触发线，
# 与开放集/结构不同分布下的经验值一致（可通过 extra_report 覆盖记录）。
ALERT_THRESHOLDS = {
    "ngram_overlap_rate": 0.0,       # 任何跨 split 长 n-gram 复制都不允许
    "distribution_tv": 0.1,          # 一元 TV 距离 > 0.1 视为明显漂移
    "distribution_js_bits": 0.1,     # JS 散度（bit）
    "eval_oov_token_rate": 0.02,     # eval 中 train 未见 token 质量占比
    "eval_unk_char_rate": 0.005,     # val/test 字符级 OOV 率
    "bad_char_line_rate": 0.01,      # 含坏字符的行占比（标签噪声）
}

# 会让 train/val/test 独立性失效的协议违规：run_all 在训练前据此硬失败
# （可显式覆盖），避免“报告里有警告但全量训练照跑”。
# 划分协议违规（群组跨 split / 字符切分 / 打乱）之外，采样/全量精确 64-gram
# 跨 split 重叠率 >0 是“语料在 split 间被复制”的直接证据——与分布漂移（temporal split
# 下属正常现象）不同，它必然导致评测读到训练见过的长串，必须训练前拦截。
CRITICAL_SPLIT_ALERT_CODES = (
    "split_shuffled", "split_char_cut", "split_documents_cross_cut",
    "split_paragraphs_cross_cut",
    "ngram_overlap_train_val", "ngram_overlap_train_test",
    "ngram_overlap_val_test",
    # 全量 train 扫描 + 逐字节复核的精确版本：采样筛查可能漏掉占比极低的
    # 复制段，精确命中没有“没抽到”的借口，同样必须在训练前拦截。
    "ngram_overlap_exact_train_val", "ngram_overlap_exact_train_test",
    "ngram_overlap_exact_val_test",
    "eval_splits_overlap", "eval_splits_overlap_documents",
    # 清洗后复核发现 train 仍与 val/test 共享精确/归一化长行或多行块：
    # 这是“去污染本应删除却仍残留”的直接证据，必须像 n-gram 重叠一样
    # 在训练前拦截，而不是只在汇总页提示。
    "post_cleanup_residual_overlap",
)


def _install(namespace):
    """把共享命名空间注入拆分模块。"""
    for key, value in globals().items():
        if key != "_install" and not key.startswith("__"):
            namespace.setdefault(key, value)

