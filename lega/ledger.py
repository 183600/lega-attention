"""Tamper-evident append-only JSONL ledgers (SHA-256 hash chain).

The experiment ledger and the test-set access ledger are scientific evidence:
they are used to decide whether a run is confirmatory, how often a sealed test
set was consumed, and what exactly was executed.  A plain append-only file can
silently lose or reorder records (accidental truncation, an editor, a cleanup
script, or a hand-edit); lifecycle checks alone cannot tell that an old record
was removed from the middle of the file.

Every record written through this module gets two fields:

``prev_record_sha256``
    digest of the immediately preceding *valid* record (``GENESIS`` for the
    first record), i.e. a hash chain link;
``record_sha256``
    digest of this record with ``record_sha256`` itself removed.

Any content edit, reordering, or middle-of-file deletion breaks the chain and
is reported by :func:`verify_chain`.  The chain is tamper-evident, not
tamper-proof: an actor who rewrites the whole file (including every digest)
can forge a new consistent chain, and truncating the tail leaves a valid
prefix.  That limitation is reported explicitly (``anchored=False``), so
callers do not over-claim integrity.  Legacy records written before the chain
existed are counted as ``legacy_records`` instead of being silently treated as
verified.
"""

import hashlib
import json
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

CHAIN_ALGO = "sha256-chain-v1"
GENESIS_SHA256 = "0" * 64
PREV_FIELD = "prev_record_sha256"
SELF_FIELD = "record_sha256"


def record_sha256(record: dict) -> str:
    """Digest of a record, excluding its own ``record_sha256`` field.

    Canonical JSON (sorted keys, compact separators) makes the digest
    independent of dict insertion order and of the on-disk line formatting.
    """
    if not isinstance(record, dict):
        raise TypeError("ledger record must be a dict")
    payload = {key: value for key, value in record.items() if key != SELF_FIELD}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def chain_record(record: dict, previous: dict = None) -> dict:
    """Return a copy of ``record`` linked to ``previous`` (or genesis)."""
    out = dict(record)
    out[PREV_FIELD] = (
        record_sha256(previous) if isinstance(previous, dict) else GENESIS_SHA256
    )
    out[SELF_FIELD] = record_sha256(out)
    return out


def verify_chain(records) -> dict:
    """Verify a sequence of ledger records.

    ``records`` must be the valid object records in file order (malformed
    lines are skipped by the readers and therefore are not part of the chain).
    Legacy records without chain fields are tolerated but counted; once a
    chained record appears, every later record must be chained too.
    """
    valid = (
        [r for r in records if isinstance(r, dict)]
        if isinstance(records, (list, tuple))
        else []
    )
    issues, legacy, chained = [], 0, 0
    expected_prev = GENESIS_SHA256
    chain_started = False
    for index, record in enumerate(valid):
        has_prev = PREV_FIELD in record
        has_self = SELF_FIELD in record
        if not has_prev and not has_self:
            if chain_started:
                issues.append({"index": index, "issue": "missing_chain_fields"})
            else:
                legacy += 1
            expected_prev = record_sha256(record)
            continue
        if not (has_prev and has_self):
            issues.append({"index": index, "issue": "partial_chain_fields"})
            chain_started = True
            expected_prev = record_sha256(record)
            continue
        chain_started = True
        chained += 1
        actual_prev = record.get(PREV_FIELD)
        if actual_prev != expected_prev:
            issues.append(
                {
                    "index": index,
                    "issue": "prev_mismatch",
                    "expected": expected_prev,
                    "actual": actual_prev if isinstance(actual_prev, str) else None,
                }
            )
        computed = record_sha256(record)
        actual_self = record.get(SELF_FIELD)
        if actual_self != computed:
            issues.append(
                {
                    "index": index,
                    "issue": "self_mismatch",
                    "expected": computed,
                    "actual": actual_self if isinstance(actual_self, str) else None,
                }
            )
        # Use the computed digest for the next link: editing only the stored
        # self field is still caught by self_mismatch, while editing content
        # breaks both self_mismatch and the next record's prev link.
        expected_prev = computed
    return {
        "algo": CHAIN_ALGO,
        "records": len(valid),
        "chained_records": chained,
        "legacy_records": legacy,
        "head_sha256": expected_prev if valid else None,
        "fully_chained": bool(valid) and legacy == 0 and not issues,
        # A hash chain cannot prove that the tail was not truncated: an
        # attacker can cut the file after any valid record.  Callers must not
        # present ``ok`` as "complete history" without an external anchor.
        "anchored": False,
        "ok": not issues,
        "issues": issues,
    }
