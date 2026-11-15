"""Evaluation & Failure Analysis: 예측과 정답을 비교해 지표와 실패 사례를 산출한다."""
from __future__ import annotations

from collections import Counter

from skills import LABELS

# 실패 유형
WRONG_SKILL = "오선택"     # 정답이 스킬 A인데 다른 스킬 B를 선택
MISSED = "누락"            # 정답이 스킬인데 NONE 선택 (필요한 스킬을 못 고름)
FALSE_TRIGGER = "과잉선택"  # 정답이 NONE인데 스킬을 선택
ERROR = "API오류"


def classify_failure(gold: str, pred: str) -> str | None:
    if pred == gold:
        return None
    if pred == "ERROR":
        return ERROR
    if gold == "NONE":
        return FALSE_TRIGGER
    if pred == "NONE":
        return MISSED
    return WRONG_SKILL


def compute_metrics(records: list[dict]) -> dict:
    """records: [{"id", "question", "gold", "pred", "reason"}, ...]"""
    n = len(records)
    correct = sum(r["gold"] == r["pred"] for r in records)
    failures = Counter(classify_failure(r["gold"], r["pred"]) for r in records)
    failures.pop(None, None)

    # 스킬별 precision / recall / F1
    per_label = {}
    for lab in LABELS:
        tp = sum(r["gold"] == lab and r["pred"] == lab for r in records)
        fp = sum(r["gold"] != lab and r["pred"] == lab for r in records)
        fn = sum(r["gold"] == lab and r["pred"] != lab for r in records)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        per_label[lab] = {"precision": p, "recall": rc, "f1": f1, "support": tp + fn}

    # 혼동 행렬 (gold → pred)
    pred_labels = LABELS + ["ERROR"]
    confusion = {g: {p: 0 for p in pred_labels} for g in LABELS}
    for r in records:
        confusion[r["gold"]][r["pred"] if r["pred"] in pred_labels else "ERROR"] += 1

    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(LABELS),
        "failure_rates": {k: failures.get(k, 0) / n for k in (WRONG_SKILL, MISSED, FALSE_TRIGGER, ERROR)},
        "failure_counts": {k: failures.get(k, 0) for k in (WRONG_SKILL, MISSED, FALSE_TRIGGER, ERROR)},
        "per_label": per_label,
        "confusion": confusion,
    }


def failure_cases(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        t = classify_failure(r["gold"], r["pred"])
        if t:
            out.append({**r, "failure_type": t})
    return out


def print_report(name: str, m: dict) -> None:
    print(f"\n=== {name} (n={m['n']}) ===")
    print(f"Accuracy: {m['accuracy']:.1%}   Macro-F1: {m['macro_f1']:.3f}")
    print("실패 유형: " + ", ".join(
        f"{k} {m['failure_counts'][k]}건({m['failure_rates'][k]:.1%})" for k in m["failure_counts"]))
    print(f"{'label':<11}{'P':>7}{'R':>7}{'F1':>7}{'n':>5}")
    for lab, v in m["per_label"].items():
        print(f"{lab:<11}{v['precision']:>7.2f}{v['recall']:>7.2f}{v['f1']:>7.2f}{v['support']:>5}")
    cols = list(next(iter(m["confusion"].values())).keys())
    print("혼동행렬 (행=정답, 열=예측)")
    print(" " * 11 + "".join(f"{c:>11}" for c in cols))
    for g, row in m["confusion"].items():
        print(f"{g:<11}" + "".join(f"{row[c]:>11}" for c in cols))
