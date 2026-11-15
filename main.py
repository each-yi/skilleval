"""SkillEval 실행 진입점.

예)
  python main.py --mock                      # API 호출 없이 파이프라인 점검
  python main.py                             # baseline vs improved, dev/test 모두 평가
  python main.py --split dev --variant improved
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from evaluate import compute_metrics, failure_cases, print_report
from router import DEFAULT_MODEL, LLMRouter, MockRouter
from skills import SKILL_SETS

ROOT = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(variant: str, split: str, mock: bool, workers: int, model: str) -> tuple[list[dict], dict]:
    data = load_jsonl(ROOT / "data" / f"{split}.jsonl")
    skills = SKILL_SETS[variant]
    router = MockRouter(skills) if mock else LLMRouter(skills, model=model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda d: router.route(d["question"]), data))

    records = [
        {"id": d["id"], "question": d["question"], "gold": d["label"], "pred": r.skill, "reason": r.reason}
        for d, r in zip(data, results)
    ]
    return records, compute_metrics(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*SKILL_SETS, "all"], default="all")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="dev=실패 분석·설명 개선용, test=최종 검증용(개선에 사용 금지)")
    ap.add_argument("--mock", action="store_true", help="LLM 대신 키워드 라우터 사용")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemini 모델 ID")
    args = ap.parse_args()

    variants = list(SKILL_SETS) if args.variant == "all" else [args.variant]
    splits = ["dev", "test"] if args.split == "all" else [args.split]

    out_dir = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split in splits:
        for variant in variants:
            records, metrics = run(variant, split, args.mock, args.workers, args.model)
            key = f"{variant}/{split}"
            summary[key] = metrics
            print_report(key, metrics)

            fails = failure_cases(records)
            if fails:
                print("실패 사례:")
                for f in fails:
                    print(f"  [{f['failure_type']}] {f['id']} 정답={f['gold']} 예측={f['pred']} | "
                          f"{f['question']}\n      └ {f['reason']}")
            with open(out_dir / f"{variant}_{split}.json", "w", encoding="utf-8") as fp:
                json.dump({"metrics": metrics, "records": records, "failures": fails},
                          fp, ensure_ascii=False, indent=2)

    # Baseline vs Improved 비교표
    if len(variants) > 1:
        print("\n=== Baseline vs Improved ===")
        print(f"{'split':<7}{'variant':<10}{'Acc':>8}{'MacroF1':>9}{'오선택':>7}{'누락':>7}{'과잉선택':>8}")
        for split in splits:
            for v in variants:
                m = summary[f"{v}/{split}"]
                r = m["failure_rates"]
                print(f"{split:<7}{v:<10}{m['accuracy']:>8.1%}{m['macro_f1']:>9.3f}"
                      f"{r['오선택']:>8.1%}{r['누락']:>8.1%}{r['과잉선택']:>9.1%}")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_dir}")


if __name__ == "__main__":
    main()
