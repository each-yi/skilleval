"""스킬 정의 (Skill Definition).

Baseline / Improved 두 버전의 스킬 설명을 정의한다.
라우터는 이 설명만 보고 스킬을 고르므로, 설명의 품질이 곧 선택 정확도를 좌우한다.
"""
from dataclasses import dataclass

# 라우터가 출력할 수 있는 레이블. NONE = 어떤 스킬에도 해당하지 않음.
LABELS = ["PAPER", "EXPERIMENT", "NONE"]


@dataclass(frozen=True)
class Skill:
    name: str          # 레이블 (PAPER / EXPERIMENT)
    title: str         # 사람이 읽는 이름
    description: str   # 사용 조건 (라우터에 그대로 전달됨)


SKILL_SETS: dict[str, list[Skill]] = {
    # 모호하고 넓은 설명 → 과잉 선택 / 오선택이 발생하기 쉬움
    "baseline": [
        Skill("PAPER", "Paper Skill", "연구 관련 질문을 처리한다."),
        Skill("EXPERIMENT", "Experiment Skill", "AI 관련 질문을 분석한다."),
    ],
    # 구체적인 사용 조건 + 비사용 조건(경계)을 명시
    "improved": [
        Skill(
            "PAPER",
            "Paper Skill",
            "특정 논문(또는 논문 초록·본문)을 대상으로 방법론, 핵심 기여, 한계점, "
            "관련 연구와의 차이를 분석·요약한다. "
            "사용하지 않는 경우: 논문 없이 일반 개념만 묻는 질문, 연구 행정·윤리 절차 질문, "
            "수치 기반의 모델 성능 비교.",
        ),
        Skill(
            "EXPERIMENT",
            "Experiment Skill",
            "실험 결과 수치(정확도, F1-score, precision/recall, loss, AUC 등)를 바탕으로 "
            "모델·설정 간 성능을 비교하거나, 실험 결과를 해석하고 실험 설계(베이스라인, "
            "하이퍼파라미터, 데이터 분할)를 점검한다. "
            "사용하지 않는 경우: AI 개념 설명, 일반 코딩 질문, 논문의 방법론·기여 분석.",
        ),
    ],
}


def render_skills(skills: list[Skill]) -> str:
    """라우터 프롬프트에 넣을 스킬 목록 텍스트."""
    lines = [f"- {s.name} ({s.title}): {s.description}" for s in skills]
    lines.append("- NONE: 위 스킬 중 어느 것의 사용 조건에도 맞지 않는 요청.")
    return "\n".join(lines)
