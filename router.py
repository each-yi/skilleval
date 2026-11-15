"""Skill Router: Gemini Flash API로 사용자 질문에 맞는 스킬을 선택한다."""
from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from skills import Skill, render_skills

# 환경변수 GEMINI_MODEL 또는 --model 옵션으로 변경 가능
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SYSTEM_TEMPLATE = """당신은 AI 에이전트의 스킬 라우터입니다.
사용자 요청을 읽고, 아래 스킬 목록의 설명(사용 조건)에 가장 잘 맞는 스킬 하나를 고르세요.
어떤 스킬의 사용 조건에도 맞지 않으면 NONE을 고르세요.
스킬 설명에 적힌 내용만 근거로 판단하세요.

[스킬 목록]
{skills}"""


class SkillLabel(str, enum.Enum):
    PAPER = "PAPER"
    EXPERIMENT = "EXPERIMENT"
    NONE = "NONE"


class RoutingDecision(BaseModel):
    skill: SkillLabel
    reason: str = Field(description="선택 근거를 한 문장으로")


@dataclass
class RouteResult:
    skill: str      # PAPER / EXPERIMENT / NONE / ERROR
    reason: str


class LLMRouter:
    def __init__(self, skills: list[Skill], model: str = DEFAULT_MODEL):
        self.client = genai.Client()  # GEMINI_API_KEY (또는 GOOGLE_API_KEY) 환경변수 사용
        self.model = model
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_TEMPLATE.format(skills=render_skills(skills)),
            response_mime_type="application/json",
            response_schema=RoutingDecision,     # JSON 구조화 출력
            thinking_config=types.ThinkingConfig(thinking_level="low"),  # 단순 분류라 low
        )

    def route(self, question: str) -> RouteResult:
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=question, config=self.config
            )
        except errors.ClientError as e:   # 4xx: 잘못된 요청, 키 오류, 429 rate limit 등
            return RouteResult("ERROR", f"client {e.code}: {e.message}")
        except errors.ServerError as e:   # 5xx
            return RouteResult("ERROR", f"server {e.code}: {e.message}")

        d = response.parsed
        if not isinstance(d, RoutingDecision):
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            return RouteResult("ERROR", f"no parsed output (finish_reason={reason})")
        return RouteResult(d.skill.value, d.reason)


class MockRouter:
    """API 키 없이 파이프라인을 점검하기 위한 키워드 기반 라우터 (--mock)."""

    def __init__(self, skills: list[Skill], model: str | None = None):
        self.skills = skills

    def route(self, question: str) -> RouteResult:
        if re.search(r"F1|정확도|accuracy|성능|loss|AUC|recall|precision|하이퍼파라미터", question, re.I):
            return RouteResult("EXPERIMENT", "mock: 성능 지표 키워드")
        if re.search(r"논문|기여|한계|방법론", question):
            return RouteResult("PAPER", "mock: 논문 키워드")
        return RouteResult("NONE", "mock: 매칭 키워드 없음")
