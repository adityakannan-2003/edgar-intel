"""Configuration, loaded from environment with sane defaults.

Every knob that affects retrieval or answer quality is captured here and
serialised into `eval_runs.config`, so a run recorded six months ago can be
explained: which strategy, which k, which models, which thresholds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EDGAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----------------------------------------------------------------- infra
    db_dsn: str = "postgresql://edgar:edgar@localhost:5433/edgar"

    # ----------------------------------------------------------------- edgar
    user_agent: str = "edgar-intel example@example.com"
    edgar_base: str = "https://data.sec.gov"
    edgar_www: str = "https://www.sec.gov"
    # The SEC's published ceiling is 10 requests/second. Staying under it is a
    # condition of access, not a nicety.
    edgar_rate_limit_rps: float = 8.0

    # The company universe. Small and fixed on purpose: a tight corpus makes
    # retrieval failures legible, and legible failures are what you talk about
    # in an interview.
    universe: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "COST", "JNJ", "CAT", "UNH", "PG"]
    )
    forms: list[str] = Field(default_factory=lambda: ["10-K"])
    years_back: int = 3

    # ------------------------------------------------------------- providers
    llm_provider: Literal["openai", "fake"] = "fake"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    cost_prompt_per_mtok: float = 0.15
    cost_completion_per_mtok: float = 0.60

    embed_provider: Literal["local", "api", "fake"] = "local"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_batch: int = 64
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------- retrieval
    default_strategy: str = "section_aware"
    retrieve_k: int = 50
    rerank_top_n: int = 8
    # Reciprocal-rank-fusion constant. 60 is the value from the original RRF
    # paper; it is deliberately not tuned, because tuning it on the same set
    # used for reporting would leak.
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    # ----------------------------------------------------------------- agent
    agent_max_steps: int = 8
    agent_max_retries: int = 2
    agent_confidence_floor: float = 0.55
    agent_max_tool_output_chars: int = 8000

    # ------------------------------------------------------------- eval/gate
    eval_numeric_tolerance: float = 0.005  # 0.5% relative tolerance
    eval_baseline_label: str = "baseline"
    reports_dir: str = "reports"

    # ------------------------------------------------------------ observability
    otlp_endpoint: str = ""
    service_name: str = "edgar-intel"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Dollar cost of one call. Reported per-request and per-1k-requests."""
        return (
            prompt_tokens * self.cost_prompt_per_mtok
            + completion_tokens * self.cost_completion_per_mtok
        ) / 1_000_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests mutate the environment; this lets them pick up the change."""
    get_settings.cache_clear()
