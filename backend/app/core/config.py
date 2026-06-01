"""
Application configuration module.
Defines global settings for database, LLM, and CUSP model hyperparameters
using Pydantic BaseSettings with environment variable support and validation.
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./mental_health.db"
    LLM_MODEL_NAME: str = r"D:\Models\huggingface\Qwen3.5-2B"
    LLM_DEVICE: str = "cuda"
    LLM_LOAD_IN_4BIT: bool = False
    LLM_MAX_NEW_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.3
    EBIC_GAMMA: float = 0.5
    SCORE_THRESHOLD: float = 0.01
    CUSP_THETA_BIFURCATION: float = 0.5
    ALLOCATION_LAMBDA_A: float = 0.3
    ALLOCATION_LAMBDA_B: float = 0.3
    ALLOCATION_LAMBDA_C: float = 0.2
    STATE_DRIFT_ETA: float = 0.1

    @field_validator("EBIC_GAMMA")
    @classmethod
    def validate_ebic_gamma(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("EBIC_GAMMA must be in [0, 1]")
        return v

    @field_validator("SCORE_THRESHOLD")
    @classmethod
    def validate_score_threshold(cls, v):
        if v <= 0:
            raise ValueError("SCORE_THRESHOLD must be positive")
        return v

    @field_validator("CUSP_THETA_BIFURCATION")
    @classmethod
    def validate_theta(cls, v):
        if not -1.0 <= v <= 2.0:
            raise ValueError("CUSP_THETA_BIFURCATION must be in [-1, 2]")
        return v

    @field_validator("ALLOCATION_LAMBDA_A", "ALLOCATION_LAMBDA_B", "ALLOCATION_LAMBDA_C")
    @classmethod
    def validate_lambda(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Allocation lambda must be in [0, 1]")
        return v

    @field_validator("STATE_DRIFT_ETA")
    @classmethod
    def validate_drift_eta(cls, v):
        if v <= 0:
            raise ValueError("STATE_DRIFT_ETA must be positive")
        return v

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be in [0, 2]")
        return v

    @field_validator("LLM_MAX_NEW_TOKENS")
    @classmethod
    def validate_max_tokens(cls, v):
        if v < 64:
            raise ValueError("LLM_MAX_NEW_TOKENS must be >= 64")
        return v

    class Config:
        env_file = ".env"


settings = Settings()
