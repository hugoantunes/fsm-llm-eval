"""Load ``configs/models.yaml``, the single source of models and parameters (T-07).

Model names, digests, ``num_ctx``, temperatures and seeds are experiment
decisions: they live in the config file, never as defaults in code, and the run
manifest of T-14a records what was read. Everything that talks to Ollama goes
through this loader, so a value is written once and read everywhere.
"""

from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_CONFIG_PATH = Path("configs/models.yaml")

#: The model roles of the experiment. The two agents share one model; simulated
#: user, event classifier and stage labeler are independently configurable.
Role = Literal["agent", "simulated_user", "classifier", "state_labeler", "judge"]

#: The same three as values, for iteration and for error messages.
ROLES: tuple[Role, ...] = get_args(Role)


class ConfigError(ValueError):
    """The config file is malformed; the message says what to fix and where."""


class ModelSpec(BaseModel):
    """One model and the sampling parameters fixed for it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    digest: str | None = None
    temperature: float | None = None
    seed: int | None = None


class ModelsByRole(BaseModel):
    """The configured models of the experiment, one per role."""

    model_config = ConfigDict(extra="forbid")

    agent: ModelSpec
    simulated_user: ModelSpec
    classifier: ModelSpec
    state_labeler: ModelSpec
    judge: ModelSpec


class OllamaSpec(BaseModel):
    """The Ollama version and the server environment both machines apply."""

    model_config = ConfigDict(extra="forbid")

    version: str
    env: dict[str, int]


class ClientSpec(BaseModel):
    """How the client of T-07 behaves when a call is slow or fails."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)


class ModelsConfig(BaseModel):
    """Everything in ``configs/models.yaml``, parsed and typed."""

    model_config = ConfigDict(extra="forbid")

    ollama: OllamaSpec
    num_ctx: int = Field(gt=0)
    max_prompt_fraction: float = Field(gt=0, le=1)
    client: ClientSpec
    models: ModelsByRole

    @property
    def max_prompt_tokens(self) -> int:
        """The prompt-token budget above which a call is refused (T-07)."""
        return int(self.num_ctx * self.max_prompt_fraction)

    def spec(self, role: Role) -> ModelSpec:
        """Return the model and parameters configured for ``role``."""
        return getattr(self.models, role)


def load_models_config(path: Path = DEFAULT_CONFIG_PATH) -> ModelsConfig:
    """Load and validate the model configuration stored at ``path``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return ModelsConfig.model_validate(raw)
    except ValidationError as invalid:
        raise ConfigError(
            f"{path} does not match the schema of sim.config:\n{invalid}\nA key the "
            f"schema does not know is refused rather than ignored: a misspelled "
            f"'temperature' would silently fall back to Ollama's default and the two "
            f"agents would no longer be sampled alike"
        ) from invalid
