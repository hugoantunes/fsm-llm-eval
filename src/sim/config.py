"""Load ``configs/models.yaml``, the single source of models and parameters (T-07).

Model names, digests, ``num_ctx``, temperatures and seeds are experiment
decisions: they live in the config file, never as defaults in code, and the run
manifest of T-14a records what was read. Everything that talks to Ollama goes
through this loader, so a value is written once and read everywhere.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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


class ClientPolicy(BaseModel):
    """Optional per-role overlay on the global HTTP client policy."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float | None = Field(default=None, gt=0)
    retries: int | None = Field(default=None, ge=0)
    backoff_s: float | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class ResolvedClientPolicy:
    """The timeout, retries and backoff one role actually uses."""

    timeout_s: float
    retries: int
    backoff_s: float


class ClientSpec(BaseModel):
    """How the client of T-07 behaves when a call is slow or fails."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)
    by_role: dict[str, ClientPolicy] = Field(default_factory=dict)

    @field_validator("by_role")
    @classmethod
    def _known_roles(cls, value: dict[str, ClientPolicy]) -> dict[str, ClientPolicy]:
        unknown = sorted(set(value) - set(ROLES))
        if unknown:
            raise ValueError(
                "unknown client role override(s): "
                + ", ".join(unknown)
                + f"; known roles are {', '.join(ROLES)}"
            )
        return value

    def policy_for(self, role: Role) -> ResolvedClientPolicy:
        """Return the global client policy with any overlay for ``role`` applied."""
        override = self.by_role.get(role)
        if override is None:
            return ResolvedClientPolicy(
                timeout_s=self.timeout_s,
                retries=self.retries,
                backoff_s=self.backoff_s,
            )
        return ResolvedClientPolicy(
            timeout_s=(
                self.timeout_s if override.timeout_s is None else override.timeout_s
            ),
            retries=self.retries if override.retries is None else override.retries,
            backoff_s=(
                self.backoff_s if override.backoff_s is None else override.backoff_s
            ),
        )


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


def with_role_model(config: ModelsConfig, role: Role, name: str) -> ModelsConfig:
    """Copy ``name`` and its digest onto ``role``; keep that role's sampling.

    ``name`` must already appear on another configured role so the digest is
    not invented. Temperature and seed stay those of ``role``.
    """
    donors = [config.spec(item) for item in ROLES if config.spec(item).name == name]
    if not donors:
        configured = ", ".join(sorted({config.spec(item).name for item in ROLES}))
        raise ConfigError(
            f"{name!r} is not a configured model; configured names are {configured}"
        )
    digest = donors[0].digest
    if any(item.digest != digest for item in donors):
        raise ConfigError(
            f"{name!r} has more than one digest in the model config; "
            "the sidecar cannot choose which weights to load"
        )
    updated = config.spec(role).model_copy(update={"name": name, "digest": digest})
    models = config.models.model_copy(update={role: updated})
    return config.model_copy(update={"models": models})


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
