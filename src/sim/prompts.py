"""Load the prompt files of ``data/prompts/`` (T-09).

Prompts are text, not Python strings: the agents, the simulated user and the judge
read them from ``data/prompts/*.md`` at run time, so the thesis appendices quote the
same file the experiment ran. Each file opens with a ``Version:`` line, which the run
manifest of T-14a records alongside the model digests.
"""

import re
from pathlib import Path
from string import Template

from pydantic import BaseModel

DEFAULT_PROMPTS_DIR = Path("data/prompts")

#: ``Version: 1``, the first line of every prompt file.
_VERSION = re.compile(r"^Version:\s*(?P<version>\d+)\s*$", re.MULTILINE)


class PromptError(ValueError):
    """A prompt file is malformed or rendered wrong; the message says what to fix."""


class Prompt(BaseModel):
    """One prompt file: its version and the template below the version line."""

    name: str
    version: int
    template: str

    def render(self, **fields: str) -> str:
        """Fill every ``$placeholder`` of the template with ``fields``.

        Both halves of the mismatch are refused: a placeholder nobody filled
        would reach the model as literal ``$knowledge_base``, and a field the
        template stopped using would drop the knowledge base from the prompt
        without a word.
        """
        template = Template(self.template)
        wanted = set(template.get_identifiers())
        given = set(fields)
        if wanted != given:
            raise PromptError(
                f"{self.name}.md takes {sorted(wanted)} and was rendered with "
                f"{sorted(given)}: missing {sorted(wanted - given)}, unused "
                f"{sorted(given - wanted)}. A hole in a prompt reaches the model "
                f"as literal text, and an unused field is a section that silently "
                f"stopped being sent"
            )
        try:
            return template.substitute(fields)
        except ValueError as error:
            raise PromptError(
                f"{self.name}.md has an invalid placeholder ({error}). "
                f"A literal dollar sign is written $$"
            ) from error


def load_prompt(name: str, *, directory: Path = DEFAULT_PROMPTS_DIR) -> Prompt:
    """Load the prompt ``name`` from ``directory/<name>.md``."""
    text = (directory / f"{name}.md").read_text(encoding="utf-8")
    version = _VERSION.search(text)
    if version is None:
        raise PromptError(
            f"{name}.md carries no 'Version: <n>' line. The manifest of T-14a "
            f"records the version of every prompt a run used, and after tag v1 a "
            f"changed prompt is a new version with a dated Decisões row"
        )
    return Prompt(
        name=name,
        version=int(version["version"]),
        template=_VERSION.sub("", text, count=1).strip(),
    )


def load_prompt_versions(directory: Path = DEFAULT_PROMPTS_DIR) -> dict[str, int]:
    """Return ``{name: version}`` for every prompt file in ``directory``."""
    return {
        path.stem: load_prompt(path.stem, directory=directory).version
        for path in sorted(directory.glob("*.md"))
    }
