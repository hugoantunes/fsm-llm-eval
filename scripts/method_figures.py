r"""Thesis method diagrams as PNG/PDF (and an optional Word pack).

Word does not render Mermaid. These plates match T-20: 16 cm wide, 300 dpi PNG,
vector PDF, Arial, no embedded title. Portuguese labels; English identifiers
where the figure documents the implementation.

Usage::

    uv run python scripts/method_figures.py
    uv run python scripts/method_figures.py --docx
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from os import environ
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.patheffects import withStroke

from sim.fsm import FsmSpec, load_fsm
from sim.reporting import FIGURE_WIDTH_IN, PNG_DPI, _configure_matplotlib, _pyplot

FIGURES_DIR = Path("docs/figures")
ARCHITECTURE_STEM = "metodos_arquiteturas"
FLOW_STEM = "metodos_fluxo_avaliacao"
FSM_STEM = "apendice_maquina_estados"
DOCX_NAME = "figuras_para_o_tcc.docx"

BASELINE_FACE = "#56B4E9"
BASELINE_LIGHT = "#D4EAF7"
FSM_FACE = "#E69F00"
FSM_LIGHT = "#F8E4B5"
SHARED_FACE = "#009E73"
NEUTRAL_FACE = "#F2F2F2"
MOTIVE_FACE = "#F4E3D7"
HUMAN_FACE = "#EFE3F7"
EDGE = "#222222"
TEXT = "#111111"

STATE_PT = {
    "greeting": "saudação",
    "identification": "identificação",
    "intent_classification": "classificação da demanda",
    "data_collection": "coleta de dados",
    "solution": "solução",
    "confirmation": "confirmação",
    "closing": "encerramento",
    "out_of_scope": "fora de escopo",
}


@dataclass(frozen=True)
class Box:
    """Axis-fraction rectangle used as an anchor for arrows."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.cx, self.y + self.h

    @property
    def bottom(self) -> tuple[float, float]:
        return self.cx, self.y

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.cy

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.cy


def _axes(plt: object, *, height_in: float):
    """Return a blank axes in axis-fraction coordinates."""
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, height_in), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _rounded(
    ax,
    box: Box,
    *,
    facecolor: str,
    lw: float = 0.9,
    linestyle: str = "-",
    radius: float = 0.012,
) -> None:
    patch = FancyBboxPatch(
        (box.x, box.y),
        box.w,
        box.h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=EDGE,
        linewidth=lw,
        linestyle=linestyle,
        mutation_aspect=1,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(patch)


def _accepting(ax, box: Box, *, facecolor: str) -> None:
    """Draw an accepting state with a visible double border."""
    outer = Box(box.x - 0.006, box.y - 0.006, box.w + 0.012, box.h + 0.012)
    _rounded(ax, outer, facecolor=EDGE, radius=0.014, lw=0.0)
    _rounded(ax, box, facecolor=facecolor, lw=1.15)


def _label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8.5,
    weight: str = "normal",
    color: str = TEXT,
    ha: str = "center",
    va: str = "center",
    style: str = "normal",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        fontstyle=style,
        color=color,
        zorder=3,
        linespacing=1.25,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    lw: float = 1.05,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=EDGE,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=1,
            clip_on=False,
        )
    )


def _edge_label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.6,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=TEXT,
        zorder=4,
        linespacing=1.15,
        path_effects=[withStroke(linewidth=3.2, foreground="white")],
    )


def build_architecture(plt: object | None = None):
    """Two-column message path: one prompt vs classifier, engine, state package."""
    if plt is None:
        plt = _pyplot()
    fig, ax = _axes(plt, height_in=7.6)

    constants = Box(0.03, 0.905, 0.94, 0.075)
    _rounded(ax, constants, facecolor=SHARED_FACE, lw=1.15)
    _label(
        ax,
        constants.cx,
        constants.cy + 0.012,
        "Constantes do experimento (iguais nos dois agentes)",
        size=8.0,
        weight="bold",
        color="white",
    )
    _label(
        ax,
        constants.cx,
        constants.cy - 0.018,
        "mesmo modelo gerador (qwen3.5:9b)   ·   mesma base de conhecimento completa"
        "   ·   mesmo bloco compartilhado",
        size=7.4,
        color="white",
    )

    user = Box(0.28, 0.805, 0.44, 0.075)
    _rounded(ax, user, facecolor=NEUTRAL_FACE)
    _label(ax, user.cx, user.cy, "Mensagem do cliente", size=9.5, weight="bold")

    baseline_col = Box(0.03, 0.255, 0.45, 0.515)
    fsm_col = Box(0.52, 0.255, 0.45, 0.515)
    ax.add_patch(
        Rectangle(
            (baseline_col.x, baseline_col.y),
            baseline_col.w,
            baseline_col.h,
            facecolor="#D7EEF8",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (fsm_col.x, fsm_col.y),
            fsm_col.w,
            fsm_col.h,
            facecolor="#F8E8C4",
            edgecolor="none",
            zorder=0,
        )
    )

    b_head = Box(0.055, 0.695, 0.40, 0.055)
    f_head = Box(0.545, 0.695, 0.40, 0.055)
    _rounded(ax, b_head, facecolor=BASELINE_FACE)
    _rounded(ax, f_head, facecolor=FSM_FACE)
    _label(ax, b_head.cx, b_head.cy, "Baseline", size=10.5, weight="bold")
    _label(ax, f_head.cx, f_head.cy, "FSM", size=10.5, weight="bold")

    prompt = Box(0.055, 0.275, 0.40, 0.395)
    _rounded(ax, prompt, facecolor=BASELINE_LIGHT)
    _label(
        ax,
        prompt.cx,
        0.62,
        "Prompt único do sistema",
        size=9.0,
        weight="bold",
    )
    _label(
        ax,
        prompt.cx,
        0.455,
        "bloco compartilhado\n(persona, tom, regras gerais)\n\n"
        "um procedimento\n(todo o fluxo no mesmo texto)\n\n"
        "base de conhecimento completa",
        size=8.0,
    )

    classifier = Box(0.545, 0.575, 0.40, 0.095)
    engine = Box(0.545, 0.425, 0.40, 0.115)
    package = Box(0.545, 0.275, 0.40, 0.115)
    _rounded(ax, classifier, facecolor=FSM_LIGHT)
    _rounded(ax, engine, facecolor=FSM_LIGHT)
    _rounded(ax, package, facecolor=FSM_LIGHT)
    _label(
        ax, classifier.cx, 0.635, "Classificador de eventos", size=8.8, weight="bold"
    )
    _label(
        ax, classifier.cx, 0.598, "regras + qwen3.5:4b  ·  eventos do estado", size=7.2
    )
    _label(ax, engine.cx, 0.497, "Motor da FSM", size=8.8, weight="bold")
    _label(
        ax,
        engine.cx,
        0.455,
        "estado atual → transições e guardas\n→ estado seguinte",
        size=7.2,
    )
    _label(ax, package.cx, 0.355, "Pacote do estado atual", size=8.8, weight="bold")
    _label(
        ax,
        package.cx,
        0.312,
        "instruções daquele estado\n+ bloco compartilhado + BC completa",
        size=7.2,
    )

    generator = Box(0.18, 0.115, 0.64, 0.09)
    reply = Box(0.28, 0.02, 0.44, 0.065)
    _rounded(ax, generator, facecolor=SHARED_FACE, lw=1.15)
    _rounded(ax, reply, facecolor=NEUTRAL_FACE)
    _label(
        ax,
        generator.cx,
        generator.cy + 0.014,
        "Modelo gerador (qwen3.5:9b) - o mesmo",
        size=9.0,
        weight="bold",
        color="white",
    )
    _label(
        ax,
        generator.cx,
        generator.cy - 0.018,
        "lê o prompt montado acima e a história do diálogo",
        size=7.2,
        color="white",
    )
    _label(ax, reply.cx, reply.cy, "Resposta do agente", size=9.0, weight="bold")

    _arrow(ax, user.bottom, (b_head.cx, 0.75))
    _arrow(ax, user.bottom, (f_head.cx, 0.75))
    _arrow(ax, (b_head.cx, b_head.y), (prompt.cx, prompt.y + prompt.h))
    _arrow(ax, (f_head.cx, f_head.y), classifier.top)
    _arrow(ax, classifier.bottom, engine.top)
    _arrow(ax, engine.bottom, package.top)
    _arrow(ax, (prompt.cx, prompt.y), (prompt.cx, generator.y + generator.h))
    _arrow(ax, (package.cx, package.y), (package.cx, generator.y + generator.h))
    _arrow(ax, generator.bottom, reply.top)
    return fig


def build_evaluation_flow(plt: object | None = None):
    """Experiment pipeline, with human scoring as a response to judge limits."""
    if plt is None:
        plt = _pyplot()
    fig, ax = _axes(plt, height_in=8.4)

    steps = [
        (
            Box(0.12, 0.875, 0.76, 0.095),
            NEUTRAL_FACE,
            "1. Cenários congelados",
            "dataset v1  ·  N = 60  ·  mesmos roteiros para os dois agentes",
        ),
        (
            Box(0.12, 0.735, 0.76, 0.095),
            NEUTRAL_FACE,
            "2. Diálogos",
            "sim run  ·  baseline e FSM  ·  mesmo usuário simulado  ·  mesmo gerador",
        ),
        (
            Box(0.12, 0.595, 0.76, 0.095),
            NEUTRAL_FACE,
            "3. Avaliação automática",
            "juiz cego (duas chamadas) + avaliadores determinísticos  ·  sim eval",
        ),
        (
            Box(0.12, 0.365, 0.76, 0.095),
            HUMAN_FACE,
            "4. Avaliação humana cega",
            "348 diálogos da população semântica primária  ·  IDs D001-D348",
        ),
        (
            Box(0.12, 0.225, 0.76, 0.095),
            HUMAN_FACE,
            "5. Congelamento das anotações",
            "planilhas congeladas em 2026-09-20  ·  antes de qualquer desvelamento",
        ),
        (
            Box(0.12, 0.085, 0.76, 0.095),
            SHARED_FACE,
            "6. Desvelamento e comparação",
            "humano vs juiz  ·  testes pareados por cenário  ·  Holm nas três métricas",
        ),
    ]
    for box, face, title, detail in steps:
        text_color = "white" if face == SHARED_FACE else TEXT
        _rounded(ax, box, facecolor=face)
        _label(
            ax,
            box.cx,
            box.cy + 0.016,
            title,
            size=9.5,
            weight="bold",
            color=text_color,
        )
        _label(ax, box.cx, box.cy - 0.018, detail, size=7.4, color=text_color)

    motive = Box(0.16, 0.485, 0.68, 0.085)
    _rounded(ax, motive, facecolor=MOTIVE_FACE, linestyle=(0, (4, 2.5)), lw=1.2)
    _label(
        ax,
        motive.cx,
        motive.cy + 0.016,
        "Motivação da etapa humana: limitações observadas no juiz",
        size=8.3,
        weight="bold",
    )
    _label(
        ax,
        motive.cx,
        motive.cy - 0.018,
        "concordância apenas moderada no piloto  ·  cegueira de estilo não garantida",
        size=7.2,
    )

    pairs = [
        (steps[0][0], steps[1][0]),
        (steps[1][0], steps[2][0]),
        (steps[2][0], motive),
        (motive, steps[3][0]),
        (steps[3][0], steps[4][0]),
        (steps[4][0], steps[5][0]),
    ]
    for start, end in pairs:
        _arrow(ax, start.bottom, end.top)
    return fig


def _state_label(name: str) -> str:
    return f"{name}\n{STATE_PT[name]}"


def _transition_label(event: str, guard: str | None) -> str:
    if not guard:
        return event
    return f"{event}\n[{guard}]"


def build_fsm(spec: FsmSpec, plt: object | None = None):
    """States and transitions of ``machine.yaml``, for the appendix."""
    if plt is None:
        plt = _pyplot()
    fig, ax = _axes(plt, height_in=9.0)

    column_x, column_w, height = 0.08, 0.46, 0.072
    chain = [
        "greeting",
        "identification",
        "intent_classification",
        "data_collection",
        "solution",
        "confirmation",
        "closing",
    ]
    top = 0.90
    boxes: dict[str, Box] = {}
    cursor = top
    for name in chain:
        boxes[name] = Box(column_x, cursor - height, column_w, height)
        cursor -= height + 0.046
    boxes["out_of_scope"] = Box(0.62, boxes["greeting"].y, 0.32, height)

    start = Box(0.21, boxes["greeting"].y + height + 0.022, 0.20, 0.028)
    end = Box(0.21, boxes["closing"].y - 0.05, 0.20, 0.028)
    _rounded(ax, start, facecolor="#444444", radius=0.02)
    _rounded(ax, end, facecolor="#444444", radius=0.02)
    _label(ax, start.cx, start.cy, "início", size=7.0, color="white")
    _label(ax, end.cx, end.cy, "fim", size=7.0, color="white")

    accepting = set(spec.accepting_states)
    for name, box in boxes.items():
        if name in accepting:
            _accepting(ax, box, facecolor=SHARED_FACE)
            _label(ax, box.cx, box.cy, _state_label(name), size=7.5, color="white")
        else:
            _rounded(ax, box, facecolor=FSM_LIGHT)
            _label(ax, box.cx, box.cy, _state_label(name), size=7.5)

    _arrow(ax, start.bottom, boxes["greeting"].top)
    _arrow(ax, boxes["closing"].bottom, end.top)

    flow = [edge for edge in spec.transitions if not edge.from_any]
    for edge in flow:
        src, dest = boxes[edge.source], boxes[edge.dest]
        label = _transition_label(edge.event, edge.guard)
        if edge.source == "confirmation" and edge.dest == "solution":
            _arrow(ax, (src.x, src.cy), (dest.x, dest.cy), rad=-0.32)
            _edge_label(
                ax, src.x - 0.01, (src.cy + dest.cy) / 2, label, size=6.4, ha="right"
            )
            continue
        if edge.source == "out_of_scope" and edge.dest == "identification":
            _arrow(ax, src.left, dest.right, rad=0.08)
            _edge_label(ax, 0.585, dest.cy + 0.04, label, size=6.2)
            continue
        if edge.source == "out_of_scope" and edge.dest == "closing":
            rail = 0.97
            ax.add_patch(
                FancyArrowPatch(
                    (src.x + src.w, src.cy),
                    (rail, src.cy),
                    arrowstyle="-",
                    linewidth=1.05,
                    color=EDGE,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                    clip_on=False,
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    (rail, src.cy),
                    (rail, dest.cy),
                    arrowstyle="-",
                    linewidth=1.05,
                    color=EDGE,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                    clip_on=False,
                )
            )
            _arrow(ax, (rail, dest.cy), dest.right)
            _edge_label(ax, 0.80, (src.y + dest.cy) / 2, label, size=6.2, ha="center")
            continue
        _arrow(ax, src.bottom, dest.top)
        _edge_label(ax, src.cx, (src.y + dest.y + dest.h) / 2, label, size=6.4)

    from_any: dict[str, list[str]] = {}
    for transition in spec.transitions:
        if not transition.from_any:
            continue
        events = from_any.setdefault(transition.dest, [])
        if transition.event not in events:
            events.append(transition.event)
    note_lines = [
        f"{event} → {dest}" for dest, events in from_any.items() for event in events
    ]
    note = Box(0.58, 0.015, 0.38, 0.12)
    _rounded(ax, note, facecolor=NEUTRAL_FACE, linestyle=(0, (4, 2.5)))
    _label(
        ax,
        note.cx,
        note.cy + 0.032,
        "De qualquer estado",
        size=7.2,
        weight="bold",
    )
    _label(ax, note.cx, note.cy - 0.012, "\n".join(note_lines), size=6.4)
    _label(
        ax,
        0.31,
        end.y - 0.028,
        "borda dupla: estados de aceitação (não terminais)",
        size=6.2,
        color="#444444",
    )
    return fig


def _legend_text(*, number: str, title: str, notes: str) -> str:
    return (
        f"Título: {number} — {title}.\n"
        "Fonte: elaboração própria.\n"
        f"Notas: {notes} Legenda abaixo da figura (ABNT); a imagem não traz "
        "título embutido.\n"
    )


def _save(fig: object, stem: Path, plt: object) -> None:
    fig.savefig(f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_method_figures(out_dir: Path = FIGURES_DIR) -> Path:
    """Write the three method plates and their ABNT captions."""
    plt = _pyplot()
    _configure_matplotlib(plt)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_fsm()

    plates = (
        (
            ARCHITECTURE_STEM,
            build_architecture(plt),
            "Figura A",
            "Comparação das arquiteturas: caminho da mensagem no baseline e na FSM",
            (
                "O baseline responde com um único prompt de sistema. A FSM classifica "
                "o evento do usuário, aplica transições e guardas, e o gerador fala a "
                "partir do pacote de instrução do estado atual. Ambos usam o mesmo "
                "modelo gerador (qwen3.5:9b) e a mesma base de conhecimento completa, "
                "embutida no prompt; o bloco compartilhado (persona, tom e regras "
                "gerais) também é o mesmo. O classificador (regras + qwen3.5:4b) "
                "pertence só à FSM."
            ),
        ),
        (
            FLOW_STEM,
            build_evaluation_flow(plt),
            "Figura B",
            "Fluxo do experimento e da avaliação",
            (
                "Cenários congelados geram diálogos com os dois agentes e o mesmo "
                "usuário simulado. A avaliação automática (juiz cego, duas chamadas, "
                "mais avaliadores determinísticos) precedeu o censo humano. A "
                "avaliação humana cega dos 348 diálogos foi motivada pelas limitações "
                "observadas no juiz (concordância apenas moderada no piloto; cegueira "
                "de estilo não garantida). As anotações foram congeladas em "
                "2026-09-20, antes do desvelamento e da comparação pareada."
            ),
        ),
        (
            FSM_STEM,
            build_fsm(spec, plt),
            "Figura C",
            "Máquina de estados da FSM",
            (
                "Nomes conforme data/fsm/machine.yaml. Identificadores em inglês são "
                "os da implementação; o glossário em português vai entre parênteses. "
                "As transições a partir de qualquer estado (out_of_scope_request e "
                "farewell) saem da caixa «qualquer estado», não como uma seta por "
                "estado. Guardas: order_and_email_present e required_data_collected. "
                "closing e out_of_scope são estados de aceitação, não estados "
                "terminais."
            ),
        ),
    )
    for stem, fig, number, title, notes in plates:
        path = out_dir / stem
        _save(fig, path, plt)
        (out_dir / f"{stem}.legenda.txt").write_text(
            _legend_text(number=number, title=title, notes=notes),
            encoding="utf-8",
        )
    return out_dir


def write_docx(out_dir: Path = FIGURES_DIR) -> Path:
    """Pack the PNGs into a Word file the thesis can copy from."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    captions = []
    for stem in (ARCHITECTURE_STEM, FLOW_STEM, FSM_STEM):
        text = (out_dir / f"{stem}.legenda.txt").read_text(encoding="utf-8")
        captions.append((stem, text))

    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    def _set_run_font(
        run, *, size: float, bold: bool = False, italic: bool = False
    ) -> None:
        run.font.name = "Arial"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    intro = document.add_paragraph()
    run = intro.add_run(
        "Figuras de método para colar no TCC. No Word: clique na figura → "
        "Copiar → colar no capítulo. Largura de página = 16 cm. A legenda "
        "ABNT vai abaixo da imagem, não dentro dela. Recolocar os números "
        "(Figura 1, 2, …) na numeração do documento."
    )
    _set_run_font(run, size=11)

    for stem, legend in captions:
        document.add_page_break()
        picture = document.add_paragraph()
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.add_run().add_picture(str(out_dir / f"{stem}.png"), width=Cm(16.0))
        title_line, _, rest = legend.partition("\n")
        caption = document.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        title_run = caption.add_run(title_line.replace("Título: ", ""))
        _set_run_font(title_run, size=10, italic=True)
        notes = document.add_paragraph()
        notes.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        notes_run = notes.add_run(rest.strip())
        _set_run_font(notes_run, size=9)

    path = out_dir / DOCX_NAME
    document.save(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Write method diagrams into ``docs/figures``."""
    parser = argparse.ArgumentParser(
        description="Write thesis method diagrams as PNG/PDF (and optional Word)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURES_DIR,
        help="output directory (default docs/figures)",
    )
    parser.add_argument(
        "--docx",
        action="store_true",
        help="also write a Word file with the three figures already inserted",
    )
    args = parser.parse_args(argv)
    environ.setdefault("SOURCE_DATE_EPOCH", "0")
    written = write_method_figures(args.out)
    print(f"sim: wrote method figures in {written}", file=sys.stdout)
    if args.docx:
        docx = write_docx(args.out)
        print(f"sim: wrote {docx}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
