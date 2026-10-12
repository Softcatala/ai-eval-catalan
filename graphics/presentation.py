import json, sys
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageChops, ImageOps

plt.rcParams["font.family"] = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

RED = "#BB2526"
DARK = "#333333"
GREY = "#666666"
LIGHT = "#EFEFEF"
SOFT = "#F6E3E3"

input_path = Path(sys.argv[1])
data = json.loads(input_path.read_text())["data"]
# Use the quantization recorded in each evaluation, rather than the broad "q4" label.
quantizations = {}
for path in (input_path.parent / "evals").glob("*.json"):
    result = json.loads(path.read_text())
    quantizations[result.get("display_name", path.stem)] = result.get("model", "").rsplit(":", 1)[-1]
S = {r["model"].replace("(*) ", ""): r for r in data if not r["quantized_analysis_only"]}
for model in ("qwen3.5-4b", "qwen3.5-9b", "qwen3-14b", "qwen3.8-27b"):
    if quantizations.get(model) != "Q4_K_M":
        raise ValueError(f"Cal una avaluació Q4_K_M per a {model}")
def card(ax, x, y, w, h, fc, ec=None, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec or fc, lw=lw))


def fmt(v):
    return f"{v:.1f}".replace(".", ",")


# Diapositives 8–10 de presentacio.pdf, amb una conclusió principal per gràfic.
big = [("gpt-6-astra", "OpenAI"), ("gemini-3-8-flash", "Google"),
       ("gemini-3-1-preview", "Google"), ("claude-opus-4-7", "Anthropic")]
recs = [("8 GB", "Context curt", "qwen3.5-9b", "Qwen 3.5 · 9B"),
        ("16 GB", "", "qwen3-14b", "Qwen 3 · 14B"),
        ("32 GB", "", "qwen3.8-27b", "Qwen 3.8 · 27B")]
source = f"Font: Softcatalà · {input_path.parent.name}/{input_path.name} · Consulta: {date.today():%d/%m/%Y}"


def slide_canvas(title, subtitle, source):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 120); ax.set_ylim(0, 67.5); ax.axis("off")
    ax.text(6, 61, title, fontsize=23, weight="bold", color=DARK, va="center")
    ax.text(6, 56, subtitle, fontsize=12, color=GREY, va="center")
    ax.plot([6, 114], [52, 52], color=RED, lw=2)
    ax.text(6, 3, source, fontsize=9, color=GREY, va="center")
    return fig, ax


def graph_canvas():
    fig = plt.figure(figsize=(12.8, 4.8), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 120); ax.set_ylim(5, 50); ax.axis("off")
    return fig, ax


def save_slide(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            bounds = text.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise ValueError(f"Text fora de la imatge {name}: {text.get_text()}")
    fig.savefig(name, facecolor="white")
    plt.close(fig)
    with Image.open(name) as image:
        rgb = image.convert("RGB")
        bounds = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        if bounds:
            ImageOps.expand(rgb.crop(bounds), border=16, fill="white").save(name)


# Panoràmica abans de les comparacions de les diapositives 8 i 9.
overview_omitted = {
    "gemma3-27b", "qwen3.5-9b",
    "alia-40b-instruct-2605", "qwen3.5-4b", "gemma3-12b", "mistral-small-24b",
    "gemma4-12b", "glm-4.7-flash", "tiny-aya-water", "gemma4-e4b",
    "llama3.1-8b", "ministral3-14b",
}
overview = sorted(
    (r for name, r in S.items() if name not in overview_omitted
     and (r["cloud"] or quantizations.get(name) == "Q4_K_M")),
    key=lambda r: r["clam"], reverse=True,
)
fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200)
all_overview = sorted(
    (r for name, r in S.items() if r["cloud"] or quantizations.get(name) == "Q4_K_M"),
    key=lambda r: r["clam"], reverse=True,
)
positions, gaps = [], []
pending = 0
y = 0
for row in all_overview:
    if row["model"].removeprefix("(*) ") in overview_omitted:
        pending += 1
        continue
    if pending:
        gaps.append((y, pending))
        y += 1
        pending = 0
    positions.append(y)
    y += 1
colors = [DARK if r["cloud"] else RED for r in overview]
ax.scatter([r["clam"] for r in overview], positions, c=colors, s=70, zorder=3)
ticks = sorted([(y, r["model"].removeprefix("(*) ")) for y, r in zip(positions, overview)]
               + [(y, "⋮") for y, _ in gaps])
ax.set_yticks([y for y, _ in ticks], [label for _, label in ticks], fontsize=11)
for y, count in gaps:
    ax.axhline(y, color="#CCCCCC", lw=1, ls=(0, (2, 4)), zorder=0)
    label = "1 model intermedi omès" if count == 1 else f"{count} models intermedis omesos"
    ax.text(2, y, label, fontsize=10, color=GREY, va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
ax.invert_yaxis()
ax.set_xlim(0, 80)
ax.set_xlabel("Puntuació CLAM · selecció de models", fontsize=12)
for y, r, color in zip(positions, overview, colors):
    ax.annotate(fmt(r["clam"]), (r["clam"], y), xytext=(9, 0),
                textcoords="offset points", va="center", fontsize=10, color=color)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(length=0, colors=GREY, pad=8)
ax.grid(axis="x", color=LIGHT)
ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], color=RED, marker="o", ls="", label="Locals"),
                   Line2D([], [], color=DARK, marker="o", ls="", label="Al núvol")],
          loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
fig.tight_layout()
save_slide(fig, "slide_7.png")

scores = [S[model]["clam"] for model, _ in big]
fig, ax = graph_canvas()
for i, (model, company) in enumerate(big):
    x, cx = 6 + 27.5 * i, 18.75 + 27.5 * i
    card(ax, x, 19, 25.5, 29, SOFT)
    ax.text(cx, 42, company, ha="center", va="center", fontsize=17, weight="bold", color=DARK)
    ax.text(cx, 37, model, ha="center", va="center", fontsize=10, color=GREY)
    ax.text(cx, 29, fmt(S[model]["clam"]), ha="center", va="center", fontsize=39, weight="bold", color=RED)
    ax.text(cx, 22.5, "punts CLAM", ha="center", va="center", fontsize=10, color=GREY)
save_slide(fig, "slide_8.png")

q = S["qwen3.8-27b"]
mini = S["gpt-5.4-mini"]
fig, ax = graph_canvas()
# Escala completa: evita exagerar visualment diferències petites.
def score_x(value):
    return 49 + value * 0.56
for value in range(0, 101, 20):
    x = score_x(value)
    ax.plot([x, x], [15, 48], color=LIGHT, lw=1, zorder=0)
    ax.text(x, 11.5, str(value), fontsize=10, color=GREY, ha="center")
rows = [
    (44, "Quatre models frontera", "de laboratoris d’IA · OpenAI, Google i Anthropic", None, DARK),
    (35, "GPT 5.4 mini", "Al núvol", mini, DARK),
    (26, f"Qwen 3.8 · 27B ({fmt(q['memory_gb'])} GB de RAM)", "Local · Q4_K_M · memòria estimada", q, RED),
    (17, f"Qwen 3.5 · 4B ({fmt(S['qwen3.5-4b']['memory_gb'])} GB de RAM)", "Local · Q4_K_M · memòria estimada", S["qwen3.5-4b"], RED),
]
for y, title, subtitle, row, color in rows:
    ax.text(6, y + 1, title, fontsize=15, weight="bold", color=color, va="center")
    ax.text(6, y - 2.5, subtitle, fontsize=10, color=GREY, va="center")
    if row is None:
        ax.plot([score_x(min(scores)), score_x(max(scores))], [y, y], color=color, lw=9, solid_capstyle="round")
        value_text = f"{fmt(min(scores))}–{fmt(max(scores))}"
    else:
        ax.scatter(score_x(row["clam"]), y, s=170, color=color, zorder=3)
        value_text = fmt(row["clam"])
    ax.text(113, y, value_text, fontsize=14, weight="bold", color=color, ha="right", va="center")
ax.text(6, 7, "CLAM (punts)", fontsize=10, color=GREY)
save_slide(fig, "slide_9.png")


fig, ax = graph_canvas()
alternatives = ["Qwen 3.5 · 4B", "Gemma 3 · 12B", "Muse Glimmer · 30B"]
for i, ((mem, desc, model, nice), alternative) in enumerate(reversed(list(zip(recs, alternatives)))):
    r = S[model]
    x, cx = 6 + 37 * i, 23 + 37 * i
    card(ax, x, 8, 34, 40, SOFT)
    ax.text(cx, 42, "32 GB o més" if mem == "32 GB" else mem,
            ha="center", va="center", fontsize=25 if mem == "32 GB" else 31, weight="bold", color=RED)
    ax.text(cx, 37.5, "de RAM", ha="center", va="center", fontsize=12, color=GREY)
    ax.plot([x + 4, x + 30], [34, 34], color="#D5A6A6", lw=1)
    ax.text(cx, 30, nice, ha="center", va="center", fontsize=17, weight="bold", color=DARK)
    ax.text(cx, 24.5, f"{fmt(r['clam'])} CLAM", ha="center", va="center", fontsize=18, weight="bold", color=RED)
    ax.plot([x + 4, x + 30], [20, 20], color="#D5A6A6", lw=1)
    ax.text(cx, 16.5, "Alternativa", ha="center", va="center", fontsize=11, color=GREY)
    ax.text(cx, 12.5, alternative, ha="center", va="center", fontsize=15, weight="bold", color=DARK)
save_slide(fig, "slide_10.png")

fig, ax = slide_canvas(
    "El criteri humà completa les proves automàtiques",
    "Arena recull preferències sobre respostes en català i construeix un rànquing públic.",
    "Font: arena.softcatala.org · Esquema explicatiu; no representa resultats ni un rànquing actual.",
)
for x, number, title, body in [
    (6, "1", "La mateixa tasca", "Correcció · reformulació\nTraducció · generació"),
    (43, "2", "Dues respostes", "Comparació a cegues:\nsense saber quin model és"),
    (80, "3", "El vostre criteri", "Preferències que alimenten\nel rànquing d’Arena"),
]:
    card(ax, x, 23, 34, 25, SOFT)
    ax.text(x + 4, 42, number, fontsize=24, weight="bold", color=RED, va="center")
    ax.text(x + 17, 35, title, fontsize=16, weight="bold", color=DARK, ha="center", va="center")
    ax.text(x + 17, 28.5, body, fontsize=11, color=DARK, ha="center", va="center", linespacing=1.6)
ax.text(6, 16, "Una puntuació automàtica i una preferència humana aporten informació diferent.", fontsize=13, weight="bold", color=DARK)
ax.text(6, 10, "El rànquing depèn de les tasques, els models i els vots recollits; no determina un guanyador universal.", fontsize=10, color=GREY)
save_slide(fig, "slide_11.png")

fig, ax = slide_canvas(
    "Quina resposta resol millor la tasca?",
    "Exemple il·lustratiu de comparació a cegues · les respostes no provenen de models avaluats",
    "Criteris i opcions de vot: arena.softcatala.org · Exemple redactat per explicar el funcionament.",
)
card(ax, 6, 42, 108, 7, LIGHT)
ax.text(9, 45.5, "Tasca: reformula «No podem publicar la notícia perquè encara no hem contrastat les dades».\nMantén el sentit i fes servir un català natural.", fontsize=11, color=DARK, va="center", linespacing=1.4)
for x, letter, response in [
    (6, "A", "Encara no podem publicar la notícia:\nens falta contrastar les dades."),
    (61, "B", "Ja podem publicar la notícia,\ntot i que les dades no estan contrastades."),
]:
    card(ax, x, 21, 53, 18, SOFT)
    ax.text(x + 4, 34.5, f"Resposta {letter}", fontsize=17, weight="bold", color=RED, va="center")
    ax.text(x + 4, 27.5, response, fontsize=12, color=DARK, va="center", linespacing=1.6)
ax.text(6, 17.5, "Valora: qualitat del català · respecte del significat · compliment de les instruccions", fontsize=11, color=DARK)
for x, width, label_text in [(6, 24, "A és millor"), (33, 24, "B és millor"), (60, 24, "Empat"), (87, 27, "Cap de les dues")]:
    card(ax, x, 9, width, 6, RED)
    ax.text(x + width / 2, 12, label_text, ha="center", va="center", fontsize=12, weight="bold", color="white")
save_slide(fig, "slide_12.png")
fig, ax = slide_canvas(
    "",
    "",
    "",
)
ax.lines[0].remove()
for left, title, rows in [
    (6, "Transcripció", [
        ("Màxima\nprecisió", "whisper-large-v3-\nca-3catparla", "Taxa d’errors:\n5,9%"),
        ("Precisió i\nvelocitat", "whisper-large-v3-\nturbo", "Taxa d’errors:\n6,3%\n6,7 cops més ràpid"),
    ]),
    (61, "Embeddings", [
        ("Màxima\nqualitat", "multilingual-e5-large", "mida del vector: 1.024\nPuntuació: 0,9"),
        ("Qualitat i\nvector petit", "nomic-embed-text-\nv2-moe", "mida del vector: 768\nPuntuació: 0,9"),
    ]),
]:
    card(ax, left, 8, 53, 40, SOFT)
    ax.text(left + 26.5, 42, title, fontsize=25, weight="bold", color=RED, ha="center", va="center")
    ax.plot([left + 2, left + 51], [34, 34], color="#D5A6A6", lw=1)
    for x, label in [(2, "Prioritat"), (16, "Model recomanat"), (37, "Descripció")]:
        ax.text(left + x, 36, label, fontsize=10, weight="bold", color=GREY, va="center")
    for i, (priority, model, description) in enumerate(rows):
        y = 28 - 12 * i
        for x, text, size, weight in [(2, priority, 11, "normal"), (16, model, 11, "bold"), (37, description, 10, "normal")]:
            ax.text(left + x, y, text, fontsize=size, weight=weight, color=DARK, va="center", linespacing=1.5)
        if i == 0:
            ax.plot([left + 2, left + 51], [22, 22], color="#D5A6A6", lw=1)
save_slide(fig, "slide_0.png")
print("Gràfics de les diapositives 0 i 7 a 12 generats")
