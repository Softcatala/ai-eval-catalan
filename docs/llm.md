# LLM — Avaluació de models de llenguatge

El pipeline `llm/model.py` avalua models GGUF (via `llama-server`) i models d'API de Google AI (Gemini/Gemma), OpenAI i serveis compatibles, inclòs OpenRouter, sobre benchmarks de català:

| Benchmark | Tasca | Mètrica |
|-----------|-------|---------|
| **STS-ca** | Similitud semàntica de frases | Correlació de Pearson |
| **CatCoLA** | Acceptabilitat gramatical | MCC |
| **CLUB / VilaQuAD** | Comprensió lectora (QA) | F1 de solapament de tokens |
| **CaSum** | Resum de notícies en català | ROUGE-1/2/L |
| **FLORES+** | Traducció automàtica EN↔CA i ES↔CA | COMET |
| **IFEval-ca** | Seguiment d'instruccions | Accuracy |
| **Mantinc** | Manteniment del català en prompts adversaris | Pass rate |

FLORES+ executa quatre tasques: `catalan_bench_flores_en-ca`,
`catalan_bench_flores_ca-en`, `catalan_bench_flores_es-ca` i
`catalan_bench_flores_ca-es`. Les taules mostren dues columnes bidireccionals:
`EN↔CA` i `ES↔CA`, cadascuna calculada com la mitjana del COMET de
les dues direccions.

El contracte de `llm/llms.json` publica aquestes mitjanes com `flores_en_ca` i
`flores_es_ca`. També conserva `flores_en2ca`, `flores_ca2en`, `flores_es2ca` i
`flores_ca2es` per a diagnòstic, però aquestes quatre columnes direccionals no es
mostren a les taules ni als gràfics per defecte.

Per calcular un baseline de COMET sense executar un traductor, el script usa la
còpia de la font com a hipòtesi (`mt = src`) sobre FLORES devtest. Avalua les
quatre direccions i, per defecte, usa 400 exemples per direcció:

```bash
cd llm
uv run python comet_baseline.py \
  --output evals/comet_source_copy_baseline.json
```

Podeu canviar-ne la mida amb `--n-samples`.

Amb el checkpoint `Unbabel/wmt22-comet-da` i 400 exemples de FLORES devtest per
direcció, el baseline de còpia de font és: EN→CA 0.6809, CA→EN 0.7549, ES→CA
0.8222 i CA→ES 0.8228.

Cada fila dels JSON publicats inclou `repo_url`, calculat amb el helper compartit
`eval_common.model_urls`.

Per evitar donar doble pes a la traducció, CLAM calcula primer
`translation_score` com la mitjana dels valors normalitzats disponibles de `flores_en_ca` i
`flores_es_ca`. El resultat final és la mitjana dels benchmarks normalitzats que no són de
traducció i aquest únic `translation_score`, multiplicada per 100.

CLAM també inclou `Mantinc`: el `pass_rate` de prompts de manteniment del
català de [Softcatalà/mantinc-catalan-drift](https://github.com/Softcatala/mantinc-catalan-drift),
executat localment amb lm-evaluation-harness.

Les dades, la tasca lm-eval i l'avaluador s'instal·len com una única
dependència Git de Mantinc fixada a un commit concret; no cal tenir-ne un
checkout local ni descarregar el dataset per separat:
`uv run python model.py --benchmarks catalan_drift`.

### Mida de model recomanada segons la memòria

L'eina [`models_recommendation.py`](../models_recommendation.py), situada a l'arrel, recomana
models locals de LLM, embeddings i ASR segons les avaluacions i la memòria
disponible. Executeu `make recommendation` des de l'arrel o
`make recommendation RECOMMENDATION_ARGS="--help"` per consultar les opcions.
Reserva un 25% fix de la memòria i assumeix que cada model s'executa
individualment. `--format table` (per defecte) mostra l'informe de consola amb
alternatives a menys de 2 punts CLAM. Dins d’aquest marge respecte de la millor
puntuació, es recomana el model que ocupa menys memòria. `--format json` aplica
el mateix criteri i mostra una alternativa, considerant tots els LLM que caben.
`--category asr --format json` exporta les recomanacions d'ASR segons el WER combinat.
Tots els formats exclouen els models marcats com a `quantized_analysis_only`
a `llm/models_config.py` o al JSON d'avaluació.

Per a models GGUF quantitzats amb **Q4_K_M**, aquestes són les mides orientatives
segons la memòria disponible del sistema:

| Memòria RAM | Mida recomanada | Límit aproximat |
|-------------|-----------------|-----------------|
| 8 GB | 7–8B | 9B, amb poc context |
| 16 GB | 12–14B | 14B |
| 32 GB | 24–27B | 30–32B, amb menys marge |

La mida que cap a la memòria no implica que sigui la millor recomanació: si
les puntuacions CLAM són dins del marge, l’eina prioritza menys consum de memòria
i més marge per al context.

Cal reservar aproximadament un 20–25% de la memòria per al sistema operatiu, el
motor d'inferència i la memòria cau KV. Un context llarg consumeix més memòria;
en aquest cas, convé triar un model de la franja inferior. Les xifres són
orientatives i poden variar segons l'arquitectura del model i la configuració de
`llama.cpp`.

### Criteri per als repositoris GGUF

Per simplificar el manteniment, es prioritzen els repositoris d'`unsloth` per a
models GGUF quantitzats. S'accepten excepcions quan no ofereixin el model o la
variant exacta necessària (especialment `Q4_K_M`), o quan calgui conservar una
avaluació reproduïble ja associada a un altre repositori. En cada cas s'ha de
registrar el repositori, el fitxer GGUF i la quantització emprada.

### Instal·lació (LLM)

Requereix [uv](https://docs.astral.sh/uv/) i [llama.cpp](https://github.com/ggml-org/llama.cpp). Per als models GGUF locals, engega un `llama-server` extern; per defecte les eines esperen l'endpoint OpenAI-compatible a `http://localhost:9090/v1`.

#### Instal·lació de llama.cpp

Instal·la una versió precompilada amb Homebrew (macOS/Linux) o Conda (Windows/macOS/Linux):

```bash
brew install llama.cpp
conda install -c conda-forge llama.cpp
```

O [compila'l](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) amb Git, CMake i un compilador de C/C++:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
# CPU; a macOS, Metal s'activa per defecte
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Per a NVIDIA amb el CUDA Toolkit, usa en canvi:
# cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
export PATH="$PWD/build/bin:$PATH"
llama-server --version
```

Des de l'arrel del repositori, instal·la les dependències Python:

```bash
cd llm
uv sync
```

#### Configuració d'inferència

Els paràmetres comuns es defineixen a `llm/inference.yaml` amb noms OpenAI-compatible:

```yaml
temperature: 0
max_tokens: 256
reasoning_effort: none
```

Cada benchmark pot sobreescriure `max_tokens`. Els adaptadors tradueixen aquesta configuració als camps i valors compatibles amb llama.cpp, OpenAI, OpenRouter o Gemini.

### Execució (LLM)

Executa les ordres `make` i `llama-server` des de l'arrel del repositori; les ordres Python, des de `llm/`. Pots passar opcions a `make llm-eval` amb `LLM_ARGS`.

**1. Descarregar els GGUF necessaris:**

```bash
make llm-download-ggufs GGUF_DIR=models/gguf
make llm-download-ggufs GGUF_DIR=models/gguf GGUF_MODELS="gemma3-12b salamandra-7b"
```

El target llegeix `MODELS` de `llm/models_config.py`, filtra els models locals GGUF i descarrega els fitxers al subdirectori indicat per `GGUF_DIR` (relatiu a l'arrel del repositori).
També escriu `presets.ini` al mateix directori, amb els ids de model que usa l'avaluador.

**2. Engegar `llama-server`:**

```bash
llama-server --models-preset models/gguf/presets.ini --models-max 1 --port 9090 --reasoning off
```

Per defecte les eines fan servir `http://localhost:9090/v1`. Es pot canviar amb `--server-url` o amb `LLAMA_SERVER_URL`.

**3. Executar l'avaluació local:**

```bash
make llm-eval LLM_ARGS="--models salamandra-7b"
make llm-eval LLM_ARGS="--models gemma3-12b --n-samples 200"
make llm-eval LLM_ARGS="--models gemma3-12b --benchmarks catcola flores"
```

Amb `presets.ini`, l'orquestrador permet avaluar diversos models per execució amb `--models`. Només `--server-model` requereix seleccionar exactament un model local.

L'orquestrador usa 400 mostres per benchmark per defecte i omet els models que ja tenen un fitxer de resultats. Per repetir benchmarks concrets i fusionar-los amb els resultats existents, usa `--rerun-benchmarks --benchmarks <benchmark>`. En canvi, `model.py` usa 100 mostres per defecte; en tots dos casos es pot indicar `--n-samples`.

Si el servidor requereix un identificador de model concret:

```bash
uv run python model.py --model "unsloth/gemma-3-12b-it-GGUF:Q4_K_M" --server-url http://localhost:9090/v1 --server-model unsloth/gemma-3-12b-it-GGUF:Q4_K_M
make llm-eval LLM_ARGS="--models gemma3-12b --server-model gemma-3-12b-it-Q4_K_M"
```

**Avaluar amb l'API de Google AI o OpenAI:**

```bash
uv run python model.py --model gemini --api-key "LA_TEVA_CLAU" --gemini-model gemini-3.6-flash
OPENAI_API_KEY="LA_TEVA_CLAU" uv run python model.py --model openai --openai-model gpt-4o
```

Els resultats es desen com a JSON a `llm/evals/`. L'orquestrador assigna un fitxer a cada model; si executes `model.py` directament, indica `--output evals/<nom_model>.json` per evitar reutilitzar el fitxer per defecte `evals/catalan_eval_results.json`.

[Torna al README](../README.md)
