# Millors models segons la memòria

Des de l'arrel del repositori:

```bash
python3 hardware_models.py
python3 hardware_models.py --memory 4 8 16 32 --reserve-percent 25
python3 hardware_models.py --memory-kind VRAM --format json
python3 hardware_models.py --llm-uncertainty 2
```

També es pot executar des de l'arrel amb `python3 -m hardware_models`.
Reutilitza els càlculs dels scripts de resum i necessita `jinja2`, ja inclòs
a l'entorn d'avaluació (`uv run --project llm python hardware_models.py`). No executa
inferència, no descarrega models i no necessita connexió a Internet.

Llegeix directament els JSON de `llm/evals/`, `embeddings/evals/` i `asr/evals/`.
No utilitza els JSON resumits, que poden estar desactualitzats, ni puntuacions
de benchmarks externs. `--repo-root` permet llegir un altre checkout amb la
mateixa estructura. La sortida JSON conserva el fitxer d'avaluació i la font
de la memòria de cada recomanació.

La sortida de text s'organitza en blocs de **LLM, embeddings i ASR**, en aquest
ordre, amb totes les configuracions de memòria dins de cada bloc.

Per a cada capacitat, selecciona **un model per categoria**, amb aquests criteris:

| Categoria | Criteri de qualitat |
| --- | --- |
| LLM | CLAM més alt, calculat amb la mateixa normalització que `summarize_results.py` |
| Embeddings | Mitjana més alta de XQuAD nDCG@10, STS-ca Spearman i TeCla F1 |
| ASR | WER més baix de FLEURS i OpenSLR-69, ponderat pel nombre de mostres |

Inclou totes les quantitzacions avaluades. Només considera models locals amb
totes les mètriques necessàries i memòria coneguda. Mostra les avaluacions
locals excloses i el motiu. En cas d'empat, prefereix el model més petit.
«Millor» vol dir la millor puntuació observada: no és una comparació de
velocitat ni una afirmació de significació estadística.

Per als LLM, les diferències de **menys de 2 punts CLAM** es consideren
**no concloents**. A més del model amb més puntuació, mostra totes les
alternatives que càpiguen en el pressupost de memòria i quedin a menys de
2 punts del millor model. Per exemple, si el millor obté 60, inclou 58,01
però exclou 58. La comparació es fa sempre respecte del millor, sense
encadenar alternatives entre si. És un criteri per mostrar opcions semblants,
no un interval de confiança ni una prova d'equivalència estadística.

La columna `Δ CLAM` indica quants punts queda cada opció per sota del millor.
`--llm-uncertainty` permet canviar el llindar exclusiu; amb `0` només mostra empats exactes.
La sortida JSON inclou `llm_uncertainty_points` i, per configuració,
`llm_alternatives`, amb `score_gap` per a cada LLM. `llm_uncertainty_points`
representa el llindar de diferència, no un marge ± per model.
Aquest llindar només s'aplica als LLM.

Les configuracions per defecte són **4, 8, 16 i 32 GB de RAM**, amb un **25% de
reserva** per al sistema, el motor, les activacions i la memòria cau. Això deixa
3, 6, 12 i 24 GB per als models. `--reserve-percent` permet ajustar aquest marge.
`--memory-kind VRAM` aplica el mateix pressupost a la memòria de la GPU, assumint
que el model s'hi carrega sencer; no suma RAM i VRAM ni calcula offloading.

Cada recomanació assumeix que s'executa **individualment**. No garanteix que els
tres models seleccionats càpiguen alhora. La memòria és orientativa: els
contextos llargs, lots grans o altres aplicacions poden exigir més reserva.

Per als LLM i ASR, reutilitza `memory_gb` de l'avaluació. Per als embeddings,
prioritza aquest camp si existeix; si manca, consulta `embedding_memory.json`,
un catàleg local amb fonts de Hugging Face fixades a una revisió. Estima els
pesos en FP32 a 4 bytes per paràmetre, en GB decimals i arrodonint cap amunt.
Aquestes estimacions no són mesures del pic de memòria durant la inferència.
El catàleg s'ha d'ampliar quan s'avaluen nous embeddings sense `memory_gb`.
