# JEV — Avaluació de models de decisió

`jev/` avalua Julia, Laya, Kev (0.8B, 4B i 9B), lev, OpenJev, Clef Flash (9B),
Clef (27B), Nimble (9B), Rune v3 (26B-A4B), Liquid d1 (3B i Omni 600M)
i GPT-6 Luna Decisions sobre les 18 categories del test MASSIVE 1.1 en català.
Mesura accuracy, macro F1 i latència mitjana. Requereix Python ≥ 3.10 i
`llama-server` amb `/v1/systemone`. Executa les ordres següents des de l'arrel
del repositori:

```bash
llama-server -hf ggml-org/Julia-1-GGUF:Q8_0 --port 9090
make jev-eval
```

Per a Clef i Clef Flash, afegeix `--batch-size 2048 --ubatch-size 2048` al servidor.
Per a Liquid d1 Omni 600M, usa `--batch-size 4096 --ubatch-size 4096`.
Per a Rune v3, usa `owao/surogate-rune-26b-a4b-GGUF:Q8_0` (amb metadades de decisió).
`--server-url` o `LLAMA_SERVER_URL` configura l'endpoint (per defecte,
`http://localhost:9090/v1`); `SYSTEMONE_API_KEY`, l'autenticació.

Per a GPT-6 Luna Decisions, configura `OPENAI_API_KEY` i executa
`make jev-eval JEV_ARGS="--provider openai --models gpt-6-luna"` (API `/v1/decisions`).

L'orquestrador descobreix models a `/v1/models`, inclosos els del router, avalua
tot el test i reutilitza resultats completats amb la mateixa configuració:

```bash
make jev-eval JEV_ARGS="--models ggml-org/Julia-1-GGUF:Q8_0 --n-samples 100"
```

MASSIVE es descarrega automàticament a `jev/data/`. Els resultats i les prediccions
es desen a `jev/evals/` (JSON i JSONL); `jev/jevs.json` n'agrega el resum.
Usa `--overwrite` per repetir avaluacions i `--output-dir` per conservar
configuracions separades. Consulta `make jev-eval JEV_ARGS="--help"` per a les
opcions de llengua, ordre de categories i dades locals.

També es pot avaluar Tornem a TE-ca (`projecte-aina/teca`, split `test`),
amb tres opcions: implicació, neutralitat i contradicció. Aquesta avaluació
requereix `datasets` (disponible a l'entorn de `llm/`). Com a ASR, cada fitxer
`jev/evals/<model>.json` conté els dos conjunts dins de `benchmarks.massive`
i `benchmarks.teca`, amb les mètriques, el nombre de mostres i la configuració
de cada avaluació. Les metadades del model són al nivell superior.
La data `evaluated_at` es desa només a cada benchmark. Els camps `requested_model` i
`models` només es desen al nivell superior si difereixen del model del fitxer.
Les seccions dels benchmarks no contenen metadades del model.
Els valors per defecte `locale=ca-ES` i `labels=ca` són implícits i no es desen al JSON.
La selecció és reproduïble amb `--seed` (42 per defecte).
Per afegir tot el test (2.117 exemples) als fitxers existents, usa
`--dataset teca --n-samples 0` sense `--output-dir`. Una nova passada conserva
els resultats de l'altre benchmark. Les prediccions TE-ca es desen en `.teca.jsonl`.

El rànquing JEV s'ordena per la mitjana aritmètica dels macro F1 de MASSIVE
i TE-ca, amb un pes del 50% per a cada dataset. Els macro F1 i les accuracy
es mostren separadament. Només es calcula la mitjana quan el model té les dues avaluacions;
els models pendents apareixen al final sense puntuació global. L'agregador
`python3 -m jev.summarize_results` llegeix tots dos benchmarks de `jev/evals/`
i exigeix els tests complets (2.974 exemples MASSIVE i 2.117 TE-ca) i la mateixa
configuració de llengua, llavor i ordre d'opcions.

Per consultar els resultats en HTML, segueix la [guia d’informes locals](desenvolupament.md#informes-html-de-depuració).

[Torna al README](../README.md)
