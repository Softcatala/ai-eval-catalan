# Eines d'avaluació de models LLM, ASR, embeddings i de decisió

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM), de reconeixement automàtic de la parla (ASR), d'embeddings i de decisió (JEV), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/ia-local/models-en-catala/

## Vols que avaluem un model?

[Obre un issue](https://github.com/Softcatala/ai-eval-catalan/issues/new) indicant l'URL de Hugging Face del model, el tipus (LLM, ASR, embeddings o JEV) i per què és rellevant per al català.

## Documentació

| Guia | Contingut |
|------|-----------|
| [LLM](docs/llm.md) | Benchmarks, puntuació CLAM, instal·lació i avaluació de models locals i d'API |
| [ASR](docs/asr.md) | Preparació de FLEURS i OpenSLR-69 i avaluació de reconeixement de la parla |
| [Embeddings](docs/embeddings.md) | Benchmarks i avaluació de models locals i d'API |
| [JEV](docs/jev.md) | Avaluació de models de decisió sobre MASSIVE i TE-ca |
| [Publicació de resultats](docs/publicacio.md) | JSON de producció, recomanacions i informes HTML locals |

## Estructura del projecte

```
ai-eval-catalan/
├── render_tables.py          # Genera taules HTML per a depuració local
├── eval_common/              # Codi compartit pels generadors de JSON
├── llm/                      # Avaluació de models LLM
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja
│   └── evals/
├── asr/                      # Avaluació de models ASR
│   ├── hf-eval.py            # Avaluació local de WER/CER sobre FLEURS i OpenSLR-69
│   ├── cloud-eval.py         # Avaluació ASR amb APIs d'OpenAI i Gemini
│   ├── dataset_preparation.py          # Prepara el benchmark local FLEURS
│   ├── openslr69_dataset_preparation.py # Prepara el benchmark local OpenSLR-69
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja
│   └── evals/
├── embeddings/               # Avaluació de models d'embeddings
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja
│   └── evals/
└── jev/                      # Avaluació de models de decisió
    ├── model.py              # Avaluador MASSIVE i TE-ca
    ├── run_evals.py          # Orquestrador de models locals i d’API
    ├── summarize_results.py  # Agrega els resultats en un JSON
    ├── table_template.jinja
    └── evals/
```

Els fitxers `.jinja` contenen les plantilles per generar les taules HTML de
resultats. Els directoris `evals/` contenen els fitxers JSON amb els resultats
de les avaluacions de cada model.

## Publicació de resultats

Quan s’actualitza `main`, GitHub Actions genera els JSON de resultats i els
publica a la branca `prod-data`. La web de Softcatalà llegeix aquests fitxers
per mostrar les avaluacions actualitzades. Consulta la [guia de publicació](docs/publicacio.md)
per als detalls.

## Comprovacions de desenvolupament

Des de l'arrel del repositori, pots executar les mateixes comprovacions que el workflow `.github/workflows/tests.yml`:

```bash
make format-check
make publish-check
make unit-test
make data-validation
```

Aquestes ordres comproven el format del codi, la generació dels fitxers de publicació, les proves unitàries i la coherència de les dades d'avaluació. Per aplicar el format automàticament, usa `make format`.

---

## Agraïments

Volem expressar el nostre agraïment als proveïdors dels datasets usats en l'avaluació:

- **[Projecte AINA](https://www.projecteaina.cat/)** (Barcelona Supercomputing Center) pels datasets [STS-ca](https://huggingface.co/datasets/projecte-aina/sts-ca), [VilaQuAD](https://huggingface.co/datasets/projecte-aina/vilaquad) i [CaSum](https://huggingface.co/datasets/projecte-aina/casum), que han fet possible l'avaluació de models en català.
- **[nbel](https://huggingface.co/nbel)** pel dataset [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA), corpus d'acceptabilitat lingüística en català.
- **[Google](https://ai.google/research/)** pel dataset [FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech), usat per avaluar models ASR en català.
- **[Softcatalà — Mantinc](https://github.com/Softcatala/mantinc-catalan-drift)** pel dataset que avalua si els models mantenen les respostes en català davant de prompts que poden induir un canvi de llengua.
- L'equip de **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** pel motor d'avaluació usat en benchmarks generatius.
- **[Meta AI](https://ai.meta.com/)** pel benchmark [FLORES+](https://huggingface.co/datasets/facebook/flores) de traducció automàtica.

---

## Com citar aquest treball

Si feu servir aquestes eines o els resultats en un treball, citeu-ho així (i citeu també els datasets originals dels benchmarks que referencieu):

```bibtex
@misc{softcatala_ai_eval_catalan,
  author       = {Softcatalà and Mas i Hernàndez, Jordi},
  title        = {Eines d'avaluació de models LLM, ASR i Embeddings en català},
  year         = {2026},
  howpublished = {\url{https://github.com/Softcatala/ai-eval-catalan}}
}
```
