# Eines d'avaluació de models LLM, ASR, embeddings i de decisió

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM), de reconeixement automàtic de la parla (ASR), d'embeddings i de decisió (JEV), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/ia-local/models-en-catala/

## Vols que avaluem un model?

[Obre un issue](https://github.com/Softcatala/ai-eval-catalan/issues/new) indicant l'URL de Hugging Face del model, el tipus (LLM, ASR, embeddings o JEV) i per què és rellevant per al català.

## Estructura del projecte

```
ai-eval-catalan/
├── docs/                     # Guies d’avaluació i desenvolupament
├── eval_common/              # Codi compartit pels generadors de JSON
├── llm/                      # Avaluació de models LLM
├── asr/                      # Avaluació de models ASR
├── embeddings/              # Avaluació de models d’embeddings
└── jev/                      # Avaluació de models de decisió
```

Els fitxers `.jinja` contenen les plantilles per generar les taules HTML de
resultats.

## Documentació

| Guia | Contingut |
|------|-----------|
| [LLM](docs/llm.md) | Benchmarks, puntuació CLAM, instal·lació i avaluació de models locals i d'API |
| [ASR](docs/asr.md) | Preparació de FLEURS i OpenSLR-69 i avaluació de reconeixement de la parla |
| [Embeddings](docs/embeddings.md) | Benchmarks i avaluació de models locals i d'API |
| [JEV](docs/jev.md) | Avaluació de models de decisió sobre MASSIVE i TE-ca |
| [Desenvolupament](docs/desenvolupament.md) | Comprovacions de desenvolupament i informes HTML de depuració |

## Publicació automàtica de resultats (CI/CD)

Quan s’actualitza `main`, GitHub Actions agrega les avaluacions de LLM, ASR,
embeddings i JEV i genera els JSON de resultats i recomanacions. Aquests fitxers
es publiquen automàticament a la branca `prod-data`, que la web de
[Softcatalà](https://www.softcatala.org/ia-local/models-en-catala/) llegeix per
mostrar els resultats actualitzats. Els informes HTML són només per a consulta local.

### Recomanacions de models automatitzada

Les recomanacions es calculen a partir de les avaluacions completes dels models
que es poden executar localment. S’exclouen els serveis de núvol i les proves
que comparen quantitzacions fora del benchmark principal.

**LLM:** per a ordinadors amb 8, 16 i 32 GB de RAM, es consideren els models que
caben deixant un 25% de memòria lliure per al sistema i l’execució. La qualitat
es mesura amb CLAM, la puntuació que combina els benchmarks de català.
Entre els models que queden a menys de 2 punts del millor, es recomana el que
consumeix menys memòria. També es mostra una alternativa amb la millor puntuació
entre la resta de models que caben. S’assumeix que s’executa un sol model alhora;
el consum real pot variar segons la llargada del context.

**ASR:** s’ofereixen dues recomanacions, sense separar-les per memòria disponible:

- **Màxima precisió:** el model amb menys errors de transcripció en el conjunt
  de FLEURS i OpenSLR-69, donant més pes al benchmark amb més mostres.
- **Precisió i velocitat:** el model amb menys errors entre els que són almenys
  el doble de ràpids a FLEURS que el de màxima precisió i tenen com a màxim
  2 punts percentuals més d’error. Per exemple, si el millor té un 5% d’error,
  aquesta opció ha de tenir com a màxim un 7%. Si cap model compleix totes dues
  condicions, no s’ofereix aquesta segona recomanació.

Els marges de 2 punts són criteris pràctics de selecció, no marges estadístics d’error.

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
