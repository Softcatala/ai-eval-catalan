# Guia per a desenvolupadors

Requereix [uv](https://docs.astral.sh/uv/). Executa totes les ordres des de
l’arrel del repositori.

## Comprovacions de desenvolupament

Des de l'arrel del repositori, pots executar les mateixes comprovacions que el workflow `.github/workflows/tests.yml`:

```bash
make format-check
make publish-check
make unit-test
make data-validation
```

Aquestes ordres comproven el format del codi, la generació dels fitxers de publicació, les proves unitàries i la coherència de les dades d'avaluació. Per aplicar el format automàticament, usa `make format`.

## Informes HTML de depuració

Els informes HTML no es publiquen ni formen part del contracte de dades de la web. Es poden generar localment per inspeccionar els resultats:

```bash
make render-local
```

Aquesta ordre genera les taules HTML i les agrupa a `index_local.html`.

[Torna al README](../README.md)
