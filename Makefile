GGUF_DIR ?= models/gguf
GGUF_MODELS ?=

.PHONY: render-local unit-test data-validation llm-download-ggufs

render-local:
	uv run --project llm python -m llm.summarize_results > /dev/null
	uv run --project asr python -m asr.summarize_results > /dev/null
	uv run --project embeddings python -m embeddings.summarize_results > /dev/null
	uv run --with jinja2 render_tables.py
	uv run render_index_local.py

unit-test:
	cd llm && uv run --with pytest python -m pytest tests --ignore=tests/data_validation

data-validation:
	cd llm && uv run --with pytest python -m pytest tests/data_validation

llm-download-ggufs:
	cd llm && uv run python download_ggufs.py --output-dir "$(abspath $(GGUF_DIR))" --presets-file "$(abspath $(GGUF_DIR))/presets.ini" $(if $(GGUF_MODELS),--models $(GGUF_MODELS),) $(if $(GGUF_INCLUDE_QUANTIZED_ANALYSIS),--include-quantized-analysis,)
