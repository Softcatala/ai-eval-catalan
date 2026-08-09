GGUF_DIR ?= models/gguf
GGUF_MODELS ?=

render-local:
	cd llm && uv run summarize_results.py > /dev/null && cd ..
	cd asr && uv run summarize_results.py > /dev/null && cd ..
	cd embeddings && uv run summarize_results.py > /dev/null && cd ..
	uv run --with jinja2 render_tables.py
	uv run render_index_local.py

llm-download-ggufs:
	cd llm && uv run python download_ggufs.py --output-dir "$(abspath $(GGUF_DIR))" --presets-file "$(abspath $(GGUF_DIR))/presets.ini" $(if $(GGUF_MODELS),--models $(GGUF_MODELS),) $(if $(GGUF_INCLUDE_QUANTIZED_ANALYSIS),--include-quantized-analysis,)
