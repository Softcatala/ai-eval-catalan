GGUF_DIR ?= models/gguf
GGUF_MODELS ?=

render-local:
	uv run --project llm python -m llm.summarize_results > /dev/null
	uv run --project asr python -m asr.summarize_results > /dev/null
	uv run --project embeddings python -m embeddings.summarize_results > /dev/null
	uv run --with jinja2 render_tables.py
	uv run render_index_local.py

llm-download-ggufs:
	cd llm && uv run python download_ggufs.py --output-dir "$(abspath $(GGUF_DIR))" --presets-file "$(abspath $(GGUF_DIR))/presets.ini" $(if $(GGUF_MODELS),--models $(GGUF_MODELS),) $(if $(GGUF_INCLUDE_QUANTIZED_ANALYSIS),--include-quantized-analysis,)
