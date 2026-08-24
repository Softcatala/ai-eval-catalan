GGUF_DIR ?= models/gguf
GGUF_MODELS ?=
PYTHON_SOURCES := render_index_local.py render_tables.py eval_common asr embeddings llm

.PHONY: render-local format format-check publish-check unit-test data-validation llm-download-ggufs

render-local:
	uv run --project llm python -m llm.summarize_results > /dev/null
	uv run --project asr python -m asr.summarize_results > /dev/null
	uv run --project embeddings python -m embeddings.summarize_results > /dev/null
	uv run --with jinja2 render_tables.py
	uv run render_index_local.py

format:
	cd llm && uv run ruff format $(addprefix ../,$(PYTHON_SOURCES))

format-check:
	cd llm && uv run ruff format --check $(addprefix ../,$(PYTHON_SOURCES))

publish-check:
	PYTHONPATH=$(CURDIR) uv run --with jinja2 python -m llm.summarize_results --json-norm /tmp/llms.json --html /tmp/llms-summary.html
	PYTHONPATH=$(CURDIR) python -m asr.summarize_results --json-out /tmp/asrs.json
	PYTHONPATH=$(CURDIR) python -m embeddings.summarize_results --json-out /tmp/embeddings.json

unit-test:
	cd llm && PYTHONPATH=.. uv run --with pytest python -m pytest tests --ignore=tests/data_validation

data-validation:
	cd llm && PYTHONPATH=.. uv run --with pytest python -m pytest tests/data_validation

llm-download-ggufs:
	cd llm && uv run python download_ggufs.py --output-dir "$(abspath $(GGUF_DIR))" --presets-file "$(abspath $(GGUF_DIR))/presets.ini" $(if $(GGUF_MODELS),--models $(GGUF_MODELS),) $(if $(GGUF_INCLUDE_QUANTIZED_ANALYSIS),--include-quantized-analysis,)
