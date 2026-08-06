render-local:
	cd llm && uv run summarize_results.py > /dev/null && cd ..
	cd asr && uv run summarize_results.py > /dev/null && cd ..
	cd embeddings && uv run summarize_results.py > /dev/null && cd ..
	uv run --with jinja2 render_tables.py
	uv run render_index_local.py
