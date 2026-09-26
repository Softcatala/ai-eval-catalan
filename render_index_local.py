"""
Assemble index_local.html by inlining the four rendered HTML table fragments.
Produces a self-contained file that works when opened directly via file://.
"""

from pathlib import Path

FRAGMENTS = [
    ("LLM — Taula", "llm/llms_table.html"),
    ("LLM — Taula Quantitzada", "llm/llms_quantized_table.html"),
    ("ASR — Taula", "asr/asrs_table.html"),
    ("Embeddings — Taula", "embeddings/embeddings_table.html"),
]

STYLE = """
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f4f4f4;
      margin: 0;
      padding: 24px;
      color: #222;
    }
    h1 { font-size: 20px; font-weight: 700; margin: 0 0 24px; color: #1a1a1a; }
    h2 { font-size: 16px; font-weight: 700; margin: 0 0 16px; color: #1a1a1a; }
    .section {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 20px 24px;
      margin-bottom: 20px;
      overflow-x: auto;
    }
"""

sections = ""
for title, path in FRAGMENTS:
    content = Path(path).read_text(encoding="utf-8")
    sections += f'  <div class="section"><h2>{title}</h2>\n{content}\n  </div>\n\n'

html = f"""<!DOCTYPE html>
<html lang="ca">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Eval Catalan</title>
  <style>{STYLE}  </style>
</head>
<body>
  <h1>AI Eval Catalan</h1>

{sections}</body>
</html>
"""

Path("index_local.html").write_text(html, encoding="utf-8")
print("Saved to index_local.html")
