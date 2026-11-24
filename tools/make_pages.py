"""Build the GitHub Pages copy of the reader page (johanwind.github.io/pronounce, /pronounce/nb for
Norwegian, /pronounce/el for Greek): the same page, pointed at that language's hosted server.
Output: dist/pages/index.html, dist/pages/<lang>/index.html. (The page's language is set here too, so
it shows the right labels before the server has answered.)

    .venv/bin/python tools/make_pages.py [en|nb|el] [API_BASE]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APIS = {  # AWS Lambda function URLs (eu-north-1): functions pronounce, pronounce-nb, pronounce-el
    "en": "https://cf3vejnu7xc46ypdcucfceoxja0peoba.lambda-url.eu-north-1.on.aws",
    "nb": "https://hps4xn34gpmc3kdium26il6zwy0xphaf.lambda-url.eu-north-1.on.aws",
    "el": "https://3kn7gyfkkgpmu7tn2md2ovoiwy0kqpyn.lambda-url.eu-north-1.on.aws",
}
lang = sys.argv[1] if len(sys.argv) > 1 else "en"
api = sys.argv[2] if len(sys.argv) > 2 else APIS[lang]
assert api, f"no server URL for {lang}: pass it as the second argument"
out = ROOT / "dist" / "pages" / ("" if lang == "en" else lang)
out.mkdir(parents=True, exist_ok=True)
html = (ROOT / "static" / "reader.html").read_text()
html = html.replace("<script>", f'<script>window.API_BASE = "{api.rstrip("/")}"; window.PAGE_LANG = "{lang}";</script>\n<script>', 1)
(out / "index.html").write_text(html)
print(out / "index.html", "->", api)
