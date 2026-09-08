#!/bin/bash
# Rebuild dark.html (dark-mode HTML version of the manuscript) from main.tex.
# Run from spie-manuscript/ after compiling the PDF (needs main.bbl).
set -e
cd "$(dirname "$0")"
mkdir -p htmlbuild
cp main.tex refs.bib spiebib.bst spie.cls main.bbl htmlbuild/
cp -r figures htmlbuild/
# make4ht exits non-zero on harmless spie.cls \maketitle warnings; check output instead
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work -e HOME=/tmp \
  texlive/texlive:latest sh -c "cd htmlbuild && make4ht -u main.tex 'mathml' >make4ht.log 2>&1" || true
[ htmlbuild/main.html -nt htmlbuild/main.tex ] || { echo "make4ht produced no fresh HTML"; exit 1; }
python3 - <<'EOF'
import base64, re, pathlib
html = pathlib.Path('htmlbuild/main.html').read_text()
css  = pathlib.Path('htmlbuild/main.css').read_text()
def embed(m):
    p = pathlib.Path('htmlbuild') / m.group(1)
    mime = 'image/png' if p.suffix == '.png' else 'image/jpeg'
    return f"src='data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}'"
html = re.sub(r"src='([^']+)'", embed, html)
dark = """
html { background:#0e1116; }
body { background:#0e1116; color:#d6dae0; max-width:46rem; margin:0 auto;
       padding:1.5rem 1.2rem 4rem; line-height:1.65; font-size:1.05rem; }
h1,h2,h3,h4,.likesectionHead,.sectionHead { color:#f2f4f7; }
a { color:#82b4ff; } a:visited { color:#b294ff; }
.author, .cmr-9, .cmr-12 { color:#c2c8d0; }
img { max-width:100%; height:auto; background:#ffffff; padding:6px;
      border-radius:8px; filter:brightness(.92); }
hr { border-color:#2a2f37; }
table { color:#d6dae0; }
::selection { background:#33507a; }
@media (max-width:600px){ body{ font-size:1rem; padding:1rem .8rem 3rem; } }
"""
extra = ("<meta name='viewport' content='width=device-width, initial-scale=1'/>"
         f"<style>{css}\n{dark}</style>")
html = re.sub(r"<link[^>]*main\.css[^>]*/>", '', html)
html = html.replace('</head>', extra + '</head>')
pathlib.Path('dark.html').write_text(html)
print('dark.html rebuilt')
EOF
