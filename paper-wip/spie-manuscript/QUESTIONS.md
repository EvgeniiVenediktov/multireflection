# Questions / uncertainties (SPIE manuscript)

## From the 2026-08-05 expansion to ~5 pages
- **Page count**: you asked for ~4 pages; the expanded draft lands at 5 (content is ~4.5 pages + half a page of references). Reaching 4 would require dropping Algorithm 1, the SSIM/preprocessing equations, or several references. Venue data (Applications of ML at Optics+Photonics) shows a floor of ~8 pages in practice, so 5 is still compact for the venue. Say the word to cut to 4 and which of those to sacrifice.
- **Still excluded per your earlier instructions** (confirm they stay out at the new length): the simulation feasibility study, the MLP baseline comparison and Table 1, the stability inequality, the illumination/limitations discussion, the appendices. (The correspondence footnote was restored on 2026-08-05: Evgenii Venediktov, e.i.venediktov@gmail.com.)
- The labeled top-down photo (`system_topdown_view.jpg`) has annotation arrows baked in (kinematic frame, stepper motors, laser source, camera, raspberry pi 4) which is why it was chosen over the unlabeled `topdown_labeled.png`/`sysphoto.jpg` variants; swap if you prefer another photo.


Updated 2026-08-04 after review round. The PDF is exactly 2 pages; the space is fully spoken for — any addition needs an equal cut.

## Still open
1. **Deadline**: the SPIE guidelines page states manuscripts are due **5 August 2026** — confirm this is the right conference/deadline for your submission. Conference/track metadata isn't embedded in the manuscript (SPIE assigns numbering); check the specific conference's author instructions.
2. **Title style**: "in situ" is set in roman, as you wrote it; italic *in situ* is also common — say the word to change it.
3. **Hyperlinks** are blue (template default); can switch to black for a print-first look.
4. **Copyright**: SPIE uses an online Permission to Publish agreement; nothing is required inside the manuscript.

## Decisions resolved in review (for the record)
- Authors: 4 (Venediktov, Zhong, Zhang, Ikpeazu), Pitt affiliation. Correspondence footnote **left out** (your call; SPIE treats it as optional).
- MLP comparison removed entirely; only ResNet-18 numbers reported.
- References final = 8, including the three G. Zhang papers: [1] FBG demodulation (full 9-author list pulled via DOI from Crossref), [2] Type-II FBG array (your BibTeX), [4] Zhao multipass photoacoustic analyzer. Cited in the intro's low-cost-instrumentation opener.
- To make room, **Burkhardt (RL lens alignment) was dropped** — Slor remains as the ML-alignment related work. Also previously cut: Qin, Yoder, Herriott-cell cite (multipasscell), AdamW/SGDR cites, hydrogen TDLAS. All remain in `refs.bib` for easy restore.
- Trims accepted by you: keywords 8 → 6; abstract clauses ("heavy optomechanics…", "without human intervention"); training details to "(AdamW, 80/20 split)"; drift/shock aside; compressed augmentation list.
- Figure sizes: compromise after your "enlarge, cut a reference" — Fig. 1 at 0.27\textwidth, Fig. 3 heatmap at 0.66 of its half-row (larger than the floor version, smaller than the 4-page draft). Figures 2–3 share one row.
- Content dropped per plan: simulation study, ABCD/stability math, algorithm floats, Table 1, appendices, biographies.

## Repo housekeeping
- `INDEX.md` is stale (maps a 715-line `access.tex`; the file is 506 lines since commit 69ebf6d) — refresh per AGENTS.md as a separate task.
- Compile: `docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work -e HOME=/tmp texlive/texlive:latest latexmk -pdf -interaction=nonstopmode main` (no local TeX installed); also compiles on Overleaf as-is.
