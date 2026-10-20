# Project homepage

The static project homepage is published at https://yihaohu0118.github.io/ChronoReplay/.
Its source is in `site/`; the GitHub Pages workflow deploys that directory on pushes to `main`.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --directory site
```

Visit `http://localhost:8000`. There is no package installation or build step.

## Update the page

- Edit `site/index.html` for prose, authors, links, and the default result table.
- Edit `site/app.js` for the three interactive result comparisons.
- Edit `site/style.css` for layout and appearance.
- Replace `site/assets/ChronoReplay.pdf` with a verified manuscript PDF.
- Keep paper figures in `site/assets/` and retain descriptive alternative text.

The default table remains available without JavaScript. Result switching,
figure enlargement, and clipboard actions use the small local script.

All benchmark values are author-reported manuscript values, not results of a
new benchmark run. Keep matched-access training comparisons separate from
adaptive-access system results. Do not add model or dataset download buttons
until the corresponding artifacts have been released.

The website references OneStreamer as presentation inspiration. Its code,
media, and experimental claims are not copied into this project.
