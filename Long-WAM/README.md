# Long-WAM · project page

Static project page for **Long-WAM: Scaling the Context of World-Action Models**.

**Live:** https://nvlabs.github.io/LongLive/Long-WAM/ (the `Long-WAM/` folder on the `page` branch of [NVlabs/LongLive](https://github.com/NVlabs/LongLive/tree/page), published by GitHub Pages). A push to `page` redeploys the whole site in about two minutes. The page has no build step and no external dependencies.

```
index.html               page content
static/css/style.css     styles (dark theme)
static/js/config.js      ← links (arXiv, code, models), arXiv id, author homepages
static/js/data.js        every number on the page, with its table / figure in the paper
static/js/main.js        film intro, charts, tables, explorers, lazy video loading
static/videos/           web-encoded videos (largest file 47 MB, total ≈ 100 MB)
static/images/           posters, GPU photos, favicon, social preview (og.jpg)
```

## Preview locally

Run this inside this folder, then open http://localhost:8765:

```bash
python3 -m http.server 8765
```

Python's built-in server can't seek inside the long videos. With the full working folder, `python3 ../project_page_tools/serve.py` serves the same page with seeking support.

## Updating links

Edit `static/js/config.js`. Empty links show as dimmed "soon" buttons, so nothing else needs editing.

- `arxiv`: the arXiv abstract link once the paper is public.
- `arxivId`: the arXiv identifier (e.g. `2610.01234`). While it is empty, the BibTeX reads "arXiv preprint" without an id.
- `code`, `models`: GitHub (`NVlabs/LongLive`, folder `Long-WAM` on `main`) and the Hugging Face collection.

## Size

The `page` branch serves several project pages, and GitHub Pages limits a published site to 1 GB. This folder is about 100 MB, almost all of it the three films. Keep new media small.

## Page behaviour

- The film (2:38) fills the screen with an aperture reveal and autoplays **muted**, because browsers block autoplay with sound. *Play with sound* restarts it with audio.
- Scrolling shrinks the film away. When the film ends, the page glides to the paper header if you haven't scrolled.
- Demo clips load and play only when they are on screen.
- Charts, tables and the two explorers read from `data.js`. Best values per column are bold, including ties.
- `prefers-reduced-motion` is respected.

## Content notes

- All numbers come from the manuscript (see comments in `data.js`).
- Real-robot and simulation clips play at the labelled speed.
- *Generated* clips are LongLive2.0-Robot video predictions from one image and one instruction.
- GPU product photos © NVIDIA.
