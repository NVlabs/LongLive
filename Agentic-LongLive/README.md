# Agentic LongLive · project page

Static project page for **Agentic LongLive: Asynchronous Agentic Memory for Long Video Generation**.

**Live:** https://nvlabs.github.io/LongLive/Agentic-LongLive/ (the `Agentic-LongLive/` folder on the `page` branch of
[NVlabs/LongLive](https://github.com/NVlabs/LongLive/tree/page), published by GitHub Pages). Code:
[`Agentic-LongLive/`](https://github.com/NVlabs/LongLive/tree/main/Agentic-LongLive) on `main`. The page has no build step.

```
index.html              page content
assets/style.css        styles
assets/app.js           hero animation, efficiency and system figures, the long-stories wall and its agent view
assets/data.js          page data: stories, shot times, memory pictures, figure numbers, video URLs
assets/robot_long.*     section 2.2, the robot examples (data, code, styles)
data/agent/<id>.json    agent logs for the 2.1 agent view, loaded when a video is opened
media/                  posters, memory pictures and figure images (WebP / JPEG)
```

## Videos

GitHub Pages limits a published site to 1 GB, and this branch already serves several project pages. The 35 videos
(617 MB) are therefore not in the branch. They are served from the Hugging Face dataset
[`Shaoteng62/Agentic-LongLive-Page`](https://huggingface.co/datasets/Shaoteng62/Agentic-LongLive-Page)
under the same paths (`media/...mp4`). `assets/data.js` and `assets/robot_long.js` hold their full URLs. To serve them
from somewhere else, replace the prefix
`https://huggingface.co/datasets/Shaoteng62/Agentic-LongLive-Page/resolve/main/` in these two files.

## Preview locally

Run this inside this folder, then open http://localhost:8765 (the videos stream from Hugging Face):

```bash
python3 -m http.server 8765
```

## Updating links

The arXiv and 🤗 Models buttons in `index.html` carry `data-pending` and show a "soon" badge. When a link is ready, set its
`href` and remove `data-pending`. Add the arXiv id to the BibTeX block (`#bibtex`) at the same time.

## Size

This folder is about 23 MB: some 2,300 small memory pictures and agent logs. Keep new videos out of the branch.

## Page behaviour

- The hero animates the system at work. `prefers-reduced-motion` is respected.
- Videos and animations run only while they are on screen.
- 2.1 Long stories: three systems (FastH3-Omni, AR, Offline). Hovering a video (tapping it on a touch screen) opens the agent
  view: prompts, the memory in the generator, and the agent's log replayed in step with the video.
- 2.2 Robotics: the video model alone next to Agentic LongLive. The robot videos play at 2x by default, and a viewer's
  own speed choice is remembered in the browser.
- All videos on the page are generated. The robot videos start from a single image from RoVid-X.
