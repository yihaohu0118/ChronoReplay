# ChronoReplay cover artwork

The README uses `cover.gif`; the project homepage uses `cover.mp4` with
`cover.png` as its still poster. The homepage has a pause control, pauses
off screen, and starts with a still image when reduced motion is requested.

## Exports

- `cover.png`: 1600 × 820 static poster.
- `cover.mp4`: 1600 × 820 H.264 video, 12 seconds, 12 fps, no audio.
- `cover.gif`: 1200 × 615 looping animation, 12 seconds.
- `sculpture.png`: original transparent project illustration.

## Scientific content

The loop illustrates chronological observation, replay at causal boundaries, and family-balanced learning with categorical, metric, and temporal source-native targets. The room sketches are conceptual diagrams, not a benchmark trajectory.

The footer numbers are author-reported results from the linked manuscript.
The generated sculpture is a visual metaphor for the project.

## Editable source

`render.js` draws all typography, diagrams and motion with Canvas 2D.
`render.html` loads the self-hosted Manrope and Space Grotesk fonts; their
licenses are in `../fonts/`. `render.py` exports the same deterministic frames.

```bash
python -m pip install playwright imageio-ffmpeg
python -m playwright install chromium
python site/assets/brand/render.py
# Static poster only:
python site/assets/brand/render.py --preview
```

The sculpture was created with the built-in image_gen tool. Its exact prompt
is saved in `imagegen-prompt.json`; the original generated PNG is preserved.
The sister project uses the same typography and composition with its own color
and sculpture. The storytelling cover format was inspired by
[Agent-Reflex](https://github.com/RUC-NLPIR/Agent-Reflex/tree/main/assets/cover).
