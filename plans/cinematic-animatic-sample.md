# Cinematic book-lesson sample — original art + cached Kokoro

The user approved the **cinematic 2.5D/3D illustration direction**, while
explicitly rejecting our simple Pillow-drawn Pip design. This first artifact
is a 13.97-second **motion animatic**, not finished character animation.

A real review file was rendered using the earlier project’s Kokoro narration
from GitHub [run #40047215101](https://github.com/TheBayoumi/clipper/actions/runs/40047215101).
Three individually cropped concept-storyboard illustrations were given gentle
camera movement and transitions with FFmpeg. The sample is 1080×1920 H.264/AAC;
full decode passed. No new Qwen3, Kokoro or LLM calls were required.

The actual output MP4, original generated storyboard image, exact audio,
frame contact sheet, and exact local render code live in the conversation’s
downloadable sample review package; generated image binaries were deliberately
not embedded in the Git repository before creative approval.

Reusable CLI source: `scripts/cinematic_animatic.py`. For three portrait
story images and a voice WAV, call:

```bash
python scripts/cinematic_animatic.py \
  --shot /assets/01-hook.png \
  --shot /assets/02-book.png \
  --shot /assets/03-explain.png \
  --kokoro-wav /assets/kokoro.wav \
  --source-run 40047215101 \
  --output /artifacts/atomic-habits-preview.mp4
```

The renderer does not regenerate scripts or audio. It checks media streams
and full decode, writes SHA256 evidence, marks
`publication_eligible=false`, and explicitly records no character acting.

**The next actual acceptance artifact must use individually generated,
full-resolution, consistent shots/character assets and real movement**—not
upsampled multi-panel collage art. It needs demonstrable on-screen acting and
accurate word/phoneme alignment as appropriate, audience-appropriate design,
source/right checks, and human approval of the exact MP4 SHA256.

The connected third-party video-generation wallet reported **0 credits** at
the time of this attempt. No video-generation model calls were charged and no
publishing was enabled. The existing Pip failure gate remains closed.
