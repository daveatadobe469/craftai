# CraftAI — Image Feature Handover

Three commits on `image-based-campaign` add **image input** and **image output** to the
pipeline. Everything sits behind a feature flag, so the branch runs exactly like the old
text-only pipeline until you turn images on.

| | |
|---|---|
| Branch | `image-based-campaign` |
| Base | `dev` @ `abdad52` |
| Size | 23 files, +1,111 / −110 |

---

## 1. Run it in five minutes

The pipeline works with **only a Groq key**. Images are off by default, so you need no
image or storage credentials to get a working app.

```bash
git checkout image-based-campaign

# Python 3.13 — the system 3.9 on macOS is too old
python3.13 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env and set just this one line:
#   GROQ_API_KEY=your_key

PYTHONPATH=. ./venv/bin/python main.py --skip-seed
```

API on `:8000` (docs at `/docs`), UI on `:8501`.
Drop `--skip-seed` on a first run if your knowledge base is empty.

---

## 2. Which keys you actually need

Only the first row is required. Everything else unlocks an optional capability.

| Setting | Needed for | |
|---|---|---|
| `GROQ_API_KEY` | Everything — copy generation, compliance judge, RAGAS, image understanding | **required** |
| `IMAGE_FEATURE_ENABLED` | Master switch. `false` = the old text-only pipeline; the two new nodes do nothing | default `false` |
| `CLOUDFLARE_ACCOUNT_ID`<br>`CLOUDFLARE_API_TOKEN` | Generating images on the free tier. Fast and free, but renders text inside images badly | optional |
| `GEMINI_API_KEY` | Generating images with *legible* headlines. Set `IMAGE_PROVIDER=gemini`. Pay per image (~$0.02–0.06) | optional |
| `CLOUDINARY_*` | Storing images on a CDN instead of local disk. Only needed if you deploy the UI | optional |

> **Turning images on:** set `IMAGE_FEATURE_ENABLED=true` plus one image provider's
> credentials. Storage needs nothing — it defaults to local disk, served by the API at `/media`.

---

## 3. What the branch adds

| Capability | How it works | Status |
|---|---|---|
| **Campaign → image** | After copy passes compliance, an `art_director` node writes an art-direction prompt and generates the visual. Runs once per brief, not per revision. | works |
| **Image → campaign** | Upload a reference image to `POST /brief/image`; a `vision` node describes it and the copy is written from that. You can leave the key message blank. | works |
| **Two image providers** | Cloudflare (free) or Gemini (legible in-image text). One env var switches. | works |
| **Two storage backends** | Local disk or Cloudinary CDN, behind one `save()` contract. `NONE`/`off`/empty all mean local. | works |
| **Image compliance judge** | Grades a generated image against brand guidelines and the approved copy. | **disabled** — see §5 |

Pipeline shape. Both new nodes are no-ops when the flag is off, so the graph is unchanged
in text-only mode:

```
vision → orchestrator → generator → compliance → art_director → human_gate → curator
  ↑                          ↑___________|              ↑
  image → text          revise loop              text → image
```

---

## 4. Bugs fixed along the way

- **The image-upload path never worked.** The configured vision model did not exist on the
  account and returned 404. Now `qwen/qwen3.6-27b`, the only vision-capable model Groq exposes.
- **Vision replies were truncated.** The model reasons before answering and the token budget
  was too small, so it never reached the answer.
- **The model's scratchpad leaked into the brief.** Its internal reasoning was being fed into
  the copy prompt as if it were the campaign brief.
- **Generated images vanished after approval.** The curator overwrote the draft record and
  dropped the image reference.
- **A rate limit looked like a compliance failure.** The text judge returned 0.0 when the API
  was throttled, triggering pointless rewrites that burned more quota. It now reports
  "could not evaluate" and sends the draft straight to a human.
- **RAGAS ran on every revision.** Now runs once, on the draft that actually reaches review —
  roughly a third of the previous evaluation cost.

---

## 5. Known limitations

> **Generated images get no automated compliance check.** The human reviewer is the only
> control on visual output.

Groq's free tier caps its vision model at 8,000 tokens per minute, and one image alone costs
about 2,600 — leaving too little room for the model to reason *and* answer. It produced a
verdict in only **3 of 7** trial runs, so the judge ships disabled.

- When it did answer it was accurate — scoring a garbled image **0.0**, an image with one typo
  **0.2**, and a clean control **1.0**. The approach works; the free-tier throughput does not.
- **Cloudflare renders text inside images poorly.** Expect garbled headlines. Use Gemini when
  the visual needs readable copy.
- **Cloudinary URLs are public and permanent.** Anyone with the link can view the image. Fine
  for a demo, worth knowing before a real deployment.
- **Groq has a daily token cap.** Heavy iteration can exhaust it; `RAGAS_ENABLED=false` is the
  quickest saving.

---

## 6. Gotchas when you run it

- **Config is read once at startup.** Editing `.env` needs a restart.
- **Model IDs drift.** Two of the three defaults were wrong when first written. If you see a
  404 from a provider, check the model still exists before debugging anything else.
- **A brief can sit at review for a while.** Compliance may rewrite the copy up to
  `MAX_REVISIONS` times, and each pass costs a full generation.
- **Images already stored on Cloudinary keep their CDN links** even after you switch back to
  local storage. The URL lives in that draft's record.
- **Never commit `.env` or any credential file.** `.env`, `data/`, and common key-file patterns
  are ignored — but this repo is public, so double-check before adding anything new.

---

**Verified on this branch:** text brief → campaign + image; image upload → campaign; both
image providers; both storage backends; feature flag off restoring the original pipeline.

**Not verified:** the image judge (disabled) and any deployed-environment behaviour.
