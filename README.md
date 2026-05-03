![Licence](https://img.shields.io/github/license/jenmcquade/whisper-asr-webservice.svg)

# Whisper ASR Webservice

A FastAPI-based speech recognition webservice powered by [WhisperX](https://github.com/m-bain/whisperX).
WhisperX provides fast batched inference (via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)), accurate
word-level timestamps via forced alignment, and optional speaker
diarization via [pyannote](https://github.com/pyannote/pyannote-audio) — all
in a single pipeline, so no custom diarization glue is needed here.

## Features

- WhisperX transcription with batched faster-whisper backend
- Word-level forced alignment (lazy per-language model cache)
- Optional speaker diarization (opt-in, lazy-loaded, requires `HF_TOKEN`)
- JSON output with per-word segments (each word carries `start`, `end`, `score`, and optionally `speaker`)
- 40-word speaker-aware segment grouping for UI display
- Streaming progress via Server-Sent Events (decode / transcribe / align / diarize)
- Podchaser-themed HTMX frontend on port 9001 with a playable transcript (word highlight, click-to-seek)
- Apple Silicon support natively (Mac arm64 CPU) via upstream WhisperX
- Linux GPU (CUDA 12) support via upstream WhisperX

## Environment variables

| Variable           | Default       | Description |
| ------------------ | ------------- | ----------- |
| `ASR_MODEL`        | `large`       | Default WhisperX model tier (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) or a raw checkpoint id (`medium.en`, `large-v3`, ...). The UI tier `large` resolves to `large-v3`; `turbo` resolves to `large-v3-turbo`. When a request doesn't specify a model, this is used. When the resolved language is English, tiers with an `.en` variant (`tiny`, `base`, `small`, `medium`) transparently swap to the English-only checkpoint. |
| `DEVICE`           | auto          | `cuda`, `cpu`. Auto-detects CUDA, otherwise CPU. |
| `COMPUTE_TYPE`     | auto          | `float16` on CUDA, `int8` on CPU by default. Override as needed. |
| `BATCH_SIZE`       | `16`          | WhisperX batched transcription batch size. |
| `HF_TOKEN`         | _(unset)_     | HuggingFace token. **Required for diarization.** Also read from `PYANNOTE_TOKEN` for back-compat. |
| `LOG_LEVEL`        | `INFO`        | Python logging level. |
| `UPLOAD_DIR`       | `./output` (local) / `/data/uploads` (Docker) | Where per-job upload + decoded WAV files live. Gitignored locally. |
| `JOB_TTL_SECONDS`  | `3600`        | Jobs + their on-disk files are deleted this long after they finish. |
| `KEEP_ARTEFACTS`   | `true` when `UPLOAD_DIR` is defaulted (local), `false` otherwise | When `true` the TTL sweeper still evicts finished jobs from the in-memory registry but leaves upload + WAV files on disk. Handy for poking at `./output/<job_id>.{mp3,wav}` during development. |
| `ASR_BACKEND_URL`  | `http://127.0.0.1:9000` | (Frontend only.) Backend URL the frontend proxies to. |
| `FRONTEND_DEFAULT_MODEL` | `base`    | (Frontend only.) Which model tier is pre-selected in the upload dropdown. |

### Diarization / HuggingFace terms

Diarization uses `pyannote/speaker-diarization-3.1`. Before it will load you
must:

1. Create a HuggingFace account and a read token.
2. Accept the terms for
   [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) and
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. Set `HF_TOKEN` in the container environment.

If `HF_TOKEN` is missing and a request sets `diarize=true`, the service
returns `400`.

## Quick start

Two options. Pick whichever fits — both end at the same place:

- Frontend UI: <http://localhost:9001>
- Swagger docs: <http://localhost:9000/docs>

### Option A — Docker

#### CPU / Apple Silicon

```sh
docker compose up --build
```

#### Linux + NVIDIA GPU

```sh
docker compose -f docker-compose.gpu.yml up --build
```

### Option B — Non-Docker (`up.sh`)

The repo ships with an `up.sh` helper that verifies prerequisites, installs
dependencies, and brings up both the backend (port 9000) and the frontend
(port 9001) with combined log tailing. It also seeds a `.env` from
`example.env` on first run.

```sh
./up.sh
```

What it checks and fixes:

- `uv` (fails with an install hint if missing — install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- `ffmpeg` and `ffprobe` (fails with an install hint if missing — on macOS,
  `brew install ffmpeg`; on Debian/Ubuntu, `apt-get install ffmpeg`).
- A working `python3` on PATH (warning only — `uv` can install its own
  interpreter via `uv python install 3.11` if nothing suitable is found).
- Ports 9000 and 9001 are free (override with `BACKEND_PORT` /
  `FRONTEND_PORT`).
- Runs `uv sync` unless you pass `--no-sync`.

Press `Ctrl+C` to stop both services cleanly. Logs are written to
`./output/logs/backend.log` and `./output/logs/frontend.log` for later
inspection.

## Manual (uv) — start services yourself

If you'd rather run the backend and frontend by hand (e.g. to attach a
debugger, override flags, or run them on different hosts), skip `up.sh` and
do this:

```sh
# One-time: install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh

# One-time: install deps into the project's .venv.
uv sync

# (Optional) seed your local env file.
cp example.env .env   # then edit to add HF_TOKEN if you need diarization
```

Start the backend in one shell (port 9000):

```sh
uv run uvicorn app.webservice:app --host 127.0.0.1 --port 9000
```

Start the frontend in a second shell (port 9001):

```sh
uv run uvicorn frontend.app:app --host 127.0.0.1 --port 9001
```

For a production-style backend with gunicorn + a uvicorn worker:

```sh
uv run gunicorn --bind 0.0.0.0:9000 --workers 1 --timeout 0 \
    app.webservice:app -k uvicorn.workers.UvicornWorker
```

Then open <http://localhost:9001>. By default the frontend proxies to
`http://127.0.0.1:9000`; set `ASR_BACKEND_URL` to point at a different backend.

Upstream `whisperx` is used on all platforms. On Apple Silicon make sure
`uv sync` runs with a uv-managed arm64 CPython (for example
`uv python install 3.11`) — Homebrew's `python@3.11` in `/usr/local` on some
Intel/Rosetta setups will be x86_64 and won't match this project's arm64
resolution environment.

## API

### `POST /asr`

Transcribes (and optionally aligns + diarizes) the uploaded audio and returns
JSON with per-word segments.

Multipart form:

- `audio_file` — any format supported by ffmpeg.

Query params:

| Param            | Type    | Default      | Description |
| ---------------- | ------- | ------------ | ----------- |
| `task`           | enum    | `transcribe` | `transcribe` or `translate` (to English). |
| `language`       | enum    | auto-detect  | ISO 639-1 language code. |
| `initial_prompt` | string  | —            | Optional biasing prompt. |
| `diarize`        | bool    | `false`      | Run speaker diarization. Requires `HF_TOKEN`. |
| `min_speakers`   | int     | —            | Lower bound for diarization. |
| `max_speakers`   | int     | —            | Upper bound for diarization. |

Response shape:

```json
{
  "language": "en",
  "word_segments": [
    { "word": "Hello",  "start": 0.12, "end": 0.43, "score": 0.97, "speaker": "SPEAKER_00" },
    { "word": "world.", "start": 0.50, "end": 0.91, "score": 0.93, "speaker": "SPEAKER_00" }
  ],
  "segments_grouped": [
    {
      "index": 0,
      "start": 0.12,
      "end": 14.8,
      "speaker": "SPEAKER_00",
      "text": "Hello world. …",
      "words": [ /* the word objects above */ ]
    }
  ]
}
```

`speaker` is only present when `diarize=true`. `segments_grouped` chunks the
`word_segments` into ~40-word speaker-aware segments for UI display.

### Job-based API (streaming progress)

For long jobs where you want progress feedback, use the job endpoints instead:

- `POST /asr/jobs` — same multipart form as `/asr`, but the body uses form
  fields instead of query params. Returns `{ "job_id": ..., "events_url":
  ..., "audio_url": ..., "result_url": ..., "probe_url": ... }` immediately.
- `GET /asr/jobs/{id}/events` — Server-Sent Events. Stages emitted:
  `stage`, `progress`, `probe`, `result`, `error`, `done`. Each `progress`
  event carries `stage`, `percent` (0-100), and an `eta_seconds` rolling
  estimate.
- `GET /asr/jobs/{id}/audio` — original upload with `Range` support (used by
  the frontend's `<audio>` tag for seeking).
- `GET /asr/jobs/{id}/probe` — ffprobe JSON (container / codec / sample rate
  / channels / duration / bitrate / tags).
- `GET /asr/jobs/{id}/result` — same JSON as `POST /asr`, available once the
  pipeline completes.
- `GET /asr/jobs/{id}` — small status snapshot (useful for polling clients).

Jobs + their on-disk upload and decoded WAV are deleted after
`JOB_TTL_SECONDS` (default 1h). When running locally with the defaulted
`UPLOAD_DIR=./output`, `KEEP_ARTEFACTS` is on by default — the job record
is still evicted from the in-memory registry on expiry, but the
`<job_id>.mp3` and `<job_id>.wav` files are left for inspection. Set
`KEEP_ARTEFACTS=false` locally to restore aggressive cleanup, or
`KEEP_ARTEFACTS=true` in Docker if you want to keep the volume populated.

### `POST /detect-language`

Multipart form:

- `audio_file`

Returns:

```json
{ "detected_language": "english", "language_code": "en" }
```

## Notes

- Apple Silicon runs on CPU with `int8` compute. It works for development
  but is noticeably slower than a CUDA GPU.
- Breaking change vs. v1.x: the `method`, `word_segments`, and `output` query
  params have been removed. WhisperX is now the only backend, alignment always
  runs, and responses are always JSON with per-word segments. Set
  `diarize=true` to get speakers.
- The first diarization request pays a one-time pipeline download + load
  cost. Subsequent requests reuse the cached pipeline in-process.
