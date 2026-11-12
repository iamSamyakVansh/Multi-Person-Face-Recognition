# Multi-Person Face Recognition

A Flask web application that performs real-time, multi-person face
recognition on a video stream — either an uploaded video file or a live IP
camera feed — using `dlib`, `face_recognition`, and OpenCV. Recognized faces
are logged with a timestamp and confidence score, and the full session can be
emailed out as a CSV summary report via the Gmail API.

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup (Windows)](#setup-windows)
- [Running the app](#running-the-app)
- [Using the web UI](#using-the-web-ui)
- [Configuration reference](#configuration-reference)
- [Training on a new dataset](#training-on-a-new-dataset-addingupdating-known-faces)
- [Gmail API setup (email reports)](#gmail-api-setup-email-reports)
- [Troubleshooting](#troubleshooting)

## Features

- **Live face recognition** on an uploaded video file or an IP camera stream,
  shown side-by-side with the raw feed in the browser.
- **Face matching against a known-faces database** built from your own
  labelled photos — unrecognized faces are flagged as `Unknown face`.
- **CSV detection report** (name/ID, time, date, camera, match confidence)
  generated on demand or on a daily schedule.
- **Email delivery of reports** via the Gmail API, with the report attached
  as a CSV.
- **Pluggable face detection models** — `hog` (CPU) or `cnn` (GPU) — and a
  configurable match tolerance.

## Project structure

```
Multi_Person_Face_Recognition/
├── src/                        # Application source
│   ├── app.py                   # Flask app entry point & routes
│   ├── dlib_face_recognition.py # Core face detection/matching logic
│   ├── dlib_face_embeddings.py  # Builds the known-faces encoding file
│   ├── database_pandas.py       # In-memory detections table + CSV export
│   ├── send_mail.py             # Gmail API email delivery
│   ├── schedule_send_report.py  # Daily scheduled report email
│   ├── custom_logging.py        # Shared logger configuration
│   ├── locks.py                 # Thread lock used by batch GPU inference
│   ├── parameters.py            # All configurable paths/settings
│   ├── util/                    # Helper scripts (folder listing, video utils)
│   ├── templates/                # Flask/Jinja HTML templates
│   └── static/                   # CSS, images, demo video
├── dataset/                     # Labelled face images, one folder per person
├── data/                        # Precomputed face encodings (.pkl) + demo video
├── reports/                     # Generated CSV detection reports (runtime output)
├── logs/                        # Runtime log files (runtime output)
├── docs/                        # Reference PDFs (dataset & inferencing guide)
├── requirements.txt
├── .gitignore
└── venv_win/                     # Windows virtual environment (not committed)
```

## Requirements

- Windows 10/11
- Python 3.12 (no CMake/Visual Studio Build Tools required — see below)
- A webcam-accessible IP camera, or a local video file, to run recognition on
- (Optional, for email reports) A Google account and a Google Cloud project
  with the Gmail API enabled

## Setup (Windows)

`dlib` has no official prebuilt wheel for recent Python versions on Windows,
and building it from source requires CMake + Visual Studio Build Tools. To
avoid that entirely, this project uses the prebuilt **`dlib-bin`** package,
which installs the same `dlib` module without compiling anything.

```powershell
# 1. Create and activate a virtual environment
python -m venv venv_win
.\venv_win\Scripts\Activate.ps1

# 2. Install dlib via the prebuilt wheel (do this BEFORE requirements.txt)
pip install dlib-bin

# 3. Install the rest of the dependencies
pip install -r requirements.txt
pip install "setuptools<81"   # face_recognition_models needs pkg_resources
```

> **Note:** `requirements.txt` intentionally leaves `opencv-python` unpinned
> to a bleeding-edge version; if `pip` resolves a very new `5.x` build and you
> see the app crash with a native `fctx->async_lock failed` error from
> `libavcodec`, pin it back to a known-stable release:
> `pip install "opencv-python==4.10.0.84"`.

## Running the app

Run from the **project root** (`Multi_Person_Face_Recognition/`), not from
`src/` — several paths in `src/parameters.py` (e.g. `reports/`, `data/`,
`dataset/`) are resolved relative to the current working directory, which is
expected to be the project root.

```powershell
.\venv_win\Scripts\python.exe src\app.py
```

The app serves at **http://127.0.0.1:5000/**.

## Using the web UI

1. Open `http://127.0.0.1:5000/` in a browser. The page shows a **raw** video
   panel and a **face-recognition-processed** panel side by side.
2. **Run on the bundled demo video** — click the video inference button; the
   app streams `src/static/video/vid.mp4` through both panels.
3. **Upload your own video** — use the upload form (MP4, MOV, AVI or WMV, up
   to 200MB); it's saved to `src/static/video/vid.mp4` and used for the next
   inference run.
4. **Use a live IP camera** instead — paste its stream URL (e.g.
   `http://192.168.x.x:4747/video` for a phone running an IP-camera app) into
   the live-stream field and submit.
5. **Download the detection report** — click the download button to get a CSV
   of everyone recognized so far (`name/id, time, date, cam name, cam_ip,
   face_distance`).
6. **Email the report** — check the email alert box, enter a recipient
   address, and submit. This requires the Gmail API to be configured first —
   see [Gmail API setup](#gmail-api-setup-email-reports) below.

## Configuration reference

All tunable paths and settings live in
[`src/parameters.py`](src/parameters.py):

| Parameter | Purpose |
|---|---|
| `DATASET_PATH` | Folder of labelled face images used to build encodings |
| `DLIB_FACE_ENCODING_PATH` | Where the generated `.pkl` encodings file is read from/written to |
| `REPORT_PATH` | Where the CSV detection report is written |
| `FILES_PATH` / `VIDEO_PATH` | Location of the bundled demo video |
| `VIDEO_UPLOAD_PATH` | Where an uploaded video is saved before inference |
| `LOG_FILE_PATH` | Application log file |
| `FACE_MATCHING_TOLERANCE` | Max face distance to count as a match (lower = stricter) |
| `FACE_RECOGNITION_MODEL` | `hog` (CPU) or `cnn` (GPU) face detector |
| `NUMBER_OF_TIMES_TO_UPSAMPLE` | Upsampling passes when detecting faces (higher finds smaller/farther faces, slower) |
| `FRAME_HEIGHT` / `FRAME_WIDTH` | Frame size frames are resized to before processing |
| `BATCH_SIZE` | Batch size for GPU-based batch inference |
| `IP_CAMS` | Named IP camera URLs |
| `EMAIL_SENDER` / `EMAIL_RECEIVER` / `EMAIL_SUBJECT` / `EMAIL_BODY` | Defaults used for report emails |
| `EMAIL_SEND_TIME` | Time of day the scheduled report email is sent (see `schedule_send_report.py`) |

## Training on a new dataset (adding/updating known faces)

The app doesn't train a model in the machine-learning sense — it builds a
lookup table of face *encodings* (128-dimensional vectors) per known person,
computed once ahead of time from your photos and loaded at runtime for
matching.

### 1. Organize your images

Under `dataset/`, create one folder per person, named with their ID or full
name. Each folder should contain several clear, front-facing photos of that
person (varying angles/lighting improves accuracy; 5–10 images per person is
typically enough):

```
dataset/
└── <Your_Dataset_Name>/
    ├── <PersonID_1>/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── ...
    ├── <PersonID_2>/
    │   └── ...
    └── ...
```

The folder name becomes the label displayed when that person is recognized,
so name it exactly as you want it to appear.

**Tips for good recognition accuracy:**
- Use well-lit, front-facing photos without heavy occlusion (sunglasses,
  masks, extreme angles).
- Include a few photos with slightly different angles/expressions per
  person — this makes matching more robust to the actual camera/video
  conditions.
- Avoid group photos — only one person's face should be identifiable per
  image, since the embedding script takes the first detected face in each
  image.
- More distinct people in the dataset increases the chance of two people
  being confused if `FACE_MATCHING_TOLERANCE` is too high — keep it as low as
  you can while still reliably matching your own data.

### 2. Point the app at your dataset

In [`src/parameters.py`](src/parameters.py), update:

```python
DATASET_PATH = 'dataset/<Your_Dataset_Name>/'
DLIB_FACE_ENCODING_PATH = 'data/<your_dataset>_face_encodings.pkl'  # output file
```

### 3. Generate the face encodings

Run the embedding script from the project root. It walks every image under
`DATASET_PATH`, computes one face encoding per image via `face_recognition`,
and writes them all to `DLIB_FACE_ENCODING_PATH`:

```powershell
.\venv_win\Scripts\python.exe src\dlib_face_embeddings.py
```

This can take a while for large datasets — it prints progress as
`[INFO] processing image i/N`. When it finishes, you'll see a new/updated
`.pkl` file at the path you configured.

### 4. Restart the app

Restart `src/app.py` so it reloads the new `.pkl` file — encodings are only
read once, at startup (`src/dlib_face_recognition.py`).

### Notes

- **Adding a person later:** add a new subfolder with their photos and
  re-run step 3 — it regenerates the *entire* encodings file from everything
  currently under `DATASET_PATH`, so make sure existing people's images are
  still present, or you'll lose their encodings.
- **Tuning match strictness:** `FACE_MATCHING_TOLERANCE` in `parameters.py`
  controls how close a detected face's encoding must be to a known encoding
  to count as a match (face distance, not a percentage) — lower values mean
  stricter matching (fewer false positives, but more missed/"Unknown"
  matches for genuine people); higher values are more lenient but risk
  misidentifying one known person as another.
- **The demo video won't match your dataset:** `data/files/vid.mp4` is a
  generic sample clip. It will correctly detect faces in it, but since those
  people aren't in your dataset, they'll show as `Unknown face` — that's
  expected, not a bug. To see real matches, run inference on a video/stream
  that actually contains the people in your dataset.

## Gmail API setup (email reports)

Email delivery uses the **Gmail API** with OAuth 2.0 (not SMTP/app
passwords), via `src/send_mail.py`. By default it is **not configured**, and
clicking "send email report" in the UI will log an error rather than send
anything, until you complete this setup.

### 1. Create a Google Cloud project and enable the Gmail API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Go to **APIs & Services → Library**, search for **Gmail API**, and click
   **Enable**.

### 2. Configure the OAuth consent screen

1. Go to **APIs & Services → OAuth consent screen**.
2. Choose **External** (or **Internal** if you're on a Google Workspace
   org) and fill in the required app details.
3. Under **Scopes**, you don't need to add anything manually — the app
   requests `https://www.googleapis.com/auth/gmail.send` at runtime.
4. If the app is in "Testing" status, add the Gmail account you intend to
   send from as a **test user**, or the OAuth flow will be rejected.

### 3. Create OAuth client credentials

1. Go to **APIs & Services → Credentials → Create Credentials → OAuth
   client ID**.
2. Application type: **Desktop app**.
3. Download the resulting JSON file.
4. Save it as **`src/client_secrets.json`** (this exact filename/path is
   hardcoded in `send_mail.py`).

> **Keep this file private.** It's excluded via `.gitignore`-style
> conventions — never commit `client_secrets.json` or `token.json` to
> version control.

### 4. Match the sender account

The Gmail account you authorize in step 5 below **must** be the one you
intend to send as. Set it in [`src/parameters.py`](src/parameters.py):

```python
EMAIL_SENDER = 'your-sending-account@gmail.com'
```

### 5. Authorize once to generate `token.json`

The first time Gmail API authentication runs, it opens a browser window for
you to log in and grant the `gmail.send` permission, then saves the
resulting token to `src/token.json` for reuse. Because the Flask email route
fires this in a background thread (where a browser popup won't surface
properly), **run the one-time authorization manually first**, from the
project root:

```powershell
.\venv_win\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from send_mail import aunthentication; aunthentication()"
```

A browser window will open — log in with the Gmail account from step 4 and
grant access. Once it completes, `src/token.json` will exist and the "send
email report" button in the web UI will work normally. The token is
refreshed automatically on expiry as long as `token.json` is present.

### 6. (Optional) Scheduled daily reports

`src/schedule_send_report.py` can send the report automatically at a fixed
time each day (`EMAIL_SEND_TIME` in `parameters.py`, default `18:00`). Run it
as a separate long-lived process alongside the Flask app:

```powershell
.\venv_win\Scripts\python.exe src\schedule_send_report.py
```

### If something goes wrong

- **`FileNotFoundError: src/client_secrets.json`** — you haven't completed
  step 3 above, or the file isn't at the exact path `src/client_secrets.json`
  relative to the project root.
- **OAuth consent screen error / access blocked** — your Google Cloud project
  is still in "Testing" and the sending account isn't listed as a test user
  (step 2.4), or the consent screen configuration is incomplete.
- **Emails silently "succeed" in the UI but never arrive** — the Flask route
  reports success immediately because the send happens in a background
  thread; check `logs/multicam_server.log` for the actual outcome (errors are
  logged there, not shown in the browser).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `dlib` fails to build from source | Use the prebuilt `dlib-bin` wheel instead (see [Setup](#setup-windows)) — don't `pip install dlib` directly on Windows unless you have CMake + Visual Studio Build Tools. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Install `setuptools<81` — newer `setuptools` dropped `pkg_resources`, which `face_recognition_models` still depends on. |
| App crashes with `fctx->async_lock failed` from `libavcodec` during video streaming | Pin `opencv-python` to a known-stable release, e.g. `pip install "opencv-python==4.10.0.84"`. The app also forces single-threaded FFmpeg decoding (`OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;1`, set in `app.py`) to avoid a native crash when the raw and processed video streams read the shared video capture concurrently. |
| Everyone shows as `Unknown face` | Either your dataset's encodings don't match anyone in the video (expected if using the bundled demo video), or `FACE_MATCHING_TOLERANCE` is set too low — see [Training on a new dataset](#training-on-a-new-dataset-addingupdating-known-faces). |
| Email report never arrives | See [Gmail API setup](#gmail-api-setup-email-reports) — most commonly a missing `src/client_secrets.json` or a Testing-mode OAuth consent screen without the sender added as a test user. |
| `'cp' is not recognized...` on startup | Harmless on Windows — `app.py` runs a Linux `cp` command to seed the demo video on startup; if `src/static/video/vid.mp4` already exists, this failing is not a problem. |
