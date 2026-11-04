# ShadAR — Real-time Prompting On-demand Visual Augmentation in AR

Speak a prompt, and your surroundings change: this Unity app takes the Meta Quest 3's
passthrough camera streams, has an LLM write a custom shader for the prompt, finds the objects
the shader targets with an open-vocabulary detector, and renders the manipulated view back onto
the headset — in stereo, in real time.

**Example prompts:** "make all plants glow neon green", "pixelate every face",
"outline the chairs in fire".

> **ShadAR: Real-time Prompting On-demand Visual Augmentation in Augmented Reality**<br>
> Yanni Mei, Samuel Hartenstein, Jonas Wombacher, Florian Mueller, Jan Gugenheimer (TU Darmstadt)<br>
> UIST '26 · [doi:10.1145/3830398.3830534](https://doi.org/10.1145/3830398.3830534) ·
> [Project page](https://teamdarmstadt.github.io/ShadAR/)
>
> Earlier version: *ShadAR: LLM-driven shader generation to transform visual perception in Augmented Reality*, CHI EA '26 · [doi:10.1145/3772363.3799378](https://doi.org/10.1145/3772363.3799378) · [arXiv:2602.17481](https://arxiv.org/abs/2602.17481)

## How it works

```
Quest 3                                   PC (same network)
┌──────────────────────────────┐          ┌───────────────────────────────────┐
│ Passthrough cameras L + R    │  frames  │ Detection server (Server/)        │
│                              ├─────────►│ YOLOE open-vocabulary seg.        │
│                              │◄─────────┤ boxes + mask polygons + slot ids  │
│ Voice prompt                 │   UDP    └───────────────────────────────────┘
│  └ speech-to-text (OpenAI)   │
│  └ LLM writes shader ────────┼─────────► OpenAI / Anthropic API
│                              │          ┌───────────────────────────────────┐
│                              │  shader  │ Bundle compiler (QuestPackager)   │
│                              ├─────────►│ Unity batchmode → Android bundle  │
│ Stereo quad renderer         │◄─────────┤ shader.bundle                     │
└──────────────────────────────┘    WS    └───────────────────────────────────┘
```

- **Open vocabulary.** Each generated shader declares the objects it needs in a header
  comment (`// @detect-classes: person, coffee mug`). When the shader becomes active, the
  detector is reconfigured with exactly that list; a detection's label id is the position of
  its phrase in the list, which is what the shader filters on. The shader also names the
  closest COCO class per phrase (`// @detect-coco: person, cup`), which the detector searches
  for as well, since short category names are found far more reliably than descriptive phrases.
- **Stereo.** Both passthrough cameras stream at the same time onto one quad per eye, placed at
  their true 1:1 angular size. Detection runs once on the left feed; for the right eye every box
  is shifted by its disparity, using real depth from the Meta Depth API.
- **Shader history.** Every generated shader (prompt, source, detection classes, compiled
  bundle) is kept on the headset and can be re-applied from an in-headset menu with live
  previews.

## Changes since the paper

The paper describes the version used in its evaluation and workshop. This repository has been
developed further since; the paper's measurements (latency, FPS, syntax errors) were taken with
that earlier version.

| | Paper (UIST '26) | This repository |
|---|---|---|
| View | Monoscopic: one camera feed (1280×960) on a canvas at a fixed distance | Stereo: both cameras at once, one quad per eye at its true angular size (1280×1280 on Horizon OS v83+); mono remains selectable |
| Right-eye overlays | — (single view) | Detection boxes are shifted into the right eye using depth from the Meta Depth API |
| Object detection | YOLOv26 segmentation with the 80 fixed COCO classes, listed in the prompt | YOLOE open vocabulary (`yoloe-26x-seg`): each shader declares its own target phrases; the COCO detector (`yolo26-seg`) remains selectable at runtime |
| LLM | OpenAI GPT-5.2, the most capable model at the time (low and high reasoning) | OpenAI or Anthropic Claude; provider, model (from the provider's live model list) and effort are picked in the settings menu |
| Error handling | — | All network requests time out with a visible error; truncated LLM answers are reported instead of compiled |

To get close to the paper's setup, choose **Render: Mono**, **Detect: COCO fast**, **LLM:
OpenAI**, the model `gpt-5.2` and **Effort: low** or **high** in the settings menu (Y). The
system prompts have changed as well, so results will not match the paper exactly.

**Choosing a model:** we always used the most capable model available, since shader quality and
the share of shaders that compile depend on it, and the defaults follow that (`claude-fable-5-1`,
`gpt-6-astra`). Older or smaller models work in principle, but expect more compile errors and
weaker effects. Claude answers are streamed; if the model declines a request, the API retries it
on a fallback model it picks per refusal category, and a refusal that remains is shown as an
error. Claude Fable 5.1 requires an Anthropic organization with 30-day data retention.

## Repository layout

| Path | Contents |
|---|---|
| `Assets/Scripts/Modes/GenerativeShader` | Prompt → LLM → shader pipeline, shader history, detection tracking |
| `Assets/Scripts/Core` | Stereo rendering (`Rendering/`), menus, controller tooltips |
| `Assets/Scripts/Networking` | Detection client (UDP), bundle compiler client (WebSocket) |
| `Assets/Scripts/Services` | Speech-to-text, OpenAI / Claude configuration and conversation state |
| `Assets/Scripts/Logging` | Latency and frame-time benchmarks |
| `Assets/Scripts/Utils` | Logger and status line, detection frame grabber, mask atlas |
| `Assets/Scripts/Editor` | Build check that keeps secrets out of release APKs |
| `Assets/StreamingAssets/Prompts` | System prompts for shader generation |
| `Server/detection` | Detection server (FastAPI + UDP + WebSocket) |
| `docs` | Project page (GitHub Pages) |

## Requirements

- **Headset:** Meta Quest 3 or 3S with passthrough camera access (Horizon OS v74+; v83+ for the
  1280×1280 camera mode, older versions fall back to 1280×960).
- **Unity 6000.3.11f1** with the Android Build Support module, for the bundle compiler and to
  build the app yourself. Packages are pinned in `Packages/manifest.json`; OpenXR with Single
  Pass Instanced is preconfigured.
- **PC with an NVIDIA GPU** on the same network as the headset, Python 3.12, CUDA-enabled
  PyTorch (tested with torch 2.10 / CUDA 13, ultralytics 8.4.12).
- **[QuestPackager](https://github.com/teamdarmstadt/QuestPackager)**, the bundle compiler
  (see below).
- **API keys:** an OpenAI key (speech-to-text always uses the OpenAI API) and, to generate
  shaders with Claude, an Anthropic key. Keys are stored on the headset only.

## Setup

### 1. Detection server

```bash
cd Server
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python detection/server.py
```

Ports: HTTP 8000, UDP 8001 (used by the app), WebSocket 8002. Model weights and the MobileCLIP
text encoder are downloaded on first start. The default detector is `yoloe-26x-seg`;
`--yoloe-model yoloe-26n-seg` is faster but finds fewer objects.
Run `python detection/test_label_mapping.py` to check the COCO label mapping without a GPU.

### 2. Bundle compiler (QuestPackager)

Generated shaders are compiled into Android AssetBundles by a separate Unity project,
[QuestPackager](https://github.com/teamdarmstadt/QuestPackager):

```bash
git clone https://github.com/teamdarmstadt/QuestPackager
cd QuestPackager
pip install websockets
python asset_websocket_server.py
```

It listens on port 8765. Open it in the **same Unity version as this app** (6000.3.11f1) with the
**Android module** installed, otherwise bundles are rejected on the headset. QuestPackager comes
with OpenXR set to Single Pass Instanced, which the stereo shader variants need; keep that
setting, or generated shaders render broken per-eye images.

### 3. Install the app

**Prebuilt APK.** Download `ShadAR.apk` from the
[releases](https://github.com/teamdarmstadt/ShadAR/releases). Installing it needs the headset in
developer mode (enable it in the Meta Horizon app on the phone under Devices → Headset settings →
Developer mode; this requires a free Meta developer account), connected over USB with USB
debugging allowed. Then either drag the APK onto the
headset in [Meta Quest Developer Hub](https://developers.meta.com/horizon/documentation/unity/ts-mqdh/)
or run:

```bash
adb install -r ShadAR.apk
```

The app is listed under *Unknown Sources* in the headset's app library.

**Or build it yourself.** Open the project in Unity, select the Android platform and build
`Assets/Scenes/PromptedVision.unity` to the headset.

Shaders are generated with OpenAI by default. To use Claude, press **Y** in the headset and
switch **LLM: OpenAI** to **LLM: Claude** (see [Usage](#usage)).

### 4. Configure the headset

Server addresses, API keys and models are read from JSON files on the headset, so they never end
up in the APK or in git. Start the app once (grant the camera and spatial data permissions when
asked) and close it again: this creates empty templates in
`/sdcard/Android/data/com.HCILab.ShadAR/files/`. Then, with the headset connected over USB:

**1. Copy the templates to the PC.**

```bash
adb pull /sdcard/Android/data/com.HCILab.ShadAR/files/server_connections.json
adb pull /sdcard/Android/data/com.HCILab.ShadAR/files/openai_config.json
adb pull /sdcard/Android/data/com.HCILab.ShadAR/files/claude_config.json
```

**2. Fill them in.** Empty fields fall back to the defaults built into the app.

`server_connections.json` — the LAN IP of the PC running the two servers:

```json
{
    "detectionServerHost": "192.168.0.42",
    "detectionServerUdpPort": 8001,
    "bundleCompilerHost": "192.168.0.42",
    "bundleCompilerPort": 8765
}
```

`openai_config.json` — always needed, because speech-to-text uses the OpenAI API. `model` is the
model that writes the shaders (default `gpt-6-astra`), `reasoningEffort` is `None`, `Minimal`,
`Low`, `Medium` or `High`:

```json
{
    "apiKey": "sk-...",
    "organizationId": "",
    "proxyDomain": "",
    "model": "",
    "reasoningEffort": ""
}
```

`claude_config.json` — only needed when shaders are generated with Claude (**LLM: Claude** in the
settings menu). `shader.model` overrides the model (default `claude-fable-5-1`),
`shader.effort` is `low` … `max`, and `shader.fallbacks` is `default` (retry declined requests on
a fallback model) or `none`:

```json
{
    "apiKey": "sk-ant-...",
    "model": "",
    "baseUrl": "",
    "maxTokens": -1,
    "shader": {
        "model": "",
        "maxTokens": -1,
        "effort": "",
        "fallbacks": ""
    }
}
```

**3. Copy them back and restart the app.**

```bash
adb push server_connections.json openai_config.json claude_config.json /sdcard/Android/data/com.HCILab.ShadAR/files/
```

The files are read once at startup, so a later change (a new key, another server) also only
needs the push and a restart. Model and effort are easier to change in the settings menu; a
choice made there is kept across restarts and takes precedence over the files.

## Usage

### Controls

| Input | Action |
|---|---|
| Hold **B**, speak, release | Generate a shader from the prompt (takes about half a minute to two minutes, most of it compiling) |
| **X** | Open / close the shader history |
| **Y** | Open / close the settings menu |
| **Trigger** (controller ray) or **finger poke** | Click in the menus |
| Hold **both index triggers** for 2 s | Clear the LLM conversation (same as the menu button) |
| Hold a **grip** | Record detection round-trip latency while held |

The controller buttons are labelled in the headset. Without controllers, the menus also work
with hand tracking (ray pinch or finger poke); hands and controllers are not tracked at the same
time, so put the controllers down to switch.

Follow-up prompts such as "make it more blue" refer to the previous shader, until the
conversation is cleared. The strip below the view shows where a prompt is: *Listening* →
*Transcribing* → *Writing shader* → *Compiling shader* (with a seconds counter) → *Shader
applied*, and any error in its place. The full log is written to `log.txt`.

### Shader history (X)

Every generated shader is a tile with a live preview. The selected tile already previews real
detections for its own objects; clicking a tile applies it, **None** turns the effect off.

### Settings (Y)

| Entry | Effect |
|---|---|
| **Render: Stereo / Mono** | Each eye gets its own camera, or both eyes show the left camera (zero disparity) |
| **Detect: Open Vocab / COCO fast** | YOLOE with the shader's own phrases, or the faster closed-vocabulary `yolo26-seg` (80 COCO classes) |
| **LLM: OpenAI / Claude** | Provider that writes the shaders; each keeps its own conversation |
| **Model** | Models the provider's API offers for your key, newest first, fetched live when the menu opens |
| **Effort** | Effort / reasoning level of the selected provider; *Default* sends none and works with every model, lower answers faster |
| **Clear conversation** | Forget earlier prompts, so the next prompt starts from scratch |

All settings are kept across restarts.

### Data written on the headset

| Folder / file | Contents |
|---|---|
| `shader_history/shader_<timestamp>/` | `transcription.txt`, `GeneratedShader.shader`, `detection_classes.json`, `shader.bundle` |
| `benchmark_logs/` | Per prompt: timestamps of every pipeline stage, frame-time statistics after the shader is applied |
| `detection_latency/` | CSV of detection round-trip times (recorded while a grip is held) |
| `Prompts/` | Editable copies of the system prompts; replaced when the app ships a new version |

## Performance notes

- Rendering never waits for the network: detection runs on a background thread with one frame
  in flight, and the last result is kept until the next one arrives.
- Detection frames are downscaled to 512 px width before they are sent.
- The Depth API costs about 1 ms of GPU time. Disabling the `EnvironmentDepth` GameObject
  trades right-eye overlay precision for that headroom (a fixed depth is used instead).

## Citation

If you use this project, please cite the paper (also available via "Cite this repository",
generated from [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{mei2026shadar,
  author    = {Mei, Yanni and Hartenstein, Samuel and Wombacher, Jonas and Mueller, Florian and Gugenheimer, Jan},
  title     = {ShadAR: Real-time Prompting On-demand Visual Augmentation in Augmented Reality},
  booktitle = {The 39th Annual ACM Symposium on User Interface Software and Technology (UIST '26)},
  year      = {2026},
  month     = nov,
  location  = {Detroit, MI, USA},
  publisher = {ACM},
  address   = {New York, NY, USA},
  numpages  = {14},
  doi       = {10.1145/3830398.3830534}
}
```

## License

[MIT](LICENSE). Third-party content keeps its own license: Meta Passthrough Camera API sample
assets (`Assets/PassthroughCameraApiSamples/LICENSE.txt`, MIT), TextMesh Pro fonts (OFL).
