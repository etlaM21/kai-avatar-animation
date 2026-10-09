## KIMODO OSC STREAMING - API Request Format

This service maintains a warm instance of the Kimodo-SOMA neural network in VRAM. It listens for HTTP POST requests, generates 3D motion data in a background thread, and immediately streams the raw mathematical matrices directly to Unreal Engine via Open Sound Control (OSC) over UDP.

To trigger a live generation, send an HTTP POST request to the `/generate` endpoint.

* Endpoint: `http://127.0.0.1:42069/generate`
* Headers: `Content-Type: application/json`
* Method: `POST`

### JSON Payload Body Structure

```JSON
{
  "prompt": "A person doing a backflip",
  "filename_prefix": "backflip_test",
  "fps": 30,
  "duration": 9.0
}

```

*(Note: `fps`, `duration`, `steps`, and `seed` are optional. If omitted, the server uses its configured fallback defaults).*

### Concurrency & Thread Blocking (Artistic Control)

The Python server uses an `asyncio.Lock()` around the inference and streaming loop. **While KASPAR is actively generating and streaming an animation, any incoming requests to the API will be forced to wait in a queue.** Currently, this acts as a deliberate artistic choice: it prevents the LLM (or a rapid succession of API calls) from interrupting or overwriting KASPAR while he is in the middle of a physical performance on stage. The lock ensures KASPAR finishes his current movement cleanly before the server accepts the next prompt.

#### Starting the Service

Because the network stream must cross the boundary between the WSL2 Linux virtual machine and the Windows host, the script automatically resolves the default gateway IP.

```Bash
python kimodo_service.py

```

#### Testing the Stream (Windows PowerShell)

To ping the server directly from Windows PowerShell, use the explicit `curl.exe` command to bypass the default Windows `Invoke-WebRequest` alias:

```Bash
curl.exe -X POST "http://127.0.0.1:42069/generate" \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"A person doing a backflip\", \"filename_prefix\": \"backflip_test\"}"

```

## UNREAL ENGINE - OSC Receiver Setup

Unlike previous iterations of this pipeline, **this system operates exclusively at Runtime (Play-In-Editor / PIE).** It no longer relies on Editor Utility Blueprints, Web Remote Control, or physical `.fbx` assets on disk.

The receiver listens continuously for UDP packets from the Kimodo Python server, unpacks the Quaternions, and stores them in memory for the Control Rig to apply to the MetaHuman skeleton.

### Blueprint Architecture

This logic is housed directly inside KASPAR's main Character Blueprint (e.g., `BP_Meta_Avatar`) and executes on `Event BeginPlay`.

#### 1. The OSC Server Initialization

On `BeginPlay`, the Blueprint creates an OSC Server bound to the address **`0.0.0.0`** on Port **`8000`**.
*(Note: Listening on `0.0.0.0` rather than `127.0.0.1` is strictly required to catch UDP packets originating from the WSL2 virtual network adapter).*

The server binds to an `On OSC Message Received` event, routing all incoming traffic to a custom parsing function.

#### 2. The Parsing Logic (`HandleIncomingPacket`)

The custom event fires dozens of times per frame, filtering the incoming string addresses:

* **Root Position Intercept:** If the OSC Address exactly matches `/kaspar/root_pos`, the payload is unpacked into a 3-float array (X, Y, Z), converted to a standard Unreal `Vector`, and saved to the `HipsTargetTranslation` variable.
* **Joint Rotation Intercept:** If the OSC Address starts with `/kaspar/joint/`, the string is trimmed to isolate the raw bone name (e.g., `LeftShoulder`). The 4-float payload is unpacked into a `Quat` (Quaternion: X, Y, Z, W) to prevent Gimbal Lock.
* **The Pose Map:** The bone name (String) and the Quaternion (Quat) are added to a Dictionary variable named `LivePoseMap`. This Map acts as a constantly updating snapshot of the entire virtual skeleton, refreshing at exactly 30 FPS.

#### Testing the Unreal Engine Node Independently

1. Ensure KASPAR's Blueprint is in the level and the OSC Server logic is wired to `Print String` diagnostic nodes.
2. Hit **Play** in the Unreal Engine editor.
3. Trigger the Python Kimodo script via terminal.
4. You will immediately see the incoming translation vectors and bone names rapidly populating the top-left corner of the active viewport.