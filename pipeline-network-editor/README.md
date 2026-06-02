## KIMODO - API Request Format

To trigger a generation, send an HTTP POST request to the /generate endpoint.

- Endpoint: http://127.0.0.1:42069/generate
- Headers: Content-Type: application/json

### JSON Payload Body Structure
```JSON
{
  "prompt": "A person frantically running around",
  "filename_prefix": "running_franctic"
}
```
#### Start the Kimodo script:
```Bash
python kimodo_service.py
```

#### Open a separate terminal and test using curl:
```Bash
curl -X POST "http://127.0.0.1:42069/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A person frantically running around", "filename_prefix": "running_franctic"}'
```

## BLENDER - API Request Format

This service runs Blender in the background to ingest ``.bvh`` files from Kimodo, parent them to an Unreal-compatible dummy mesh, and export clean ``.fbx`` files. It listens for tasks via ZeroMQ and automatically pings Unreal Engine upon completion.

To trigger a conversion, send a raw string message containing the absolute file path to the ZeroMQ ``PULL`` socket.

- Protocol: ZeroMQ (ZMQ_PUSH to ZMQ_PULL)
- Address: tcp://127.0.0.1:42070
- Payload Format: Raw UTF-8 String (Absolute file path)

### String Payload Example
```Plaintext
K:\project_kaspar\modules\kai-avatar-animation\kimodo\kimodo-gen\running_franctic.bvh
```

See example usage in [./tests/blender_testscript.py](./tests/blender_testscript.py)

#### Outgoing JSON Payload Body
```JSON
{
  "objectPath": "/Game/Scripts/ImportListener.Default__ImportListener_C",
  "functionName": "ImportAndRetargetFBX",
  "parameters": {
    "FilePath": "K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert/running_franctic.fbx"
  }
}
```

#### Starting the Service
Launch the listener using Blender's headless background flag (``-b``) and the python execution flag (``-P``):
```Bash
blender -b -P blender_service.py
```


## UNREAL ENGINE - API Request Format

This service runs inside the Unreal Engine Editor using the Web Remote Control plugin. It listens for completed `.fbx` paths from Blender, auto-imports the animation, retargets it to the MetaHuman skeleton, and triggers instant playback on the Avatar in the active Play-In-Editor (PIE) session.

To trigger an import and playback sequence, send an HTTP POST request to the Web Remote Control endpoint.

* Endpoint: [http://127.0.0.1:30010/remote/object/call](http://127.0.0.1:30010/remote/object/call)
* Headers: Content-Type: application/json

### JSON Payload Body Structure

```JSON
{
  "objectPath": "/Game/Scripts/ImportListener.Default__ImportListener_C",
  "functionName": "ImportAndRetargetFBX",
  "parameters": {
    "FilePath": "K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert/running_franctic.fbx"
  }
}
```

## UNREAL ENGINE - API Request Format

This service runs inside the Unreal Engine Editor using the Web Remote Control plugin. It listens for completed `.fbx` paths from Blender, auto-imports the animation, retargets it to the MetaHuman skeleton, updates the level instance properties, and triggers viewport playback.

To trigger an import and playback sequence, send an HTTP PUT request to the Web Remote Control endpoint.

* Endpoint: http://127.0.0.1:30010/remote/object/call
* Headers: Content-Type: application/json
* Method: PUT

### JSON Payload Body Structure

``JSON
{
  "objectPath": "/Game/Scripts/ImportListener.Default__ImportListener_C",
  "functionName": "ImportAndRetargetFBX",
  "parameters": {
    "FilePath": "K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert/running_franctic.fbx"
  }
} ```

### Unreal Engine Setup

This node requires two components inside the Unreal Engine project: an Editor Utility Blueprint to catch the incoming HTTP request and a Python script to execute the asset operations.

#### 1. Blueprint Listener

Create an **Editor Utility Blueprint** (Parent Class: `EditorUtilityObject`) named `ImportListener` in `/Game/Scripts/`.

Create a public function named `ImportAndRetargetFBX` with a single String input named `FilePath`. Add an **Execute Python Command** node with the following string to pass the data to the Python worker (ensure no `py` prefix is included):

```Python
import ue_import_pipeline
import importlib
importlib.reload(ue_import_pipeline)
ue_import_pipeline.process_single_fbx('{FilePath}')

```

#### 2. Python Pipeline Worker

Save the ingestion logic as `ue_import_pipeline.py` directly inside the project's `Content/Python/` directory so Unreal automatically registers it.

### Execution & Viewport Playback Integration

Because Unreal Engine completely locks asset generation and database serialization during live simulation, **this pipeline must be executed while the Editor is in Edit Mode (Not in PIE/Play mode).**

To view the playback instantly:

1. Ensure your Avatar Blueprint (`BP_Meta_Avatar`) is placed in your current Editor Level.
2. Enable viewport updates by clicking the viewport options menu and checking **Realtime** (Shortcut: `Ctrl + R`).

The automated execution flow operates as follows:

1. Imports and processes the incoming asset data inside the content directories.
2. Dynamically scans the active Editor level to locate the `BP_Meta_Avatar_C` instance.
3. Accesses the target Skeletal Mesh Component (`Body`).
4. Forces an immediate single-node playback override in the active viewport layout.
5. Permanently serializes the animation sequence to the instance's `animation_data` properties so that it persists when jumping into a live Play-In-Editor (PIE) session.

#### Testing the Unreal Engine Node Independently

With Unreal Engine open in Edit mode, you can mock the incoming data packet from Blender using `curl` from any system terminal:

```Bash
curl -X PUT "[http://127.0.0.1:30010/remote/object/call](http://127.0.0.1:30010/remote/object/call)" \
     -H "Content-Type: application/json" \
     -d '{"objectPath": "/Game/Scripts/ImportListener.Default__ImportListener_C", "functionName": "ImportAndRetargetFBX", "parameters": {"FilePath": "K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert/running_franctic.fbx"}}'

```