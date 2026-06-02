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