import bpy
import os
import sys
import json
import urllib.request
import urllib.error

# --- Dependency Check ---
try:
    import zmq
except ImportError:
    print("\n[CRITICAL ERROR] 'pyzmq' is not installed in Blender's internal Python.")
    print("Please install it using Blender's specific python executable. Example:")
    print("C:\\Program Files\\Blender Foundation\\Blender 4.0\\4.0\\python\\bin\\python.exe -m pip install pyzmq\n")
    sys.exit(1)

# --- Configuration ---
ZMQ_PORT = 42070
OUTPUT_DIR = r"K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert"
UE_REMOTE_URL = "http://127.0.0.1:30010/remote/object/call"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def purge_blender_data():
    """Deep cleans the scene and data blocks to prevent animation bleeding."""
    # Safer mode check to prevent headless context crashes
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.armatures: bpy.data.armatures.remove(block)
    for block in bpy.data.actions: bpy.data.actions.remove(block)

def ping_unreal_engine(fbx_path):
    """Signals Unreal Engine via Web Remote Control to ingest the new FBX."""
    # Note: We will set up the exact receiving Blueprint/Python script in the Node 3 phase.
    payload = {
        # This object path will be updated when we build the Unreal side
        "objectPath": "/Game/Scripts/ImportListener.Default__ImportListener_C", 
        "functionName": "ImportAndRetargetFBX",
        "parameters": {
            "FilePath": fbx_path
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(UE_REMOTE_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        response = urllib.request.urlopen(req, timeout=5) 
        print(f"   [Unreal Handoff] Success: Triggered UE Import (HTTP {response.status})")
    except urllib.error.URLError as e:
        print(f"   [Unreal Handoff] Warning: Could not reach Unreal Engine at {UE_REMOTE_URL}. Is it running? Error: {e}")

def run_service():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://127.0.0.1:{ZMQ_PORT}")
    
    print("\n" + "="*60)
    print("Blender Headless Service Online.")
    print(f"Listening for BVH conversion tasks via ZMQ on port {ZMQ_PORT}...")
    print("="*60 + "\n")

    while True:
        # 1. Block and wait for a task. This uses 0% CPU while idle.
        bvh_path = socket.recv_string()
        print(f"\n[Incoming Task] Received: {bvh_path}")

        if not os.path.exists(bvh_path):
            print(f"   [Error] File not found: {bvh_path}. Skipping.")
            continue

        try:
            # 2. Clean Slate
            purge_blender_data()

            # 3. Import
            bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=1.0, update_scene_fps=True, update_scene_duration=True)
            
            armature = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE'][0]
            armature.name = "Kimodo_Rig"
            
            bpy.ops.mesh.primitive_cube_add(size=0.2)
            dummy_mesh = bpy.context.active_object
            dummy_mesh.name = "UE_Dummy_Mesh"
            
            # Skinning
            bpy.ops.object.select_all(action='DESELECT')
            dummy_mesh.select_set(True)
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature 
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
            
            # 4. Export
            filename = os.path.basename(bvh_path)
            fbx_filename = filename.replace(".bvh", ".fbx")
            fbx_path = os.path.join(OUTPUT_DIR, fbx_filename)
            
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.export_scene.fbx(
                filepath=fbx_path,
                use_selection=True,
                bake_anim=True,
                add_leaf_bones=False,
                axis_forward='-Z',
                axis_up='Y',
                bake_anim_use_all_actions=False, 
                bake_anim_use_nla_strips=False
            )
            print(f"   [Conversion] Clean Export Finished: {fbx_filename}")

            # 5. Ping Node 3 (Unreal Engine)
            ping_unreal_engine(fbx_path)

        except Exception as e:
            print(f"   [Error] Conversion failed: {str(e)}")

if __name__ == "__main__":
    run_service()