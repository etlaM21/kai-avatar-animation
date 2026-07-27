import bpy
import sys

# Extract arguments passed after '--'
argv = sys.argv
if "--" not in argv:
    print("Error: Missing '--' separator for arguments.")
    sys.exit(1)
    
args = argv[argv.index("--") + 1:]
if len(args) < 2:
    print("Usage: blender -b -P normalize_skeleton.py -- <input.bvh> <output.fbx>")
    sys.exit(1)

bvh_in = args[0]
fbx_out = args[1]

# Clear the default Blender scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the BVH
print(f"Importing BVH: {bvh_in}")
bpy.ops.import_anim.bvh(filepath=bvh_in, filter_glob="*.bvh", global_scale=1.0, use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE')

armature = bpy.context.active_object
if armature and armature.type == 'ARMATURE':
    
    # --- 1. STRIP RESIDUAL BVH ANIMATION ---
    armature.animation_data_clear()
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.transforms_clear()
    
    # --- 2. NORMALIZE BONE ROLLS ---
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # --- 3. GENERATE DUMMY MESH WITH WEIGHTS ---
    bpy.ops.mesh.primitive_cylinder_add(radius=2.0, depth=4.0, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    dummy_mesh = bpy.context.active_object
    dummy_mesh.name = "DummyMesh"
    
    bpy.ops.object.select_all(action='DESELECT')
    dummy_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    # --- 4. FREEZE TRANSFORMS (THE UE5 FIX) ---
    print("Freezing transforms for Unreal Engine compatibility...")
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    dummy_mesh.select_set(True)
    # This bakes the location, rotation, and scale into the vertices and bones permanently
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # --- 5. EXPORT FBX ---
    print(f"Exporting clean FBX: {fbx_out}")
    bpy.ops.export_scene.fbx(
        filepath=fbx_out, 
        use_selection=True, 
        add_leaf_bones=False, 
        bake_anim=False,
        apply_scale_options='FBX_SCALE_ALL', # Forces scale to be written directly into the mesh/bones
        axis_forward='Y',
        axis_up='Z'
    )
    print("SUCCESS: Skeleton normalized, weighted, and frozen for UE5.")
else:
    print("ERROR: Armature not found after import.")