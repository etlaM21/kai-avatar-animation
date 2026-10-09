import bpy
import os

# Set your paths
input_dir = r"K:/project_kaspar/modules/kimodo_playground/kimodo-gen"
output_dir = r"K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def purge_blender_data():
    """Deep cleans the scene and data blocks to prevent animation bleeding."""
    # Ensure we are in object mode
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Purge orphan data (Meshes, Armatures, and crucially, Actions)
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.armatures: bpy.data.armatures.remove(block)
    for block in bpy.data.actions: bpy.data.actions.remove(block)

# Process all BVH files
for filename in os.listdir(input_dir):
    if filename.endswith(".bvh"):
        purge_blender_data() # Start with a 100% clean slate

        # Import BVH
        bvh_path = os.path.join(input_dir, filename)
        # update_scene_fps=True ensures the timeline matches the Kimodo output
        bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=1.0, update_scene_fps=True, update_scene_duration=True)
        
        # 1. Identify the imported armature
        armature = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE'][0]
        armature.name = "Kimodo_Rig"
        
        # 2. Create Dummy Mesh (A simple cube)
        bpy.ops.mesh.primitive_cube_add(size=0.2) # Small cube so it doesn't block the view
        dummy_mesh = bpy.context.active_object
        dummy_mesh.name = "UE_Dummy_Mesh"
        
        # 3. Skin the mesh to the Armature
        # We select the mesh, then the armature, then parent with 'Automatic Weights'
        bpy.ops.object.select_all(action='DESELECT')
        dummy_mesh.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature 
        
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        # 4. Export as FBX
        fbx_filename = filename.replace(".bvh", ".fbx")
        fbx_path = os.path.join(output_dir, fbx_filename)
        
        # Select everything to export
        bpy.ops.object.select_all(action='SELECT')
        
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            bake_anim=True,
            add_leaf_bones=False,
            axis_forward='-Z',
            axis_up='Y',
            # CRITICAL: Set these to False to prevent 'All Actions' from leaking into the file
            bake_anim_use_all_actions=False, 
            bake_anim_use_nla_strips=False
        )
        
        print(f"Clean Export Finished: {fbx_filename}")