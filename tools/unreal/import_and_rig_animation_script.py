import unreal
import os

# --- 1. SETTINGS & PATHS ---
windows_fbx_dir = r"K:/work/kai_avatar/ue_project/Kai_Avatar/Content/BVH/FBX_Convert"

ue_import_path = "/Game/BVH/FBX_Convert"
ue_retarget_path = "/Game/BVH/MetaHuman_Ready"

skeleton_path = "/Game/BVH/FBX_Convert/wave_Skeleton.wave_Skeleton" 
source_mesh_path = "/Game/BVH/FBX_Convert/wave.wave" 
target_mesh_path = "/Game/MetaHumans/Meta_Avatar/Body/SKM_Meta_Avatar_BodyMesh.SKM_Meta_Avatar_BodyMesh"
ik_retargeter_path = "/Game/BVH/RTG_Kimodo_to_MetaHuman.RTG_Kimodo_to_MetaHuman" # Verify this matches your Retargeter path!


def import_fbx_to_skeleton(fbx_file, dest_path, skeleton_asset):
    """Imports an FBX and strictly returns the AnimSequence AssetData."""
    task = unreal.AssetImportTask()
    task.filename = fbx_file
    task.destination_path = dest_path
    task.automated = True
    task.replace_existing = True
    
    options = unreal.FbxImportUI()
    options.import_mesh = False 
    options.import_as_skeletal = True
    options.import_animations = True
    options.skeleton = skeleton_asset
    options.mesh_type_to_import = unreal.FBXImportType.FBXIT_ANIMATION
    options.automated_import_should_detect_type = False
    
    task.options = options
    
    # Run the import
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    
    asset_name = os.path.splitext(os.path.basename(fbx_file))[0]
    
    # We check _Anim first, as that is standard for animations
    possible_paths = [
        f"{dest_path}/{asset_name}_Anim.{asset_name}_Anim",
        f"{dest_path}/{asset_name}.{asset_name}"
    ]
    
    # Verify the class is actually an Animation before returning
    for path in possible_paths:
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            asset_data = unreal.EditorAssetLibrary.find_asset_data(path)
            # Ensure Unreal recognizes this specific file as an AnimSequence
            if str(asset_data.asset_class_path.asset_name) == "AnimSequence":
                return asset_data
                
    return None


def automate_pipeline():
    """Main function to run the import and retarget logic."""
    skeleton = unreal.EditorAssetLibrary.load_asset(skeleton_path)
    source_mesh = unreal.EditorAssetLibrary.load_asset(source_mesh_path)
    target_mesh = unreal.EditorAssetLibrary.load_asset(target_mesh_path)
    retargeter = unreal.EditorAssetLibrary.load_asset(ik_retargeter_path)
    
    if not all([skeleton, source_mesh, target_mesh, retargeter]):
        unreal.log_error("Asset load failed. Double-check your /Game/... paths!")
        return

    for filename in os.listdir(windows_fbx_dir):
        if filename.endswith(".fbx"):
            file_path = os.path.join(windows_fbx_dir, filename)
            unreal.log(f"--- Processing: {filename} ---")
            
            # Step 1: Import the FBX and get the validated AssetData
            anim_asset_data = import_fbx_to_skeleton(file_path, ue_import_path, skeleton)
            
            if anim_asset_data:
                unreal.log(f"> Retargeting: {filename}")
                
                # Step 2: Retarget and CAPTURE the output
                retargeted_assets = unreal.IKRetargetBatchOperation.duplicate_and_retarget(
                    assets_to_retarget=[anim_asset_data],
                    source_mesh=source_mesh,
                    target_mesh=target_mesh,
                    ik_retarget_asset=retargeter,
                    search="",
                    replace="",
                    prefix="MH_", 
                    suffix="",
                    include_referenced_assets=False
                )
                
                # Step 3: Move the exact file Unreal generated
                if retargeted_assets and len(retargeted_assets) > 0:
                    new_anim_data = retargeted_assets[0]
                    
                    # Ask Unreal exactly where it dumped the file
                    actual_saved_path = str(new_anim_data.package_name)
                    asset_name = str(new_anim_data.asset_name)
                    
                    # Define our specific target destination
                    final_destination = f"{ue_retarget_path}/{asset_name}"
                    
                    # Move it to the correct folder
                    if actual_saved_path != final_destination:
                        success = unreal.EditorAssetLibrary.rename_asset(actual_saved_path, final_destination)
                        if success:
                            unreal.log(f"> Success: Moved to {final_destination}")
                        else:
                            unreal.log_error(f"> Error: Could not move to {final_destination}")
                else:
                    unreal.log_error(f"> Error: Retargeting returned no assets for {filename}")
            else:
                unreal.log_error(f"Failed to find valid AnimSequence for {filename}")
                        
    unreal.log("Automation complete!")

# Execute the pipeline
automate_pipeline()