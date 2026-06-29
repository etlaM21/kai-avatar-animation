import unreal
import os
import time
from datetime import datetime

# --- 1. SETTINGS & PATHS ---
ue_import_path = "/Game/BVH/FBX_Convert"
ue_retarget_path = "/Game/BVH/MetaHuman_Ready"

skeleton_path = "/Game/Kimodo_Export_A-Skeleton.Kimodo_Export_A-Skeleton" 
source_mesh_path = "/Game/Kimodo_Export_A-SkeletalMesh.Kimodo_Export_A-SkeletalMesh" 
target_mesh_path = "/Game/MetaHumans/MH_Kaspar/Body/SKM_MH_Kaspar_BodyMesh.SKM_MH_Kaspar_BodyMesh"
ik_retargeter_path = "/Game/RTG_Kimodo_to_MetaHuman.RTG_Kimodo_to_MetaHuman" 

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
    
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    
    asset_name = os.path.splitext(os.path.basename(fbx_file))[0]
    
    possible_paths = [
        f"{dest_path}/{asset_name}_Anim.{asset_name}_Anim",
        f"{dest_path}/{asset_name}.{asset_name}"
    ]
    
    for path in possible_paths:
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            asset_data = unreal.EditorAssetLibrary.find_asset_data(path)
            if str(asset_data.asset_class_path.asset_name) == "AnimSequence":
                return asset_data
                
    return None

def process_single_fbx(file_path):
    """Main function triggered via Web Remote Control."""
    # --- START TIMER ---
    ue_start_time = time.time()
    unreal.log(f"=== Beginning Automated Pipeline Ingestion for: {file_path} ===")

    if not os.path.exists(file_path):
        unreal.log_error(f"Pipeline Interrupted: FBX path does not exist on disk: {file_path}")
        return

    skeleton = unreal.EditorAssetLibrary.load_asset(skeleton_path)
    source_mesh = unreal.EditorAssetLibrary.load_asset(source_mesh_path)
    target_mesh = unreal.EditorAssetLibrary.load_asset(target_mesh_path)
    retargeter = unreal.EditorAssetLibrary.load_asset(ik_retargeter_path)
    
    if not all([skeleton, source_mesh, target_mesh, retargeter]):
        unreal.log_error("Asset load failed. Double-check your /Game/... paths!")
        return

    filename = os.path.basename(file_path)
    unreal.log(f"--- Processing: {filename} ---")
    
    # Step 1: Import the FBX
    anim_asset_data = import_fbx_to_skeleton(file_path, ue_import_path, skeleton)
    
    if anim_asset_data:
        unreal.log(f"> Retargeting: {filename}")
        
        # Step 2: Retarget
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
        
        # Step 3: Move to final destination
        if retargeted_assets and len(retargeted_assets) > 0:
            new_anim_data = retargeted_assets[0]
            
            actual_saved_path = str(new_anim_data.package_name)
            asset_name = str(new_anim_data.asset_name)
            final_destination = f"{ue_retarget_path}/{asset_name}"
            
            if actual_saved_path != final_destination:
                success = unreal.EditorAssetLibrary.rename_asset(actual_saved_path, final_destination)
                if success:
                    unreal.log(f"> Success: Moved to {final_destination}")

                     # --- Step 4: TRIGGER PLAYBACK ---
                        
                    # 1. Load the final animation object into memory
                    final_anim_sequence = unreal.EditorAssetLibrary.load_asset(f"{final_destination}.{asset_name}")
                    
                    # 2. Get all actors currently sitting in your active Editor Level
                    all_actors = unreal.EditorLevelLibrary.get_all_level_actors()
                    
                    avatar = None
                    # 3. Safely scan for Kai without relying on fragile hardcoded string paths
                    for actor in all_actors:
                        if actor.get_class().get_name() == "BP_Meta_Avatar_C":
                            avatar = actor
                            break
                    
                    if avatar:
                        # 4. Grab the Body skeletal mesh component
                        body_mesh = avatar.get_component_by_class(unreal.SkeletalMeshComponent.static_class())
                        if body_mesh:
                            # 5. Override AnimBP and play immediately in the Editor viewport
                            body_mesh.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
                            body_mesh.play_animation(final_anim_sequence, False)
                            unreal.log(">> Playback triggered on Avatar in Level!")
                            
                            # 6. Overwrite the properties so it survives hitting "Play" (PIE)
                            body_mesh.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_SINGLE_NODE)
                            
                            # Access the Animation Data struct, update the sequence, and re-assign it
                            anim_data = body_mesh.get_editor_property("animation_data")
                            anim_data.set_editor_property("anim_to_play", final_anim_sequence)
                            body_mesh.set_editor_property("animation_data", anim_data)
                            
                            unreal.log(f">> Sequence permanently assigned to Avatar instance: {asset_name}")
                           
                            # --- LOGGING TO SHARED FILE ---
                            ue_duration = time.time() - ue_start_time
                            
                            # Hardcode the exact absolute path to your Windows folder
                            log_path = r"C:\Users\philip\Repos\shk\repos\project_kaspar\modules\kai-avatar-animation\pipeline-network-editor\pipeline_timing_log.txt"
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_line = f"[{timestamp}] | File: {filename} | Stage: Unreal_Import_And_Retarget | Duration: {ue_duration:.3f}s\n"
                            
                            with open(log_path, "a") as f:
                                f.write(log_line)
                            # ------------------------------
                            
                    else:
                        unreal.log_warning("Could not find an actor of class 'BP_Meta_Avatar' in the current level.")
                else:
                    unreal.log_error(f"> Error: Could not move to {final_destination}")
        else:
            unreal.log_error(f"> Error: Retargeting returned no assets for {filename}")
    else:
        unreal.log_error(f"Failed to find valid AnimSequence for {filename}")
        
    return None