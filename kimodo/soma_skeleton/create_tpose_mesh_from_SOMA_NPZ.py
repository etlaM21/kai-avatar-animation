import bpy
import numpy as np
import mathutils
import sys
# This is the path where your pip installation currently lives
sys.path.append(r"C:\Users\malte\AppData\Roaming\Python\Python313\site-packages")
from scipy.sparse import csc_matrix, diags

file_path = r"C:\Users\malte\Desktop\project_kaspar\modules\kai-avatar-animation\kimodo\soma_skeleton\skin_standard.npz"

R_YUP_TO_ZUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0,  1,  0]
], dtype=np.float64)

def transform_points(pts, R):
    return (R @ pts.T).T

def import_soma_final(path):
    data = np.load(path, allow_pickle=True)

    # 1. Decode joint names and parents
    raw_names     = data['joint_names']
    joint_names   = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in raw_names]
    joint_parents = data['joint_parent_ids'].astype(int)
    n_joints      = len(joint_names)

    # Use bind_shape and transform it to t_pose_world using LBS
    bind_shape_raw = data['bind_shape'][:, :3].astype(np.float64)
    bind_pose      = data['bind_pose_world']
    t_pose         = data['t_pose_world']

    # Skinning weights (CSC format)
    sw_data    = data['skinning_weights_data'].astype(np.float64)
    sw_indices = data['skinning_weights_indices'].astype(np.int32)
    sw_indptr  = data['skinning_weights_indptr'].astype(np.int32)
    sw_shape   = tuple(data['skinning_weights_shape'].astype(int))

    W_csc = csc_matrix((sw_data, sw_indices, sw_indptr), shape=sw_shape)
    W_csr = W_csc.tocsr()
    row_sums = np.array(W_csr.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums) @ W_csr
    W_coo  = W_norm.tocoo()

    # Precompute M_i = T_i @ B_i^-1 for each joint
    M = np.zeros((n_joints, 4, 4), dtype=np.float64)
    for i in range(n_joints):
        try:
            B_inv = np.linalg.inv(bind_pose[i])
            M[i] = t_pose[i] @ B_inv
        except np.linalg.LinAlgError:
            M[i] = np.eye(4)

    # Apply LBS to get t_shape_raw (the mesh vertices in T-pose before Z-up conversion)
    verts_homo = np.hstack([bind_shape_raw, np.ones((bind_shape_raw.shape[0], 1))])
    t_shape_raw = np.zeros_like(verts_homo)

    for v_idx, j_idx, w in zip(W_coo.row, W_coo.col, W_coo.data):
        if w > 1e-6:
            t_shape_raw[v_idx] += w * (M[j_idx] @ verts_homo[v_idx])

    t_shape_raw = t_shape_raw[:, :3]
    t_coords_raw = t_pose[:, :3, 3].astype(np.float64)

    print("=== DEBUG ===")
    print(f"t_shape centroid (Y-up): {t_shape_raw.mean(axis=0)}")
    print(f"t_pose_world translations (first 5 joints, Y-up):")
    for i in range(min(5, n_joints)):
        print(f"  [{i}] {joint_names[i]}: {t_coords_raw[i]}")
    print(f"t_shape Y range: {t_shape_raw[:,1].min():.3f} to {t_shape_raw[:,1].max():.3f}")
    print(f"bone    Y range: {t_coords_raw[:,1].min():.3f} to {t_coords_raw[:,1].max():.3f}")
    print("=== END DEBUG ===")

    # 2. Vertices — apply Y-up -> Z-up
    vertices = transform_points(t_shape_raw, R_YUP_TO_ZUP)
    n_verts  = len(vertices)

    # 3. Bone positions from t_pose_world — apply Y-up -> Z-up conversion
    t_coords = transform_points(t_coords_raw, R_YUP_TO_ZUP)

    # 4. Reconstruct faces
    face_vert_counts  = data['face_vert_counts'].astype(int)
    face_vert_indices = data['face_vert_indices'].astype(int)
    faces = []
    idx = 0
    for count in face_vert_counts:
        faces.append(face_vert_indices[idx: idx + count].tolist())
        idx += count

    # --- 1. Create Mesh ---
    mesh = bpy.data.meshes.new("SOMA_T_Mesh")
    obj  = bpy.data.objects.new("SOMA_Body_T", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices.tolist(), [], faces)
    if mesh.validate(verbose=True):
        print("WARNING: Mesh had validation issues.")
    mesh.update()

    # --- 2. Create Armature ---
    arm_data = bpy.data.armatures.new("SOMA_T_Armature")
    arm_obj  = bpy.data.objects.new("SOMA_Skeleton_T", arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones
    bones_list = []

    for i, name in enumerate(joint_names):
        b   = edit_bones.new(name)
        pos = t_coords[i]
        b.head = pos.tolist()
        b.tail = (pos + np.array([0, 0, 0.05])).tolist()
        bones_list.append(b)

    # Parent relationships
    for i, p_idx in enumerate(joint_parents):
        if 0 <= p_idx < len(bones_list):
            bones_list[i].parent = bones_list[p_idx]

    # Point parent tails toward first child
    child_map = {}
    for i, p_idx in enumerate(joint_parents):
        if 0 <= p_idx < len(bones_list):
            child_map.setdefault(int(p_idx), []).append(i)

    for parent_idx, child_indices in child_map.items():
        child_head  = bones_list[child_indices[0]].head
        parent_head = bones_list[parent_idx].head
        if (mathutils.Vector(child_head) - mathutils.Vector(parent_head)).length > 1e-5:
            bones_list[parent_idx].tail = child_head.copy()
        else:
            h = bones_list[parent_idx].head
            bones_list[parent_idx].tail = (h[0], h[1], h[2] + 0.05)

    bpy.ops.object.mode_set(mode='OBJECT')

    # --- 3. Skinning Weights ---
    obj.vertex_groups.clear()
    vg_list = [obj.vertex_groups.new(name=name) for name in joint_names]

    bone_verts   = {j: [] for j in range(n_joints)}
    bone_weights = {j: [] for j in range(n_joints)}
    for v_idx, j_idx, w in zip(W_coo.row, W_coo.col, W_coo.data):
        if w > 1e-6:
            bone_verts[int(j_idx)].append(int(v_idx))
            bone_weights[int(j_idx)].append(float(w))

    for j_idx, vg in enumerate(vg_list):
        verts   = bone_verts[j_idx]
        weights = bone_weights[j_idx]
        if not verts:
            continue
        for v_idx, w in zip(verts, weights):
            vg.add([v_idx], w, 'REPLACE')

    # --- 4. UV Map ---
    if 'uv_coord_st' in data.files and 'uv_indices_st' in data.files:
        try:
            uv_coords  = data['uv_coord_st'].astype(np.float32)
            uv_indices = data['uv_indices_st'].astype(int)
            uv_layer   = mesh.uv_layers.new(name="UVMap")
            loop_idx = 0
            uv_ptr   = 0
            for count in face_vert_counts:
                for _ in range(count):
                    if uv_ptr < len(uv_indices) and loop_idx < len(uv_layer.data):
                        uv = uv_coords[uv_indices[uv_ptr]]
                        uv_layer.data[loop_idx].uv = (float(uv[0]), float(uv[1]))
                    loop_idx += 1
                    uv_ptr   += 1
            print("UV map applied.")
        except Exception as e:
            print(f"UV map skipped: {e}")

    # --- 5. Finalize ---
    obj.parent = arm_obj
    mod = obj.modifiers.new(name="SOMA_Skin", type='ARMATURE')
    mod.object = arm_obj
    mod.use_vertex_groups = True

    print(f"T-pose Import complete. Verts: {n_verts}, Faces: {len(faces)}, Bones: {n_joints}")

# RUN
import_soma_final(file_path)
