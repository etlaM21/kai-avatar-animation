import bpy
import numpy as np
import mathutils

# Your specific file path
file_path = r"C:\Users\malte\Desktop\project_kaspar\modules\kai-avatar-animation\kimodo\soma_skeleton\skin_standard.npz"

# Standard swap for Y-up to Z-up
R_YUP_TO_ZUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0,  1,  0]
], dtype=np.float64)

def import_soma_standard(path):
    data = np.load(path, allow_pickle=True)

    # --- 1. Geometry ---
    verts_raw = data['bind_vertices'].astype(np.float64)
    # Apply axis conversion
    verts = (R_YUP_TO_ZUP @ verts_raw.T).T
    
    faces_raw = data['faces']
    if faces_raw.ndim == 1:
        faces_raw = faces_raw.reshape(-1, 3)
    faces = faces_raw.tolist()

    # --- 2. Skeleton Data ---
    raw_names = data['rig_joint_names']
    joint_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in raw_names]
    
    transforms = data['bind_rig_transform']
    # Extract translation based on array shape (4x4 matrix vs 3D vector)
    if transforms.ndim == 3 and transforms.shape[1] == 4:
        t_coords_raw = transforms[:, :3, 3]
    else:
        t_coords_raw = transforms[:, :3]
        
    t_coords = (R_YUP_TO_ZUP @ t_coords_raw.T).T

    # Map Parent-Child Connections
    connections = data['rig_joint_connections']
    parent_map = {}
    if connections.ndim == 2:
        for row in connections:
            parent_map[int(row[1])] = int(row[0]) # Assumes [parent, child]
    elif connections.ndim == 1:
        for c, p in enumerate(connections):
            parent_map[c] = int(p)

    # --- 3. Build Mesh ---
    mesh = bpy.data.meshes.new("SOMA_Standard_Mesh")
    obj = bpy.data.objects.new("SOMA_Body", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()

    # --- 4. Build Armature ---
    arm_data = bpy.data.armatures.new("SOMA_Standard_Armature")
    arm_obj = bpy.data.objects.new("SOMA_Skeleton", arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones
    bones_list = []

    for i, name in enumerate(joint_names):
        b = edit_bones.new(name)
        pos = t_coords[i]
        b.head = pos.tolist()
        b.tail = (pos + np.array([0, 0, 0.05])).tolist()
        bones_list.append(b)

    # Apply Parenting
    for c, p in parent_map.items():
        if 0 <= p < len(bones_list) and 0 <= c < len(bones_list):
            bones_list[c].parent = bones_list[p]

    # Align Tails to Children for better visualization
    child_map = {}
    for c, p in parent_map.items():
        child_map.setdefault(p, []).append(c)

    for p, children in child_map.items():
        if children and 0 <= p < len(bones_list):
            c_head = bones_list[children[0]].head
            if (mathutils.Vector(c_head) - mathutils.Vector(bones_list[p].head)).length > 1e-5:
                bones_list[p].tail = c_head.copy()

    bpy.ops.object.mode_set(mode='OBJECT')

    # --- 5. Apply Skinning Weights ---
    vg_list = [obj.vertex_groups.new(name=name) for name in joint_names]
    lbs_indices = data['lbs_indices']
    lbs_weights = data['lbs_weights']

    for v_idx in range(len(verts)):
        v_inds = lbs_indices[v_idx]
        v_wgts = lbs_weights[v_idx]
        for j_idx, w in zip(v_inds, v_wgts):
            if float(w) > 1e-6:
                vg_list[int(j_idx)].add([v_idx], float(w), 'REPLACE')

    # Bind Mesh to Armature
    obj.parent = arm_obj
    mod = obj.modifiers.new(name="SOMA_Skin", type='ARMATURE')
    mod.object = arm_obj
    mod.use_vertex_groups = True

    print(f"--- SUCCESS: Imported {len(verts)} verts and {len(joint_names)} joints. ---")

# Run it
import_soma_standard(file_path)