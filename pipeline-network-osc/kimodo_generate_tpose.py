import torch
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

MODEL_NAME = "Kimodo-SOMA-RP-v1.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 30 # Required by the API

print(f"Loading {MODEL_NAME}...")
model = load_model(MODEL_NAME, device=DEVICE)
skeleton = model.output_skeleton
num_joints = len(skeleton.bone_order_names)

print("Forging absolute Zero-Rotation Rest Pose...")
# Create absolute identity matrices (No rotation)
# Shape: [Batch=1, Frame=1, Joints=61, 3, 3]
zero_rots = torch.eye(3, device=DEVICE).view(1, 1, 1, 3, 3).repeat(1, 1, num_joints, 1, 1)

# Create absolute zero translation (0, 0, 0)
# Shape: [Batch=1, Frame=1, XYZ=3]
zero_pos = torch.zeros((1, 1, 3), device=DEVICE)

output_filename = "kimodo_rest_t-pose.bvh"
print(f"Exporting pristine base to {output_filename}...")

# Export using the strict API signature with the T-Pose flag
save_motion_bvh(
    path=output_filename,
    local_rot_mats=zero_rots,
    root_positions=zero_pos,
    skeleton=skeleton,
    fps=FPS,
    standard_tpose=True
)

print("Done. You may now run the Blender normalization script on this file.")