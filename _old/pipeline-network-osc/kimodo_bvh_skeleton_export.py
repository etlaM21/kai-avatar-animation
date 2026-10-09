import torch
from kimodo.model.load_model import load_model
from kimodo.exports.bvh import save_motion_bvh

print("Loading Kimodo model...")
model = load_model("Kimodo-SOMA-RP-v1.1", device="cpu")
skeleton = model.output_skeleton

print("Generating pure mathematical zero-rotation pose...")
num_joints = len(skeleton.bone_order_names)

# Create Identity matrices for 1 frame: shape (1, 1, Joints, 3, 3)
zero_rots = torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 1, num_joints, 3, 3)
# Create zero positions: shape (1, 1, 3)
zero_pos = torch.zeros(1, 1, 3)

# Exact signature from the Kimodo docs
save_motion_bvh(
    "True_SOMA_Rest_Pose.bvh",  # path
    zero_rots,                  # local_rot_mats
    zero_pos,                   # root_positions
    skeleton=skeleton,          # kwarg
    fps=30                      # kwarg
)

print("Saved! Import True_SOMA_Rest_Pose.bvh into Unreal Engine.")