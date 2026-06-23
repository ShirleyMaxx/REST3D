"""Quaternion math utilities (numpy + torch, xyzw convention)."""
import math
import numpy as np
import torch

IDENTITY_QUAT   = np.array([0.0, 0.0, 0.0, 1.0])
IDENTITY_QUAT_T = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)


def quat_angle_distance(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * math.acos(min(d, 1.0))


def quat_to_rotmat(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def quat_inverse_np(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def local_group_child_offset_parent_frame(root_quat, root_pos, child_pos, child_quat):
    R = quat_to_rotmat(root_quat)
    delta_w = child_pos - root_pos
    ofs = (R.T @ delta_w).astype(np.float32)
    qrel = quat_multiply(quat_inverse_np(root_quat), child_quat)
    return ofs, qrel


def axis_angle_to_quat_batch(aa):
    angle = torch.norm(aa, dim=1, keepdim=True)
    axis = aa / (angle + 1e-12)
    s = torch.sin(angle / 2)
    c = torch.cos(angle / 2)
    quat = torch.cat([axis * s, c], dim=1)
    small = (angle.squeeze(1) < 1e-8)
    if small.any():
        quat[small] = IDENTITY_QUAT_T
    return quat


def quat_multiply_batch(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=1)


def quat_angle_distance_batch(q1, q2):
    if q2.dim() == 1:
        q2 = q2.unsqueeze(0)
    dot = torch.abs(torch.sum(q1 * q2, dim=1))
    dot = torch.clamp(dot, max=1.0)
    return 2.0 * torch.acos(dot)


def quaternion_to_euler_phyrecon_degrees(x, y, z, w):
    psi   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    theta = np.arcsin(np.clip(2 * (w * x - y * z), -1.0, 1.0))
    phi   = np.arctan2(2 * (w * y + z * x), 1 - 2 * (x**2 + y**2))
    return np.degrees(psi), np.degrees(theta), np.degrees(phi)


def quaternion_to_euler_phyrecon_radians(x, y, z, w):
    psi   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    theta = np.arcsin(np.clip(2 * (w * x - y * z), -1.0, 1.0))
    phi   = np.arctan2(2 * (w * y + z * x), 1 - 2 * (x**2 + y**2))
    return float(psi), float(theta), float(phi)


def wrap_angle_rad(delta):
    d = float(delta)
    return (d + math.pi) % (2.0 * math.pi) - math.pi
