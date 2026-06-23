import torch
import cv2
import numpy as np
import trimesh

def visualize_masks_on_image_cv2(img, mask_list, out_path="sofa.png"):
    # --- PIL -> numpy ---
    img = np.array(img)  # RGB, uint8

    # --- RGB -> BGR ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    overlay = img.copy()
    alpha = 0.6

    for idx, mask in enumerate(mask_list):
        # torch.Tensor -> numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        # ensure HxW
        mask = mask.reshape(h, w).astype(bool)

        color = np.random.randint(0, 256, size=3, dtype=np.uint8)

        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * color
        ).astype(np.uint8)

    cv2.imwrite(out_path, overlay)



def save_seg_obj(img, mask, out_path="sofa.png"):
    # Save RGBA image with only the masked objects shown
    # --- PIL -> numpy ---
    img = np.array(img)  # RGB, uint8

    # --- RGB -> BGR ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # RGBA output for vis_obj (with transparency)
    vis_obj = np.zeros((h, w, 4), dtype=np.uint8)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # ensure HxW
    mask = mask.reshape(h, w).astype(bool)

    vis_obj[..., :3][mask] = img[mask]
    vis_obj[..., 3][mask] = 255

    cv2.imwrite(out_path, vis_obj)


def vis_axes_ply(center, axes, save_path, up_axis_idx=None, radius=0.02):
    """
    Save axes visualization as PLY (for debug visualization only).
    Args:
      - center: (3,) torch/np
      - axes:   (3,3) torch/np
      - save_path: path to PLY file
    """
    from .mesh import axes_to_cylinder_mesh

    axes_mesh = axes_to_cylinder_mesh(
        axes=axes,
        centroid=center,
        up_axis_idx=up_axis_idx,
        radius=radius
    )
    axes_mesh.export(save_path)

def vis_bboxes_ply(bbox_corners, save_path):
    """
    Save a 3D bounding box (given by 8 corners) as a wireframe PLY (for debug visualization only).
    bbox_corners: (8,3) torch.Tensor or np.ndarray
    """
    if torch.is_tensor(bbox_corners):
        corners = bbox_corners.detach().cpu().numpy()
    else:
        corners = bbox_corners

    # edges (pairs of vertex indices), same topology as before
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ])

    path = trimesh.load_path(corners[edges])
    path.export(save_path)