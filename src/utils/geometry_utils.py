import torch
import numpy as np
import trimesh
from pytorch360convert import e2c, c2e


def erp_to_cubemap(erp_tensor, face_w = 768, cube_format = "stack", mode = "bilinear", **kwargs):
    return e2c(erp_tensor, face_w=face_w, cube_format=cube_format, mode=mode, **kwargs)


def cubemap_to_erp(cube_tensor, erp_h = 1024, erp_w = 2048, cube_format = "stack", mode = "bilinear", **kwargs):
    return c2e(cube_tensor, h=erp_h, w=erp_w, cube_format=cube_format, mode=mode, **kwargs)

def roll_augment(data, shift_x):
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
        originally_2d = True
    else:
        originally_2d = False
    if data.ndim == 3 and data.shape[0] != 3:
        data = np.moveaxis(data, -1, 0)
        moved_axis = True
    else:
        moved_axis = False

    data_rolled = np.roll(data, int(shift_x), axis=2)

    if moved_axis:
        data_rolled = np.moveaxis(data_rolled, 0, -1)
    if originally_2d:
        data_rolled = data_rolled[:, :, 0]
    return data_rolled


def roll_normals(normals, shift_x):
    if normals.ndim == 2:
        normals = normals[:, :, np.newaxis]
        originally_2d = True
    else:
        originally_2d = False
    if normals.ndim == 3 and normals.shape[0] != 3:
        normals = np.moveaxis(normals, -1, 0)
        moved_axis = True
    else:
        moved_axis = False

    _, H, W = normals.shape

    angle = - 2.0 * np.pi * (shift_x / float(W))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [ cos_a, 0.0, -sin_a],
        [ 0.0,   1.0,  0.0  ],
        [ sin_a, 0.0,  cos_a]
    ], dtype=normals.dtype)

    n_flat = normals.reshape(3, -1)
    normals = (R @ n_flat).reshape(3, H, W)

    if moved_axis:
        normals = np.moveaxis(normals, 0, -1)

    if originally_2d:
        normals = normals[:, :, 0]
    return normals

def compute_scale_and_shift(pred_g, targ_g, mask_g = None, eps = 0.0, fit_shift = True):
    if mask_g is None:
        mask_g = torch.ones_like(pred_g, dtype=torch.bool)
    if pred_g.shape[0] == 6:
        pred_g = pred_g.view(1, 6, pred_g.shape[2], pred_g.shape[3])
        targ_g = targ_g.view(1, 6, targ_g.shape[2], targ_g.shape[3])
        mask_g = mask_g.view(1, 6, mask_g.shape[2], mask_g.shape[3])
    elif pred_g.shape[0] == 1 and pred_g.dim() == 3:
        pred_g = pred_g.unsqueeze(0)
        targ_g = targ_g.unsqueeze(0)
        mask_g = mask_g.unsqueeze(0)

    mask_g = mask_g.to(dtype=pred_g.dtype)

    a_00 = torch.sum(mask_g * pred_g * pred_g, dim=(1, 2, 3))
    a_01 = torch.sum(mask_g * pred_g, dim=(1, 2, 3))         
    a_11 = torch.sum(mask_g, dim=(1, 2, 3))                  
    b_0  = torch.sum(mask_g * pred_g * targ_g, dim=(1, 2, 3))
    b_1  = torch.sum(mask_g * targ_g, dim=(1, 2, 3))         

    if fit_shift:
        det = a_00 * a_11 - a_01 * a_01
        det = det + eps
        scale = torch.zeros_like(b_0)
        shift = torch.zeros_like(b_1)
        valid = det > 0
        scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
        return scale, shift
    else:
        denom = a_00 + eps                       
        scale = b_0 / denom                      
        shift = torch.zeros_like(scale)          
        return scale, shift


def compute_shift(pred, targ, mask, eps = 1e-6):
    if pred.shape[0] == 6:
        pred = pred.view(1, 6, *pred.shape[2:])
        targ = targ.view(1, 6, *targ.shape[2:])
        mask     = mask.view(1, 6, *mask.shape[2:])

    w = mask.float()
    num = torch.sum(w * (targ - pred), dim=(1,2,3))            
    den = torch.sum(w, dim=(1,2,3)).clamp_min(eps)             
    beta = num / den                                           
    return beta


def get_positional_encoding(H, W, pixel_center = True, hw = 96):
    jj = np.arange(W, dtype=np.float64)
    ii = np.arange(H, dtype=np.float64)
    if pixel_center:
        jj = jj + 0.5
        ii = ii + 0.5

    U = (jj / W) * 2.0 - 1.0 
    V = (ii / H) * 2.0 - 1.0 
    U, V = np.meshgrid(U, V, indexing='xy')      

    erp = np.stack([U, V], axis=-1)              

    erp_tensor = torch.from_numpy(erp).permute(2, 0, 1).float()
    faces = erp_to_cubemap(erp_tensor, face_w=hw)
    return faces


def unit_normals(n, eps = 1e-6):
    assert n.dim() >= 3 and n.size(-3) == 3, "normals must have channel=3 at dim -3"
    denom = torch.clamp(torch.linalg.norm(n, dim=-3, keepdim=True), min=eps)
    return n / denom


def _erp_dirs(H, W, device=None, dtype=None):
    u = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W 
    v = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H 
    theta = u * (2.0 * torch.pi) - torch.pi                     
    phi   = (0.5 - v) * torch.pi                                

    theta = theta.view(1, W).expand(H, W)                  
    phi   = phi.view(H, 1).expand(H, W)                    

    cosphi = torch.cos(phi)
    sinphi = torch.sin(phi)
    costhe = torch.cos(theta)
    sinthe = torch.sin(theta)

    x =  cosphi * costhe
    y = sinphi
    z =  -cosphi * sinthe

    dirs = torch.stack([x, y, z], dim=0)
    return dirs


def depth_to_normals_erp(depth, eps = 1e-6):

    assert depth.dim() == 3 and depth.size(0) == 1, "depth must be (B,1,H,W)"
    _, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    dirs = _erp_dirs(H, W, device=device, dtype=dtype)       
    P = depth * dirs                                         

    dtheta = 2.0 * torch.pi / W
    dphi   = torch.pi / H

    P_l = torch.roll(P, shifts=+1, dims=-1)
    P_r = torch.roll(P, shifts=-1, dims=-1)
    dP_dtheta = (P_r - P_l) / (2.0 * dtheta)                     

    P_u = torch.cat([P[:, :1, :],  P[:, :-1, :]], dim=-2) 
    P_d = torch.cat([P[:, 1:, :],  P[:, -1:, :]], dim=-2) 
    dP_dphi = (P_d - P_u) / (2.0 * dphi)                  

    n = torch.cross(dP_dtheta, dP_dphi, dim=0)            
    n = unit_normals(n, eps=eps)

    return n


def compute_edge_mask(depth, abs_thresh = 0.1, rel_thresh = 0.1):
    assert depth.ndim == 2
    depth = depth.astype(np.float32, copy=False)

    valid = depth > 0
    eps = 1e-6

    edge = np.zeros_like(valid, dtype=bool)

    d1 = depth[:, :-1]
    d2 = depth[:, 1:]
    v_pair = valid[:, :-1] & valid[:, 1:]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)

    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:, :-1] |= edge_pair
    edge[:, 1:] |= edge_pair

    d1 = depth[:-1, :]
    d2 = depth[1:, :]
    v_pair = valid[:-1, :] & valid[1:, :]

    diff = np.abs(d1 - d2)
    rel = diff / (np.minimum(d1, d2) + eps)

    edge_pair = v_pair & (diff > abs_thresh) & (rel > rel_thresh)

    edge[:-1, :] |= edge_pair
    edge[1:, :]  |= edge_pair

    keep = valid & (~edge)
    return keep


def erp_to_pointcloud(rgb, depth, mask = None):
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, "rgb must be (H, W, 3)"
    assert depth.ndim == 2 and depth.shape[:2] == rgb.shape[:2], "depth must be (H, W) and match rgb H,W"

    H, W, _ = rgb.shape

    depth = depth.astype(np.float32, copy=False)

    u = (np.arange(W, dtype=np.float32) + 0.5) / W          
    v = (np.arange(H, dtype=np.float32) + 0.5) / H          

    theta = u * (2.0 * np.pi) - np.pi                       
    phi   = (1 - v) * np.pi - (np.pi / 2.0)                 

    theta, phi = np.meshgrid(theta, phi, indexing="xy")     

    cos_phi = np.cos(phi)
    dir_x = cos_phi * np.cos(theta)
    dir_y = np.sin(phi)
    dir_z = cos_phi * np.sin(theta)

    X = depth * dir_x
    Y = depth * dir_y
    Z = depth * dir_z

    if mask is None:
        keep = depth > 0
    else:
        keep = (mask.astype(bool)) & (depth > 0)

    points = np.stack([X, Y, Z], axis=-1)[keep] 

    rgb_clamped = np.clip(rgb, -1.0, 1.0)
    colors = ((rgb_clamped * 0.5 + 0.5) * 255.0).astype(np.uint8)
    colors = colors.reshape(H, W, 3)[keep]                  

    return points.astype(np.float32, copy=False), colors


def erp_to_point_cloud_glb(rgb, depth, mask=None, export_path=None):
    points, colors = erp_to_pointcloud(rgb, depth, mask)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    scene.export(export_path)
    return scene
