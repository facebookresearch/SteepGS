#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

# eigensolver is called with a fixed batch to avoid memory issues
# cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling `cusolverDnSsyevjBatched_bufferSize( handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize)`
def eigh_in_batch(As, batch_size=1_000_000, least_k=-1):

    assert As.shape[1] == As.shape[2]

    out_eigvals = []
    out_eigvecs = []

    if least_k == -1:
        least_k = As.shape[-1]

    for A in torch.split(As, batch_size, dim=0):
        eigvals, eigvecs = torch.linalg.eigh(A)
        out_eigvals.append(eigvals[..., :least_k])
        out_eigvecs.append(eigvecs[..., :, :least_k])

    return torch.cat(out_eigvals, 0), torch.cat(out_eigvecs, 0)

# Compute the smallest eigenvalue and corresponding eigenvector of symmetric 3x3 matrices using root formula 
def smallest_eig_3x3_symmetric(A: torch.Tensor):
    assert A.shape[-2:] == (3, 3)

    A_xx = A[:, 0, 0]
    A_yy = A[:, 1, 1]
    A_zz = A[:, 2, 2]
    A_xy = A[:, 0, 1]
    A_xz = A[:, 0, 2]
    A_yz = A[:, 1, 2]

    trace_A = A_xx + A_yy + A_zz
    mean_trace = trace_A / 3

    dev_xx = A_xx - mean_trace
    dev_yy = A_yy - mean_trace
    dev_zz = A_zz - mean_trace

    frob_sq = (
        dev_xx**2 + dev_yy**2 + dev_zz**2 +
        2 * (A_xy**2 + A_xz**2 + A_yz**2)
    )
    p = torch.sqrt(frob_sq / 6)

    I = torch.eye(3, device=A.device).unsqueeze(0)
    B = (1 / p[:, None, None]) * (A - mean_trace[:, None, None] * I)
    det_B = torch.linalg.det(B)
    r = torch.clamp(det_B / 2, -1.0, 1.0)

    angle = torch.acos(r) / 3
    smallest_eigval = mean_trace + 2 * p * torch.cos(angle + (4 * torch.pi / 3))

    shifted_A = A - smallest_eigval[:, None, None] * I
    _, _, V = torch.linalg.svd(shifted_A)
    eigvec = V[:, -1]
    eigvec = eigvec / torch.norm(eigvec, dim=1, keepdim=True)

    return smallest_eigval, eigvec

def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 99
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]

    return np.uint8(colorized[..., ::-1] * 255)


def chamfer_dist(array1, array2):
    dist = torch.norm(array1[None] - array2[:, None], 2, dim=-1)
    return dist.min(1)[0]
