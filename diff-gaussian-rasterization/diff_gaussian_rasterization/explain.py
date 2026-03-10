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

from typing import NamedTuple              # Used to define an immutable, typed settings container
import torch.nn as nn                      # PyTorch neural network module base class
import torch                               # Core PyTorch tensor + autograd library
from . import _C                           # Loads the compiled extension module (C/C++)  <-- C/C++


def cpu_deep_copy_tuple(input_tuple):
    # Utility: make a CPU+cloned copy of any tensors in a tuple (useful for debug/saving)
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    # Return a new tuple with CPU copies of tensors, and untouched non-tensors
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,            # (N, 3) 3D Gaussian centers in world space
    means2D,            # (N, 2) optional/aux 2D means buffer (often for grad tricks / debug)
    sh,                 # (N, ... ) spherical harmonics coeffs (if using SH shading); else empty
    colors_precomp,     # (N, 3) precomputed RGB colors (if NOT using SH); else empty
    opacities,          # (N, 1) or (N,) alpha/opacity per Gaussian
    scales,             # (N, 3) per-axis scale (if using scale+rotation param); else empty
    rotations,          # (N, 4) quaternion (or similar) rotation (if using scale+rotation); else empty
    cov3Ds_precomp,     # (N, 3, 3) precomputed covariance (if NOT using scale+rotation); else empty
    raster_settings,    # GaussianRasterizationSettings: camera, image size, flags, etc.
):
    # Calls the custom autograd Function so PyTorch knows how to backprop through rasterization
    return _RasterizeGaussians.apply(
        means3D,         # input 3D means
        means2D,         # input 2D means (often not used by forward but can carry gradients)
        sh,              # SH coefficients or empty tensor
        colors_precomp,  # precomputed colors or empty tensor
        opacities,       # opacity per Gaussian
        scales,          # scale parameters or empty tensor
        rotations,       # rotation parameters or empty tensor
        cov3Ds_precomp,  # covariance matrix or empty tensor
        raster_settings, # packed config
    )


class _RasterizeGaussians(torch.autograd.Function):
    """
    Custom autograd operator:
      - forward(): runs the rasterizer and returns rendered outputs
      - backward(): computes gradients w.r.t. Gaussian parameters
    """

    @staticmethod
    def forward(
        ctx,                 # context object used to stash tensors/metadata for backward
        means3D,             # see above
        means2D,             # see above
        sh,                  # see above
        colors_precomp,      # see above
        opacities,           # see above
        scales,              # see above
        rotations,           # see above
        cov3Ds_precomp,      # see above
        raster_settings,     # immutable settings (camera/image/flags)
    ):

        # Pack arguments EXACTLY in the order the extension expects (contract/interface boundary)
        args = (
            raster_settings.bg,             # background color (tensor) to composite with
            means3D,                        # 3D positions
            colors_precomp,                 # RGB colors if using precomputed colors
            opacities,                      # alpha/opacity
            scales,                         # scale params (if used)
            rotations,                      # rotation params (if used)
            raster_settings.scale_modifier, # global multiplier on scale (used for training tweaks)
            cov3Ds_precomp,                 # covariance matrices (if used)
            raster_settings.viewmatrix,     # camera view matrix
            raster_settings.projmatrix,     # camera projection matrix
            raster_settings.tanfovx,        # tan(fov_x/2) to help screen projection math
            raster_settings.tanfovy,        # tan(fov_y/2)
            raster_settings.image_height,   # output image height
            raster_settings.image_width,    # output image width
            sh,                             # SH coeffs if using SH shading
            raster_settings.sh_degree,      # SH degree (0,1,2,3 typically)
            raster_settings.campos,         # camera position in world (for view-dependent SH)
            raster_settings.prefiltered,    # flag: indicates prefiltering mode/behavior
            raster_settings.antialiasing,   # flag: whether to use AA mode
            raster_settings.debug           # flag: enable debug behaviors/buffers
        )

        # Call the high-performance rasterizer implementation (does actual rendering)  <-- C/C++
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)  # C/C++

        # Stash settings needed for backward (non-tensor metadata)
        ctx.raster_settings = raster_settings

        # Stash how many Gaussians contributed (useful for backward and debugging)
        ctx.num_rendered = num_rendered

        # Save tensors that backward needs (PyTorch will keep them alive)
        # These typically include:
        #  - inputs needed to recompute partials
        #  - intermediate buffers produced by forward to avoid recomputation
        ctx.save_for_backward(
            colors_precomp,   # used if colors are precomputed
            means3D,          # positions needed for gradients
            scales,           # scale params needed for gradients
            rotations,        # rotation params needed for gradients
            cov3Ds_precomp,   # covariance if used
            radii,            # screen-space radii/extent per Gaussian (from forward)
            sh,               # SH coeffs (if used)
            opacities,        # alpha values
            geomBuffer,       # intermediate geometry buffer (accelerates backward)
            binningBuffer,    # intermediate tile/bin buffer (accelerates backward)
            imgBuffer         # intermediate image buffer (accelerates backward)
        )

        # Return outputs to the caller:
        #  - color: rendered image (H, W, 3) or (3, H, W) depending on implementation
        #  - radii: per-Gaussian screen radius/extent (useful for pruning/visibility)
        #  - invdepths: inverse depth image or per-pixel inverse depth (for depth losses)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        """
        Backprop signature matches forward outputs:
          forward returned (color, radii, invdepths)
          backward receives grads for those outputs:
            grad_out_color for color
            _ for radii (ignored here)
            grad_out_depth for invdepths
        """

        # Restore metadata saved in forward
        num_rendered = ctx.num_rendered              # number of effective rendered Gaussians
        raster_settings = ctx.raster_settings        # immutable settings for consistent backward

        # Restore saved tensors from forward
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
         radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        # Pack arguments EXACTLY in the order expected by the backward extension
        args = (
            raster_settings.bg,              # background (needed for correct compositing grads)
            means3D,                         # 3D positions
            radii,                           # per-Gaussian screen extent
            colors_precomp,                  # precomputed colors (if used)
            opacities,                       # opacities
            scales,                          # scales (if used)
            rotations,                       # rotations (if used)
            raster_settings.scale_modifier,  # same modifier as forward
            cov3Ds_precomp,                  # covariance (if used)
            raster_settings.viewmatrix,      # camera view matrix
            raster_settings.projmatrix,      # camera projection matrix
            raster_settings.tanfovx,         # fov helper
            raster_settings.tanfovy,         # fov helper
            grad_out_color,                  # upstream gradient dL/d(color)
            grad_out_depth,                  # upstream gradient dL/d(invdepth)
            sh,                              # SH coeffs
            raster_settings.sh_degree,       # SH degree
            raster_settings.campos,          # camera position
            geomBuffer,                      # forward-produced intermediates
            num_rendered,                    # forward-produced scalar/meta
            binningBuffer,                   # intermediates
            imgBuffer,                       # intermediates
            raster_settings.antialiasing,    # AA flag (must match forward)
            raster_settings.debug            # debug flag (must match forward)
        )

        # Call extension backward to compute gradients wrt inputs  <-- C/C++
        (grad_means2D,
         grad_colors_precomp,
         grad_opacities,
         grad_means3D,
         grad_cov3Ds_precomp,
         grad_sh,
         grad_scales,
         grad_rotations) = _C.rasterize_gaussians_backward(*args)  # C/C++

        # Return gradients in the exact order of forward() inputs:
        grads = (
            grad_means3D,           # gradient for means3D
            grad_means2D,           # gradient for means2D
            grad_sh,                # gradient for SH coeffs
            grad_colors_precomp,    # gradient for precomputed colors
            grad_opacities,         # gradient for opacities
            grad_scales,            # gradient for scales
            grad_rotations,         # gradient for rotations
            grad_cov3Ds_precomp,    # gradient for covariance
            None,                   # raster_settings is not a tensor -> no gradient
        )

        # PyTorch uses this to populate .grad for each input tensor
        return grads


class GaussianRasterizationSettings(NamedTuple):
    """
    Immutable settings blob passed into rasterizer.
    Using NamedTuple makes it:
      - lightweight
      - easy to access with dot notation
      - safe from accidental mutation inside autograd
    """
    image_height: int             # output height in pixels
    image_width: int              # output width in pixels
    tanfovx: float                # tan(fov_x/2)
    tanfovy: float                # tan(fov_y/2)
    bg: torch.Tensor              # background color tensor
    scale_modifier: float         # global scale multiplier
    viewmatrix: torch.Tensor      # camera view matrix
    projmatrix: torch.Tensor      # camera projection matrix
    sh_degree: int                # SH degree
    campos: torch.Tensor          # camera position in world space
    prefiltered: bool             # prefiltering flag (quality/perf behavior)
    debug: bool                   # debug flag
    antialiasing: bool            # AA flag


class GaussianRasterizer(nn.Module):
    """
    nn.Module wrapper so it can be used like a normal PyTorch layer:
        rasterizer = GaussianRasterizer(settings)
        color, radii, invdepth = rasterizer(...)
    """
    def __init__(self, raster_settings):
        super().__init__()                 # initialize nn.Module base
        self.raster_settings = raster_settings  # store the immutable settings

    def markVisible(self, positions):
        # Returns a boolean mask of which Gaussians are within the camera frustum.
        # Runs without gradients because visibility is typically used for pruning/filtering.
        with torch.no_grad():
            raster_settings = self.raster_settings

            # Call extension function that does frustum culling / visibility test  <-- C/C++
            visible = _C.mark_visible(     # C/C++
                positions,                # (N,3) positions to test
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )

        return visible                     # (N,) bool tensor mask

    def forward(
        self,
        means3D,                 # (N,3)
        means2D,                 # (N,2) or buffer tensor
        opacities,               # (N,1) or (N,)
        shs=None,                # optional SH coeffs
        colors_precomp=None,     # optional precomputed RGB
        scales=None,             # optional scale
        rotations=None,          # optional rotation
        cov3D_precomp=None       # optional covariance
    ):

        raster_settings = self.raster_settings  # local alias for convenience

        # Enforce: you must provide exactly ONE of (shs, colors_precomp)
        #  - SH: view-dependent color
        #  - colors_precomp: fixed RGB per Gaussian
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # Enforce: you must provide exactly ONE of:
        #  - (scales AND rotations)  (parameterized covariance)
        #  - cov3D_precomp          (explicit covariance)
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Convert missing optional inputs into empty tensors.
        # This keeps the call signature fixed for the extension (it expects tensors).
        if shs is None:
            shs = torch.Tensor([])               # empty: signals "not using SH"
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])    # empty: signals "not using precomputed colors"

        if scales is None:
            scales = torch.Tensor([])            # empty: signals "not using scale param"
        if rotations is None:
            rotations = torch.Tensor([])         # empty: signals "not using rotation param"
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])     # empty: signals "not using covariance matrix"

        # Call the differentiable rasterizer operator (autograd-enabled)
        return rasterize_gaussians(
            means3D,             # 3D Gaussian centers
            means2D,             # 2D buffer (can receive gradients)
            shs,                 # SH coeffs or empty
            colors_precomp,      # colors or empty
            opacities,           # alpha
            scales,              # scales or empty
            rotations,           # rotations or empty
            cov3D_precomp,       # covariance or empty
            raster_settings,     # camera + rendering settings
        )