import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def get_projection_matrix(znear, zfar, fovx, fovy, device):
    P = torch.zeros((4, 4), device=device, dtype=torch.float32)
    P[0, 0] = 1.0 / torch.tan(fovx * 0.5)
    P[1, 1] = 1.0 / torch.tan(fovy * 0.5)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def rasterize_gaussians_single(
    means3D, scales, rotations, opacities, colors,
    pose_c2w, K, H, W,
    znear=0.01, zfar=100.0,
):
    device = means3D.device
    pose_c2w = pose_c2w.float()
    K = K.float()

    w2c = torch.inverse(pose_c2w)
    campos = pose_c2w[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    fovx = 2.0 * torch.atan((W * 0.5) / fx)
    fovy = 2.0 * torch.atan((H * 0.5) / fy)
    tanfovx = torch.tan(fovx * 0.5)
    tanfovy = torch.tan(fovy * 0.5)

    P = get_projection_matrix(znear, zfar, fovx, fovy, device)
    viewmatrix = w2c.transpose(0, 1).contiguous()
    projmatrix = (P @ w2c).transpose(0, 1).contiguous()

    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0.0, 0.0, 0.0], device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    
    )

    rasterizer = GaussianRasterizer(settings)
    means2D = torch.zeros_like(means3D)

    try:
        rendered, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return rendered
    except Exception as e:
        print("Rasterizer failure:", e)
        return None

        