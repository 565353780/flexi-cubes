import torch
import trimesh
import warp as wp
import numpy as np


# 1. 初始化 Warp
wp.init()

@wp.kernel
def compute_sdf_kernel(
    mesh: wp.uint64,                 # Mesh 句柄
    query_points: wp.array(dtype=wp.vec3),
    out_sdf: wp.array(dtype=float),
    out_gradients: wp.array(dtype=wp.vec3)  # 可选：SDF 梯度（即方向）
):
    tid = wp.tid()
    p = query_points[tid]

    # max_dist 设置为一个足够大的数
    # MeshQueryPoint 包含: result, face, u, v, sign
    query_res = wp.mesh_query_point(mesh, p, 1.0e6)

    if query_res.result:
        # 使用重心坐标计算最近点位置
        face_idx = query_res.face
        u = query_res.u
        v = query_res.v
        closest_p = wp.mesh_eval_position(mesh, face_idx, u, v)

        # 计算距离 (Unsigned)
        dist = wp.length(p - closest_p)

        # 使用 query_res.sign 直接获取符号（正数表示外部，负数表示内部）
        sdf_val = query_res.sign * dist

        out_sdf[tid] = sdf_val

        # 计算方向向量
        diff = p - closest_p

        # 计算梯度（SDF 的导数就是指向表面的单位向量）
        if dist > 1e-6:
            out_gradients[tid] = wp.normalize(diff)
        else:
            # 在表面上直接使用法线
            normal = wp.mesh_eval_face_normal(mesh, face_idx)
            out_gradients[tid] = normal

    else:
        # 如果超出 max_dist
        out_sdf[tid] = 1.0e6
        out_gradients[tid] = wp.vec3(0.0, 0.0, 0.0)

def compute_gt_sdf_batch(wp_mesh, points: torch.Tensor) -> torch.Tensor:
    """使用 Warp 计算 GT SDF。

    Args:
        points: [N, 3] 或 [..., 3] 采样点坐标。

    Returns:
        sdf: [N] 或 [...] GT SDF 值。
    """
    original_shape = points.shape[:-1]
    points_flat = points.view(-1, 3)
    num_points = points_flat.shape[0]

    # 准备查询点
    points_np = points_flat.detach().cpu().numpy().astype(np.float32)

    # 创建 Warp 数组
    wp_points = wp.array(points_np, dtype=wp.vec3)
    wp_sdf = wp.zeros(num_points, dtype=float)
    wp_gradients = wp.zeros(num_points, dtype=wp.vec3)

    # 运行 kernel
    wp.launch(
        kernel=compute_sdf_kernel,
        dim=num_points,
        inputs=[wp_mesh.id, wp_points, wp_sdf, wp_gradients]
    )

    # 同步并转换回 PyTorch
    wp.synchronize()
    sdf_np = wp_sdf.numpy()
    sdf = torch.from_numpy(sdf_np).float().to(points.device)

    # 恢复原始形状
    sdf = sdf.view(*original_shape)
    return sdf

def meshToSDF(
    mesh: trimesh.Trimesh,
    query_points: torch.Tensor,
) -> torch.Tensor:
    """
    在指定查询点计算mesh的SDF值（使用trimesh的精确计算）

    Args:
        mesh: trimesh.Trimesh对象
        query_points: [N, 3] 查询点坐标（可以是任意坐标范围）
        device: 计算设备

    Returns:
        sdf: [N] SDF值（内部为负，外部为正）
    """
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32).flatten()

    wp_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        indices=wp.array(faces, dtype=int)
    )

    return compute_gt_sdf_batch(wp_mesh, query_points)
