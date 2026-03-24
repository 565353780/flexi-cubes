import os
import torch
import trimesh
import numpy as np
from typing import Union, Dict, Optional, Tuple

from flexi_cubes.Method.io import loadMeshFile
from flexi_cubes.Method.sdf import meshToSDF
# from flexi_cubes.Module.flexi_cubes import FlexiCubes
from flexi_cubes.Module.flexi_cubes_sh import FlexiCubes

from camera_control.Method.data import toTensor 
from flexi_cubes.Module.sh_utils import eval_sh, RGB2SH, SH2RGB 



class FCConvertor(object):
    def __init__(self) -> None:
        return
    

    @staticmethod 
    def _queryPtsColors(
        mesh: trimesh.Trimesh, 
        query_uvs: torch.Tensor
    ) -> Optional[torch.Tensor]: 
        '''
        输入带颜色的mesh，计算空间中查询点距离最近的颜色 
        '''
        uvs_np = mesh.visual.uv 

        vertices_count = mesh.vertices.shape[0] 
        if uvs_np.shape[0] != vertices_count:
            print(f'[WARN][NVDiffRastRenderer::renderTexture] UV count ({uvs_np.shape[0]}) != vertex count ({vertices_count}), falling back to vertex colors')
            return None 
        else:
            tex_img = None
            mat = mesh.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                tex_img = np.array(mat.baseColorTexture)
            elif hasattr(mat, 'image') and mat.image is not None:
                tex_img = np.array(mat.image)

            if tex_img is not None:
                if len(tex_img.shape) == 2:
                    tex_img = np.stack([tex_img] * 3, axis=-1)
                elif tex_img.shape[-1] == 4:
                    tex_img = tex_img[:, :, :3]
                
                H, W = tex_img.shape[:2]
                texture = toTensor(tex_img, torch.float32, 'cpu') / 255.0
                texture = texture.flip(0)
                uvs = uvs_np

        texture = texture.to(query_uvs.device) 

        uvs_tensor = toTensor(uvs, torch.float32, query_uvs.device) 
        uvs_scaled = uvs_tensor * torch.tensor([W-1, H-1], device=query_uvs.device)
        uvs_int = uvs_scaled.long()  # [V,2], integer pixel coords 
        uvs_int[:,0] = uvs_int[:,0].clamp(0, W-1)
        uvs_int[:,1] = uvs_int[:,1].clamp(0, H-1)

        vertex_color = texture[uvs_int[:,1], uvs_int[:,0], :]  # 注意 row=V, col=U 


        def query(mesh: trimesh.Trimesh, query_uvs: torch.Tensor, vertex_color: torch.Tensor) -> torch.Tensor:
            """
            使用 face_id + uv (barycentric) 从 mesh 顶点颜色计算 query point 颜色

            Args:
                mesh: trimesh 对象
                query_uvs: [N,3] tensor, 每个点 (face_idx, u, v)
                vertex_color: [V,3] tensor, mesh 顶点颜色

            Returns:
                query_colors: [N,3] tensor
            """
            N = query_uvs.shape[0]
            query_colors = torch.zeros((N, 3), device=query_uvs.device)

            # 提取 face_idx, u, v
            face_idx = query_uvs[:,0].long()
            u = query_uvs[:,1]                # [N]
            v = query_uvs[:,2]                # [N]   
            
            # mesh faces
            faces = torch.from_numpy(mesh.faces).to(query_uvs.device)  # [F,3]

            # 顶点索引
            v0_idx = faces[face_idx, 0]  # [N]
            v1_idx = faces[face_idx, 1]
            v2_idx = faces[face_idx, 2]

            # 顶点颜色
            c0 = vertex_color[v0_idx]  # [N,3]
            c1 = vertex_color[v1_idx]
            c2 = vertex_color[v2_idx]

            # barycentric 插值
            w0 = 1.0 - u - v
            w1 = u
            w2 = v
            query_colors = w0.unsqueeze(-1) * c0 + w1.unsqueeze(-1) * c1 + w2.unsqueeze(-1) * c2  # [N,3]

            return query_colors
        
        query_colors = query(mesh, query_uvs, vertex_color)  
        return query_colors 


    @staticmethod
    def createFC(
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 64,
        device: str = 'cuda:0',
        initColor: bool = False,
        sh_deg: int = 2, 
        sh_channel: int = 3
    ) -> Optional[Dict]:
        """
        从三角网格创建FlexiCubes参数

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            mesh: 网格文件路径或trimesh对象，如果为None则随机初始化SDF
            resolution: FlexiCubes分辨率（体素网格分辨率）
            device: 计算设备
            initColor: 输入mesh带颜色且需要优化球谐函数系数 

        Returns:
            dict包含: fc, sdf, deform, weight, x_nx3, cube_fx8, grid_edges
        """
        if isinstance(mesh, str):
            if not os.path.exists(mesh):
                print('[ERROR][FCConvertor::createFC]')
                print('\t mesh file not exist!')
                print('\t mesh_file_path:', mesh)
                return None
            mesh = loadMeshFile(mesh)

        # 创建FlexiCubes对象
        fc = FlexiCubes(device=device)

        # 构建体素网格
        x_nx3, cube_fx8 = fc.construct_voxel_grid(resolution)
        # x_nx3: [N, 3] 网格顶点坐标，范围[-1, 1]
        # cube_fx8: [F, 8] 每个立方体的8个顶点索引

        queryColors = None 
        if mesh is not None:
            sdf_values, grid2mesh_uvs = meshToSDF(mesh, x_nx3) 
            if initColor: 
                queryColors = FCConvertor._queryPtsColors(mesh, grid2mesh_uvs) 

        else:
            # 随机初始化SDF（参考官方示例）
            sdf_values = torch.rand_like(x_nx3[:, 0]) - 0.15
            if initColor: 
                queryColors = torch.rand( (x_nx3.shape[0], 3)).to(device)

        # 创建可学习参数
        sdf = torch.nn.Parameter(sdf_values.clone().detach(), requires_grad=True)

        # deform: 顶点位移，形状与x_nx3相同 [N, 3]
        deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

        ##spherical harmonics . sh  [..., C, (deg + 1) ** 2]
        if queryColors is not None and initColor : 
            C0_SH = RGB2SH(queryColors) ##N*3 
            tmp = torch.zeros((x_nx3.shape[0], sh_channel, (sh_deg + 1) ** 2), device=C0_SH.device) 
            tmp[..., 0] = C0_SH 
            tmp = tmp.reshape(x_nx3.shape[0], -1 )
            sh_coeff = torch.nn.Parameter(tmp, requires_grad=True) 
        else:
            sh_coeff = None 

        # weight: FlexiCubes特有的权重
        # 参考官方示例：weight形状为 [F, 21]
        # beta: [:, :12] - 12个边的插值权重
        # alpha: [:, 12:20] - 8个顶点的权重
        # gamma_f: [:, 20] - 每个立方体的权重
        num_cubes = cube_fx8.shape[0]
        weight = torch.nn.Parameter(
            torch.zeros((num_cubes, 21), dtype=torch.float32, device=device),
            requires_grad=True
        )

        # 获取所有边用于正则化损失（参考官方示例）
        all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
        grid_edges = torch.unique(all_edges, dim=0)
    
        return {
            'fc': fc,
            'sdf': sdf,
            'deform': deform,
            'weight': weight,
            'x_nx3': x_nx3,
            'cube_fx8': cube_fx8,
            'resolution': resolution,
            'grid_edges': grid_edges,
            "sh_channel": sh_channel, 
            "sh_deg": sh_deg,
            "sh_coeff": sh_coeff, 
        }

    @staticmethod
    def extractMesh(
        fc_params: Dict,
        training: bool = True,
    ) -> Tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从FlexiCubes参数提取三角网格

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            fc_params: createFC返回的参数字典
            training: 是否处于训练模式（影响L_dev的计算）

        Returns:
            tuple: (mesh, vertices, faces, L_dev)
                - mesh: trimesh.Trimesh对象
                - vertices: [V, 3] 顶点tensor（保持梯度）
                - L_dev: developability损失
        """
        fc = fc_params['fc']
        sdf = fc_params['sdf']
        deform = fc_params['deform']
        weight = fc_params['weight']
        x_nx3 = fc_params['x_nx3']
        cube_fx8 = fc_params['cube_fx8']
        resolution = fc_params['resolution']

        sh_coeff = fc_params['sh_coeff']  ## zzh 
        sh_channel = fc_params['sh_channel']  ## zzh 
        sh_deg = fc_params['sh_deg']  ## zzh 

        # 检查输入参数是否包含NaN或Inf，创建清理后的副本用于提取
        sdf_clean = sdf.clone()
        if torch.isnan(sdf_clean).any() or torch.isinf(sdf_clean).any():
            # 如果SDF包含NaN或Inf，使用裁剪后的值
            sdf_clean = torch.clamp(sdf_clean, -10.0, 10.0)
            sdf_clean = torch.where(torch.isnan(sdf_clean), torch.zeros_like(sdf_clean), sdf_clean)
            sdf_clean = torch.where(torch.isinf(sdf_clean), torch.zeros_like(sdf_clean), sdf_clean)
        else:
            # 即使没有NaN/Inf，也进行裁剪以防止极端值
            sdf_clean = torch.clamp(sdf_clean, -10.0, 10.0)

        deform_clean = deform.clone()
        if torch.isnan(deform_clean).any() or torch.isinf(deform_clean).any():
            # 如果deform包含NaN或Inf，设置为0
            deform_clean = torch.where(torch.isnan(deform_clean), torch.zeros_like(deform_clean), deform_clean)
            deform_clean = torch.where(torch.isinf(deform_clean), torch.zeros_like(deform_clean), deform_clean)

        weight_clean = weight.clone()
        if torch.isnan(weight_clean).any() or torch.isinf(weight_clean).any():
            # 如果weight包含NaN或Inf，设置为0
            weight_clean = torch.where(torch.isnan(weight_clean), torch.zeros_like(weight_clean), weight_clean)
            weight_clean = torch.where(torch.isinf(weight_clean), torch.zeros_like(weight_clean), weight_clean)

        # 应用变形（参考官方示例：使用tanh限制变形范围）
        # 变形范围限制为网格单元大小的一半
        max_deform = (1.0 - 1e-8) / (resolution * 2)
        grid_verts = x_nx3 + max_deform * torch.tanh(deform_clean)

        # 使用FlexiCubes提取mesh
        # 参考官方API：
        # beta: [F, 12] - 12条边的插值权重
        # alpha: [F, 8] - 8个顶点的权重
        # gamma_f: [F] - 每个立方体的权重
        try:
            vertices, faces, L_dev, verts_sh_coeff = fc(
                grid_verts,           # voxelgrid_vertices
                sdf_clean,            # scalar_field (使用清理后的SDF)
                cube_fx8,             # cube_idx
                resolution,           # resolution
                beta=weight_clean[:, :12],        # 12条边的插值权重
                alpha=weight_clean[:, 12:20],     # 8个顶点的权重
                gamma_f=weight_clean[:, 20],      # 每个立方体的权重（注意是1D）
                training=training,
                voxelgrid_features=sh_coeff
            )   ## sh_coeff -> return coeff of mesh vertices
            if verts_sh_coeff is not None: 
                verts_sh_coeff = verts_sh_coeff.reshape (-1, sh_channel, (sh_deg + 1) ** 2 )

            # 检查提取的顶点和面片是否有效
            if vertices is None or faces is None:
                raise ValueError("FlexiCubes returned None vertices or faces")

            if len(vertices) == 0 or len(faces) == 0:
                raise ValueError("FlexiCubes returned empty mesh")

            # 检查是否包含NaN或Inf
            if torch.isnan(vertices).any() or torch.isinf(vertices).any():
                raise ValueError("Extracted vertices contain NaN or Inf")

            mesh = trimesh.Trimesh(
                vertices=vertices.detach().cpu().numpy(),
                faces=faces.cpu().numpy(), 
                process=False  # disable mesh fixing 
            )
            trimesh.repair.fix_winding(mesh) 


        except Exception as e:
            # 如果提取失败，返回一个空的网格
            print(f'[WARNING][FCConvertor::extractMesh] Failed to extract mesh: {e}')
            # 创建一个最小的空网格（使用正确的形状）
            device = sdf.device
            empty_vertices = torch.zeros((0, 3), device=device, dtype=torch.float32)
            # 创建空网格时使用正确的numpy数组形状
            empty_vertices_np = np.zeros((0, 3), dtype=np.float32)
            empty_faces_np = np.zeros((0, 3), dtype=np.int32)
            empty_mesh = trimesh.Trimesh(vertices=empty_vertices_np, faces=empty_faces_np)
            empty_L_dev = torch.tensor(0.0, device=device, dtype=torch.float32)
            empty_sdf = torch.tensor(0.0, device=device, dtype=torch.float32)
            return empty_mesh, empty_vertices, empty_L_dev, empty_sdf

        return mesh, vertices, L_dev, sdf_clean, verts_sh_coeff
