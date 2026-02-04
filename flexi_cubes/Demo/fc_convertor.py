from flexi_cubes.Module.fc_convertor import FCConvertor


def demo():
    mesh_file_path = ""
    resolution = 128
    device = 'cuda:0'

    fc_params = FCConvertor.createFC(mesh_file_path, resolution, device)
    assert fc_params is not None

    mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)
    print(mesh)
    return True
