def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Check if :class:`torch.Tensor` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the tensor to be tested.
        shape (list or tuple of int, optional): the expected shape,
            if a dimension is set at ``None`` then it's not verified.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if true (default), will throw if checks fail

    Return:
        (bool) True if checks pass
    """
    if shape is not None:
        if len(shape) != tensor.ndim:
            if throw:
                raise ValueError(f"tensor have {tensor.ndim} ndim, should have {len(shape)}")
            return False
        for i, dim in enumerate(shape):
            if dim is not None and tensor.shape[i] != dim:
                if throw:
                    raise ValueError(f"tensor shape is {tensor.shape}, should be {shape}")
                return False
    if dtype is not None and dtype != tensor.dtype:
        if throw:
            raise TypeError(f"tensor dtype is {tensor.dtype}, should be {dtype}")
        return False
    if device is not None and device != tensor.device.type:
        if throw:
            raise TypeError(f"tensor device is {tensor.device.type}, should be {device}")
        return False
    return True
