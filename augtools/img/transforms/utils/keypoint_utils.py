def crop_keypoint_by_coords(keypoint, crop_coords):  
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1, _, _ = crop_coords
    return x - x1, y - y1, angle, scale