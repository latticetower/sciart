import matplotlib.path as mpath


flower_vertices = [
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0],
    [1, 0],
    [0, 0],
    [0, -1],
    [1, 0],
    [0.3, 0.7],
    [-0.2, 0],
    [0.3, -0.7],
    [1, 0]
]

flower_codes = [
    mpath.Path.MOVETO,
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
    
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
    
    mpath.Path.MOVETO,
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
    mpath.Path.CURVE3,
    mpath.Path.LINETO,
]


def get_flower_markers(xoffset=-0.5, yoffset=0) -> tuple[mpath.Path, mpath.Path]:
    flower_marker = mpath.Path(flower_vertices, flower_codes)
    flower_marker_shifted = mpath.Path(
        [(x + xoffset, y + yoffset) for (x, y) in flower_vertices], 
        flower_codes
    )
    return flower_marker, flower_marker_shifted
