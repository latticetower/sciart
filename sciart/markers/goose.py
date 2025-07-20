import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.path import Path


plain_vertices = [
    (0.0, 0.0),    # tail
    (0.5, 1.5),    
    (2.0, 1.5),
    (5.0, 3.0), # right wing
    
    (1, 4),
    (0.5, 5),
    (0.3, 6.5),
    (0.5, 7.0),
    (0.3, 7.5),
    (0, 8.0), # nose
    
    (-0.3, 7.5), 
    (-0.5, 7.0),
    (-0.3, 6.5),
    (-0.5, 5),
    (-1, 4),
    
    (-5.0, 3.0), # left wing
    (-2.0, 1.5),
    (-0.5, 1.5),
    
    (0.0, 0.0)
]

plain_codes = [
    Path.MOVETO,   # tail
    Path.LINETO,
    Path.LINETO,
    Path.LINETO, # right wing
    
    Path.LINETO, 
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO, # nose
    
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    
    Path.LINETO, # left wing
    Path.LINETO, 
    Path.LINETO, 
    
    Path.CLOSEPOLY
]



def get_goose_markers(xoffset=-0.5, yoffset=0) -> tuple[mpath.Path, mpath.Path]:
    vertices = [(-y, x) for (x, y) in plain_vertices]
    marker = mpath.Path(vertices, plain_codes)
    marker_shifted = mpath.Path(
        [(x + xoffset, y + yoffset) for (x, y) in vertices], 
        plain_codes
    )
    return marker, marker_shifted
