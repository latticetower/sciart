import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.path import Path
import numpy as np


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

smooth_vertices = np.asarray([
    [33., 32.],
    [36., 34.],
    [38., 34.],
    [40., 34.],
    [40., 32.],
    [38., 32.],
    [38., 30.],
    [38., 28.],
    [46., 26.],
    [50., 22.],
    [56., 14.],
    [42., 16.],
    [38., 22.],
    [34., 20.],
    [30., 15.],
    [30., 10.],
    [26., 10.],
    [24., 10.],
    [24., 10.],
    [22., 12.],
    [16., 14.],
    [25., 16.],
    [26., 16.],
    [30., 18.],
    [30., 20.],
    [26., 22.],
    [20., 22.],
    [12., 22.],
    [20., 28.],
    [20., 24.],
    [30., 28.],
    [30., 28.],
    [30., 28.],
    [33., 32.],
    [33., 32.]
]) -[32, 10]

smooth_codes = [ 
    1,  4,  4,  4,  4,  4,  4,  2,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  2,  2,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,
    79
]



def get_goose_markers(xoffset=-0.5, yoffset=0) -> tuple[mpath.Path, mpath.Path]:
    #vertices = [(-y, x) for (x, y) in plain_vertices]
    vertices = [(y, x) for x, y in smooth_vertices]
    marker = mpath.Path(vertices, smooth_codes)
    marker_shifted = mpath.Path(
        [(x + xoffset, y + yoffset) for (x, y) in vertices], 
        smooth_codes
    )
    return marker, marker_shifted
