import numpy as np
from matplotlib.path import Path
from pathlib import Path as FilePath

_DATA_PATH = FilePath(__file__).parent / 'silhouette_data.npz'
_data = np.load(_DATA_PATH)
FATNESS_KEYS = _data['fatness_keys']
LEFT_PROFILES = _data['left_profiles']
RIGHT_PROFILES = _data['right_profiles']
N_Y = LEFT_PROFILES.shape[1]


def make_silhouette(fatness):
    '''Interpolate between keyframe silhouettes. Returns (x_coords, y_coords) as a closed polygon.'''
    fatness = np.clip(fatness, 0.0, 1.0)

    if fatness <= FATNESS_KEYS[0]:
        left, right = LEFT_PROFILES[0], RIGHT_PROFILES[0]
    elif fatness >= FATNESS_KEYS[-1]:
        left, right = LEFT_PROFILES[-1], RIGHT_PROFILES[-1]
    else:
        idx = np.searchsorted(FATNESS_KEYS, fatness) - 1
        t = (fatness - FATNESS_KEYS[idx]) / (FATNESS_KEYS[idx + 1] - FATNESS_KEYS[idx])
        left = LEFT_PROFILES[idx] * (1 - t) + LEFT_PROFILES[idx + 1] * t
        right = RIGHT_PROFILES[idx] * (1 - t) + RIGHT_PROFILES[idx + 1] * t

    y_vals = np.linspace(0, 1, N_Y)
    outline_x = np.concatenate([right[::-1], left])
    outline_y = np.concatenate([y_vals[::-1], y_vals])
    return outline_x, outline_y


def make_silhouette_path(fatness, x_center=0.0, y_base=0.0, scale=1.0, height=1.0):
    '''Generate a closed matplotlib Path for a side-profile silhouette.'''
    outline_x, outline_y = make_silhouette(fatness)
    pts = np.column_stack([outline_x * scale + x_center, outline_y * height + y_base])

    n = len(pts)
    codes = [Path.LINETO] * n
    codes[0] = Path.MOVETO
    codes.append(Path.CLOSEPOLY)
    pts = np.vstack([pts, pts[0]])

    return Path(pts, codes)
