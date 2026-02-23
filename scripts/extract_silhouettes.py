'''Extract silhouette outlines from reference images and save as keyframe data.'''

import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
OUTPUT_PATH = Path(__file__).parent.parent / 'pictogram_people' / 'silhouette_data.npz'

REFERENCE_IMAGES = [
    ('example_skinny.png', 10.2),
    ('example_mediumskinny.png', 18.4),
    ('example_medium.png', 26.5),
    ('example_mediumfat.png', 32.2),
    ('example_very_fat.png', 44.2),
]

VAL_MIN, VAL_MAX = 10.2, 44.2
N_Y_SAMPLES = 1000
SMOOTH_SIGMA_BODY = 5
SMOOTH_SIGMA_HEAD = 1.5
HEAD_FRAC = 0.12


def load_and_threshold(image_path):
    '''Load image and create binary mask of the silhouette pixels.'''
    img = np.array(Image.open(image_path).convert('RGB'))
    r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
    brightness = max_rgb / 255.0

    # Silhouette pixels: saturated and not too bright (not white background) and not too dark
    mask = (saturation > 0.15) & (brightness > 0.15) & (brightness < 0.85)
    return mask.astype(np.uint8), img


def clean_mask(mask):
    '''Keep only the largest connected component, fill holes.'''
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n_features = ndimage.label(mask)
    if n_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


def extract_profile(mask):
    '''For each row, find leftmost and rightmost silhouette pixel.'''
    rows_with_pixels = np.where(mask.any(axis=1))[0]
    if len(rows_with_pixels) == 0:
        raise ValueError('No silhouette pixels found')

    y_top, y_bottom = rows_with_pixels[0], rows_with_pixels[-1]
    height = y_bottom - y_top + 1

    left_x = np.zeros(height)
    right_x = np.zeros(height)
    for i, row_idx in enumerate(range(y_top, y_bottom + 1)):
        cols = np.where(mask[row_idx])[0]
        if len(cols) > 0:
            left_x[i] = cols[0]
            right_x[i] = cols[-1]
        else:
            left_x[i] = np.nan
            right_x[i] = np.nan

    # Interpolate any gaps
    valid = ~np.isnan(left_x)
    if not valid.all():
        idxs = np.arange(height)
        left_x = np.interp(idxs, idxs[valid], left_x[valid])
        right_x = np.interp(idxs, idxs[valid], right_x[valid])

    return left_x, right_x


def normalize_profile(left_x, right_x):
    '''Normalize x coords: remove low-frequency lateral drift via polynomial fit
    while preserving local body curves (back, belly, etc.).
    y goes from 0.0 (feet/bottom) to 1.0 (head/top).'''
    height = len(left_x)
    midline = (left_x + right_x) / 2.0
    t = np.linspace(0, 1, height)
    poly_coeffs = np.polyfit(t, midline, 2)
    trend = np.polyval(poly_coeffs, t)

    left_normalized = (left_x - trend) / height
    right_normalized = (right_x - trend) / height

    # Flip vertically so y=0 is feet (bottom of image) and y=1 is head (top)
    left_normalized = left_normalized[::-1]
    right_normalized = right_normalized[::-1]

    return left_normalized, right_normalized


def resample_and_smooth(left_x, right_x, n_samples, sigma_body, sigma_head, head_frac):
    '''Resample profile to n_samples evenly-spaced y values, then apply adaptive
    smoothing — lighter near the head to preserve its round shape.'''
    old_y = np.linspace(0, 1, len(left_x))
    new_y = np.linspace(0, 1, n_samples)
    left_resampled = np.interp(new_y, old_y, left_x)
    right_resampled = np.interp(new_y, old_y, right_x)

    left_body = ndimage.gaussian_filter1d(left_resampled, sigma=sigma_body)
    right_body = ndimage.gaussian_filter1d(right_resampled, sigma=sigma_body)
    left_head = ndimage.gaussian_filter1d(left_resampled, sigma=sigma_head)
    right_head = ndimage.gaussian_filter1d(right_resampled, sigma=sigma_head)

    blend = np.clip((new_y - (1.0 - head_frac)) / (head_frac * 0.5), 0, 1)
    left_smooth = left_body * (1 - blend) + left_head * blend
    right_smooth = right_body * (1 - blend) + right_head * blend
    return left_smooth, right_smooth


def main():
    fatness_keys = []
    left_profiles = []
    right_profiles = []

    for filename, value in REFERENCE_IMAGES:
        image_path = EXAMPLES_DIR / filename
        print(f'Processing {filename} (value={value})...')

        mask, img = load_and_threshold(image_path)
        mask = clean_mask(mask)

        pixel_count = mask.sum()
        print(f'  Silhouette pixels: {pixel_count}')

        left_x, right_x = extract_profile(mask)
        print(f'  Profile height: {len(left_x)}px')

        left_norm, right_norm = normalize_profile(left_x, right_x)
        left_resampled, right_resampled = resample_and_smooth(left_norm, right_norm, N_Y_SAMPLES, SMOOTH_SIGMA_BODY, SMOOTH_SIGMA_HEAD, HEAD_FRAC)

        fatness = (value - VAL_MIN) / (VAL_MAX - VAL_MIN)
        fatness_keys.append(fatness)
        left_profiles.append(left_resampled)
        right_profiles.append(right_resampled)

        print(f'  Fatness key: {fatness:.3f}')
        width_at_belly = right_resampled[N_Y_SAMPLES // 3] - left_resampled[N_Y_SAMPLES // 3]
        print(f'  Width at ~belly: {width_at_belly:.4f}')

    fatness_keys = np.array(fatness_keys)
    left_profiles = np.array(left_profiles)
    right_profiles = np.array(right_profiles)

    np.savez(OUTPUT_PATH, fatness_keys=fatness_keys, left_profiles=left_profiles, right_profiles=right_profiles)
    print(f'\nSaved to {OUTPUT_PATH}')
    print(f'  fatness_keys shape: {fatness_keys.shape}')
    print(f'  left_profiles shape: {left_profiles.shape}')
    print(f'  right_profiles shape: {right_profiles.shape}')


if __name__ == '__main__':
    main()
