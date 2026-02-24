import numpy as np
from matplotlib.path import Path

from dfmp.silhouette import make_silhouette, make_silhouette_path, FATNESS_KEYS, N_Y


class TestMakeSilhouette:
    def test_returns_two_arrays(self):
        x, y = make_silhouette(0.5)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_output_length(self):
        x, y = make_silhouette(0.5)
        assert len(x) == 2 * N_Y
        assert len(y) == 2 * N_Y

    def test_fatness_zero_returns_first_profile(self):
        x0, y0 = make_silhouette(0.0)
        x0b, y0b = make_silhouette(0.0)
        np.testing.assert_array_equal(x0, x0b)

    def test_fatness_one_returns_last_profile(self):
        x1, y1 = make_silhouette(1.0)
        x1b, y1b = make_silhouette(1.0)
        np.testing.assert_array_equal(x1, x1b)

    def test_mid_fatness_interpolates(self):
        x0, _ = make_silhouette(0.0)
        x1, _ = make_silhouette(1.0)
        x_mid, _ = make_silhouette(0.5)
        assert not np.allclose(x_mid, x0)
        assert not np.allclose(x_mid, x1)

    def test_y_coordinates_span_zero_to_one(self):
        _, y = make_silhouette(0.5)
        assert np.isclose(y.min(), 0.0)
        assert np.isclose(y.max(), 1.0)

    def test_fatness_below_zero_clamps(self):
        x_neg, y_neg = make_silhouette(-0.5)
        x_zero, y_zero = make_silhouette(0.0)
        np.testing.assert_array_equal(x_neg, x_zero)
        np.testing.assert_array_equal(y_neg, y_zero)

    def test_fatness_above_one_clamps(self):
        x_over, y_over = make_silhouette(1.5)
        x_one, y_one = make_silhouette(1.0)
        np.testing.assert_array_equal(x_over, x_one)
        np.testing.assert_array_equal(y_over, y_one)

    def test_exact_keyframe_boundary(self):
        x, y = make_silhouette(FATNESS_KEYS[2])
        assert len(x) == 2 * N_Y


class TestMakeSilhouettePath:
    def test_returns_path(self):
        path = make_silhouette_path(0.5)
        assert isinstance(path, Path)

    def test_vertex_count(self):
        path = make_silhouette_path(0.5)
        assert len(path.vertices) == 2 * N_Y + 1

    def test_path_codes(self):
        path = make_silhouette_path(0.5)
        assert path.codes[0] == Path.MOVETO
        assert path.codes[-1] == Path.CLOSEPOLY
        assert all(c == Path.LINETO for c in path.codes[1:-1])

    def test_x_center_shifts_x(self):
        path_base = make_silhouette_path(0.5, x_center=0.0)
        path_shifted = make_silhouette_path(0.5, x_center=5.0)
        np.testing.assert_allclose(path_shifted.vertices[:, 0] - path_base.vertices[:, 0], 5.0)

    def test_y_base_shifts_y(self):
        path_base = make_silhouette_path(0.5, y_base=0.0)
        path_shifted = make_silhouette_path(0.5, y_base=3.0)
        np.testing.assert_allclose(path_shifted.vertices[:, 1] - path_base.vertices[:, 1], 3.0)

    def test_scale_multiplies_x(self):
        path_1 = make_silhouette_path(0.5, scale=1.0)
        path_2 = make_silhouette_path(0.5, scale=2.0)
        np.testing.assert_allclose(path_2.vertices[:, 0], path_1.vertices[:, 0] * 2.0)

    def test_height_multiplies_y(self):
        path_1 = make_silhouette_path(0.5, height=1.0)
        path_2 = make_silhouette_path(0.5, height=3.0)
        np.testing.assert_allclose(path_2.vertices[:, 1], path_1.vertices[:, 1] * 3.0)
