import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as mpatches

from dfmp import plot
from dfmp.chart import _format_value
from dfmp.config import DEFAULT_COLOR


class TestFormatValue:
    def test_integer_valued_float(self):
        assert _format_value(10.0, None) == '10'

    def test_non_integer_float(self):
        assert _format_value(3.14, None) == '3.14'

    def test_format_string(self):
        assert _format_value(3.14, '{:.1f}%') == '3.1%'

    def test_callable(self):
        assert _format_value(42, lambda v: f'${v}') == '$42'


class TestPlot:
    def test_returns_axes(self, sample_data):
        ax = plot(*sample_data)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close('all')

    def test_string_x_tick_labels(self, sample_data):
        ax = plot(*sample_data)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ['A', 'B', 'C']
        plt.close('all')

    def test_tick_label_overrides(self, numeric_data):
        ax = plot(*numeric_data, tick_label=['X', 'Y', 'Z'])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ['X', 'Y', 'Z']
        plt.close('all')

    def test_default_color(self, sample_data):
        ax = plot(*sample_data)
        for p in ax.patches:
            if isinstance(p, mpatches.PathPatch):
                assert matplotlib.colors.to_hex(p.get_facecolor()) == DEFAULT_COLOR.lower()
        plt.close('all')

    def test_single_color_string(self, sample_data):
        ax = plot(*sample_data, color='red')
        for p in ax.patches:
            if isinstance(p, mpatches.PathPatch):
                assert matplotlib.colors.to_hex(p.get_facecolor()) == matplotlib.colors.to_hex('red')
        plt.close('all')

    def test_color_list(self, sample_data):
        colors = ['red', 'green', 'blue']
        ax = plot(*sample_data, color=colors)
        path_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        for patch, expected in zip(path_patches, colors):
            assert matplotlib.colors.to_hex(patch.get_facecolor()) == matplotlib.colors.to_hex(expected)
        plt.close('all')

    def test_correct_number_of_patches(self, sample_data):
        ax = plot(*sample_data)
        path_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        assert len(path_patches) == 3
        plt.close('all')

    def test_show_values_true(self, sample_data):
        ax = plot(*sample_data, show_values=True)
        texts = [t for t in ax.texts]
        assert len(texts) == 3
        plt.close('all')

    def test_show_values_false(self, sample_data):
        ax = plot(*sample_data, show_values=False)
        assert len(ax.texts) == 0
        plt.close('all')

    def test_ax_none_creates_figure(self, sample_data):
        ax = plot(*sample_data, ax=None)
        assert ax.figure is not None
        plt.close('all')

    def test_existing_ax_reused(self, sample_data, fresh_ax):
        returned = plot(*sample_data, ax=fresh_ax)
        assert returned is fresh_ax

    def test_alpha(self, sample_data):
        ax = plot(*sample_data, alpha=0.5)
        for p in ax.patches:
            if isinstance(p, mpatches.PathPatch):
                assert p.get_alpha() == 0.5
        plt.close('all')

    def test_label_first_patch_only(self, sample_data):
        ax = plot(*sample_data, label='people')
        path_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        assert path_patches[0].get_label() == 'people'
        for p in path_patches[1:]:
            assert p.get_label() is None or p.get_label().startswith('_')
        plt.close('all')

    def test_value_format_string(self, sample_data):
        ax = plot(*sample_data, value_format='{:.1f}%')
        texts = [t.get_text() for t in ax.texts]
        assert '10.0%' in texts
        plt.close('all')

    def test_value_format_callable(self, sample_data):
        ax = plot(*sample_data, value_format=lambda v: f'${int(v)}')
        texts = [t.get_text() for t in ax.texts]
        assert '$10' in texts
        plt.close('all')

    def test_kwargs_passthrough(self, sample_data):
        ax = plot(*sample_data, edgecolor='red')
        for p in ax.patches:
            if isinstance(p, mpatches.PathPatch):
                assert matplotlib.colors.to_hex(p.get_edgecolor()) == matplotlib.colors.to_hex('red')
        plt.close('all')

    def test_all_equal_heights(self):
        ax = plot(['A', 'B', 'C'], [5, 5, 5])
        path_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        assert len(path_patches) == 3
        plt.close('all')

    def test_single_element(self):
        ax = plot(['A'], [42])
        path_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        assert len(path_patches) == 1
        plt.close('all')

    def test_y_axis_hidden(self, sample_data):
        ax = plot(*sample_data)
        assert not ax.yaxis.get_visible()
        plt.close('all')

    def test_spines_hidden(self, sample_data):
        ax = plot(*sample_data)
        assert not ax.spines['left'].get_visible()
        assert not ax.spines['right'].get_visible()
        assert not ax.spines['top'].get_visible()
        plt.close('all')

    def test_aspect_equal(self, sample_data):
        ax = plot(*sample_data)
        assert ax.get_aspect() in ('equal', 1.0)
        plt.close('all')
