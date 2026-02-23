import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .silhouette import make_silhouette_path
from .config import DEFAULT_COLOR, LABEL_FONTSIZE, VALUE_FONTSIZE, MIN_HEIGHT_FRAC


def plot(x, height, *, color=None, alpha=None, tick_label=None,
         label=None, show_values=True, value_format=None,
         ax=None, **kwargs):
    '''Draw a pictogram people chart where silhouettes are scaled to represent data values.

    Drop-in alternative to ``plt.bar()`` — pass the same ``x`` and ``height``
    and get scaled human silhouettes instead of rectangular bars.

    Parameters
    ----------
    x : array-like of str or numeric
        Category labels (strings) or numeric positions. When strings are
        provided they are used as tick labels automatically.
    height : array-like of float
        Data values that drive both the fatness and the vertical scaling of
        each silhouette.
    color : str or list of str, optional
        Face color(s) for the silhouettes. A single string applies the same
        color to every figure; a list assigns one color per figure. Defaults
        to ``'#5B9A8B'``.
    alpha : float, optional
        Opacity (0–1) applied to every silhouette.
    tick_label : list of str, optional
        Override x-axis labels when *x* is numeric.
    label : str, optional
        Legend label for this set of silhouettes.
    show_values : bool, default ``True``
        Annotate each silhouette with its data value.
    value_format : str or callable, optional
        Format string (e.g. ``'{:.1f}%'``) or callable ``f(val) -> str``
        for value annotations. By default integers display without decimals.
    ax : matplotlib.axes.Axes, optional
        Target axes. A new figure is created when *None*.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the chart, following the seaborn convention.

    Examples
    --------
    Basic usage — drop-in replacement for ``plt.bar()``:

    >>> plot(['A', 'B', 'C'], [10, 20, 30])

    Custom colors and value formatting:

    >>> plot(['X', 'Y'], [3.14, 2.72], color=['#E76F51', '#264653'],
    ...           value_format='{:.1f}')

    Draw on an existing axes:

    >>> fig, ax = plt.subplots()
    >>> plot(['A', 'B'], [5, 10], ax=ax)
    '''
    x = list(x)
    n = len(x)
    values = np.array(height, dtype=float)

    if all(isinstance(xi, str) for xi in x):
        positions = list(range(n))
        tick_labels = x
    else:
        positions = [float(xi) for xi in x]
        tick_labels = tick_label

    if color is None:
        colors = [DEFAULT_COLOR] * n
    elif isinstance(color, str):
        colors = [color] * n
    else:
        colors = list(color)

    val_min, val_max = values.min(), values.max()
    if val_max == val_min:
        normalized = np.full(n, 0.5)
    else:
        normalized = (values - val_min) / (val_max - val_min)

    height_scales = MIN_HEIGHT_FRAC + (1.0 - MIN_HEIGHT_FRAC) * normalized

    spacing = 1.4
    max_silhouette_height = 3.5
    y_base = 0.0

    if ax is None:
        fig_width = max(1.8 * n, 4.0)
        fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    patch_kwargs = dict(edgecolor='none', lw=0, **kwargs)
    if alpha is not None:
        patch_kwargs['alpha'] = alpha

    for i in range(n):
        x_center = positions[i] * spacing
        fatness = normalized[i]
        h = max_silhouette_height * height_scales[i]

        path = make_silhouette_path(fatness, x_center=x_center, y_base=y_base, scale=h, height=h)
        patch = patches.PathPatch(path, facecolor=colors[i], label=label if i == 0 else None, **patch_kwargs)
        ax.add_patch(patch)

        if show_values:
            display_val = _format_value(values[i], value_format)
            ax.text(x_center, y_base + h + 0.08, display_val, ha='center', va='bottom', fontsize=VALUE_FONTSIZE)

    x_coords = [p * spacing for p in positions]
    if tick_labels is not None:
        ax.set_xticks(x_coords, tick_labels, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='x', length=0, pad=4)

    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.yaxis.set_visible(False)

    ax.set_xlim(min(x_coords) - 1.0, max(x_coords) + 1.0)
    ax.set_ylim(-0.1, max_silhouette_height + 0.5)
    ax.set_aspect('equal')
    ax.figure.tight_layout()

    return ax


def _format_value(val, value_format):
    if value_format is None:
        return str(int(val)) if val == int(val) else str(val)
    if callable(value_format):
        return value_format(val)
    return value_format.format(val)
