import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def sample_data():
    return ['A', 'B', 'C'], [10, 20, 30]


@pytest.fixture
def numeric_data():
    return [0, 1, 2], [5, 15, 25]


@pytest.fixture
def fresh_ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)
