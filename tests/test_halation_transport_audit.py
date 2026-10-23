"""Independent limits for the conditional audit, not validation of real 800T."""
from types import SimpleNamespace

import numpy as np

from tools.eval.halation_transport_audit import return_stack


def test_return_stack_matches_homogeneous_slab_beer_lambert():
    stock = SimpleNamespace(log_sensitivity=np.zeros((81, 3)), base_density=np.zeros(81))
    v = SimpleNamespace(layer_absorbance=(.1, .1, .1), yellow_filter=0.)
    mu = np.array([1., .5, .25])
    back, transmitted = return_stack(stock, v, mu)
    # At 550 nm each layer has absorption D=.1 and scattering extinction D=.15.
    wavelength_index = (550 - 380) // 5
    layer_T = 10 ** (-.25 / mu)
    layer_A = (1 - layer_T) * (.1 / .25)
    expected = np.column_stack([layer_A, layer_T * layer_A, layer_T**2 * layer_A])
    np.testing.assert_allclose(back[:, wavelength_index], expected, rtol=1e-13)
    np.testing.assert_allclose(transmitted[:, wavelength_index], layer_T**3, rtol=1e-13)
    assert np.all(back.sum(-1) + transmitted <= 1 + 1e-12)


def test_grazing_return_cannot_transmit_to_remote_layers():
    stock = SimpleNamespace(log_sensitivity=np.zeros((81, 3)), base_density=np.zeros(81))
    v = SimpleNamespace(layer_absorbance=(.1, .1, .1), yellow_filter=0.)
    back, transmitted = return_stack(stock, v, [1e-4])
    assert np.all(back[..., 0] > 0)
    assert np.all(back[..., 1:] == 0)
    assert np.all(transmitted == 0)
