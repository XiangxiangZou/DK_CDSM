import numpy as np

from koopman_control.data.normalization import Normalizer


def test_normalizer_round_trip() -> None:
    values = np.array(
        [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
        dtype=np.float64,
    )
    normalizer = Normalizer.fit(values)
    restored = normalizer.inverse(normalizer.transform(values))
    np.testing.assert_allclose(restored, values)


def test_normalizer_constant_dimension_is_finite() -> None:
    values = np.ones((4, 2), dtype=np.float64)
    normalizer = Normalizer.fit(values)
    np.testing.assert_allclose(normalizer.std, np.ones(2))
