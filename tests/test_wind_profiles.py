from pathlib import Path

import numpy as np

from uav_sway.disturbances.wind_profiles import generate_wind_profile, load_wind_config


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_constant_and_one_cosine_boundaries():
    config = load_wind_config(ROOT / "configs/wind_profiles.yaml")
    constant = generate_wind_profile("constant_crosswind", config)
    assert constant.time.size == 2401
    assert constant.wind_x[np.searchsorted(constant.time, 3.995)] == 0.0
    assert constant.wind_x[np.searchsorted(constant.time, 4.0)] == 3.0
    gust = generate_wind_profile("one_cosine_gust", config)
    assert gust.wind_x[np.searchsorted(gust.time, 5.0)] == 0.0
    assert gust.wind_x[np.searchsorted(gust.time, 6.0)] == 3.0
    assert gust.wind_x[np.searchsorted(gust.time, 7.0)] == 0.0
    assert np.all(gust.wind_x[(gust.time < 5.0) | (gust.time > 7.0)] == 0.0)


def test_random_wind_is_finite_and_clipped():
    config = load_wind_config(ROOT / "configs/wind_profiles.yaml")
    series = generate_wind_profile("low_frequency_random", config, seed=0)
    assert np.isfinite(series.wind_x).all()
    assert np.max(np.abs(series.wind_x)) <= 3.0
