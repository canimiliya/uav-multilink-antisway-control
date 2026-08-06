from pathlib import Path

import numpy as np

from uav_sway.disturbances.wind_io import sha256_file, write_wind_csv
from uav_sway.disturbances.wind_profiles import generate_wind_profile, load_wind_config


ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_is_array_and_byte_reproducible(tmp_path):
    config = load_wind_config(ROOT / "configs/wind_profiles.yaml")
    first = generate_wind_profile("low_frequency_random", config, seed=7)
    second = generate_wind_profile("low_frequency_random", config, seed=7)
    assert np.array_equal(first.wind_x, second.wind_x)
    path_a, path_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_wind_csv(path_a, first)
    write_wind_csv(path_b, second)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert sha256_file(path_a) == sha256_file(path_b)


def test_different_seed_changes_random_sequence_and_fixed_profiles_ignore_seed():
    config = load_wind_config(ROOT / "configs/wind_profiles.yaml")
    assert not np.array_equal(
        generate_wind_profile("low_frequency_random", config, seed=0).wind_x,
        generate_wind_profile("low_frequency_random", config, seed=1).wind_x,
    )
    assert np.array_equal(
        generate_wind_profile("constant_crosswind", config, seed=0).wind_x,
        generate_wind_profile("constant_crosswind", config, seed=1).wind_x,
    )
