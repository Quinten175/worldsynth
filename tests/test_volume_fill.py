import pytest
import yaml

from worldsynth.config import load_config


def _config(tmp_path, volume_fill):
    asset = tmp_path / "part.usd"
    asset.write_text("#usda 1.0\n")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scene": {"physics_settle": True},
                "volume_fill": {"enabled": True, "asset": str(asset), **volume_fill},
            }
        )
    )
    return path


def test_loads_valid_volume_fill_config(tmp_path):
    config = load_config(
        _config(
            tmp_path,
            {
                "count": 4,
                "spawn": {"x": [1.0, 2.0], "y": [-1.0, 0.0], "z": [2.0, 3.0]},
            },
        )
    )

    assert config.volume_fill.enabled
    assert config.volume_fill.count == 4
    assert config.volume_fill.asset == str(tmp_path / "part.usd")
    assert config.volume_fill.spawn.x == (1.0, 2.0)


@pytest.mark.parametrize(
    ("volume_fill", "message"),
    [
        ({"count": 0}, "volume_fill.count"),
        ({"count": 1.5}, "volume_fill.count"),
        ({"spawn": {"x": [2.0, 1.0]}}, "volume_fill.spawn.x"),
        ({"spawn": {"z": [float("nan"), 2.0]}}, "volume_fill.spawn.z"),
        ({"drop_interval_seconds": 0}, "drop_interval_seconds"),
    ],
)
def test_rejects_invalid_volume_fill_config(tmp_path, volume_fill, message):
    with pytest.raises(ValueError, match=message):
        load_config(_config(tmp_path, volume_fill))


def test_requires_physics_settle_for_volume_fill(tmp_path):
    path = _config(tmp_path, {})
    data = yaml.safe_load(path.read_text())
    data["scene"]["physics_settle"] = False
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="scene.physics_settle"):
        load_config(path)
