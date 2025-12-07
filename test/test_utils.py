import yaml
import json
import pandas as pd

from src.base.utils import load_yaml, load_json, get_X_y


def test_load_configuration(tmp_path):
    yaml_file = tmp_path / "file.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump({"test": "ok"}, f)

    yaml_file = load_yaml(yaml_file)

    assert isinstance(yaml_file, dict)
    assert yaml_file.get("test") == "ok"


def test_load_json(tmp_path):
    json_file = tmp_path / "file.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"test": "ok"}, f)

    json_file = load_json(json_file)

    assert isinstance(json_file, dict)
    assert json_file.get("test") == "ok"


def test_load_dataframe():
    target = "Weight"
    df = pd.DataFrame(
        [
            [1, "Bob", 45.2],
            [2, "Tom", 100.1],
            [3, "Alice", 50.4],
        ],
        columns=["Id", "Name", "Weight"],
    )
    X, y = get_X_y(df, target)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert target not in X.columns
    assert y.name == target
