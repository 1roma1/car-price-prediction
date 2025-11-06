import yaml

from src.utils import load_configuration


def test_load_configuration(tmp_path):
    yaml_file = tmp_path / "json_file.json"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump({"test": "ok"}, f)

    yaml_file = load_configuration(yaml_file)

    assert isinstance(yaml_file, dict)
    assert yaml_file.get("test") == "ok"
