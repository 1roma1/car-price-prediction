import pytest
import pandas as pd

from src.data.preprocessing import (
    extract_categories,
    split_and_take_first,
    replace,
)


@pytest.mark.parametrize(
    "x, categories, other, res",
    [
        ("бензин (гибрид)", ["бензин", "дизель"], "другой", "бензин"),
        ("дизель", ["бензин", "дизель"], "другой", "дизель"),
        ("воздух", ["бензин", "дизель"], "другой", "другой"),
    ],
)
def test_extract_categories(x, categories, other, res):
    category = extract_categories(x, categories, other)
    assert isinstance(category, str)
    assert category == res


@pytest.mark.parametrize(
    "series, res",
    [
        (
            pd.Series(["one", "one two", "one two three"]),
            pd.Series(["one", "one", "one"]),
        )
    ],
)
def test_split_and_take_first(series, res):
    preprocessed_series = split_and_take_first(series)

    assert isinstance(preprocessed_series, pd.Series)
    assert preprocessed_series.equals(res)


@pytest.mark.parametrize(
    "series, old, new, res",
    [
        (
            pd.Series(["one", "one two", "one two three"]),
            " ",
            "",
            pd.Series(["one", "onetwo", "onetwothree"]),
        )
    ],
)
def test_replace(series, old, new, res):
    preprocessed_series = replace(series, old, new)

    assert isinstance(preprocessed_series, pd.Series)
    assert preprocessed_series.equals(res)
