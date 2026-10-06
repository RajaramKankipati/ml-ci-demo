import numpy as np
import pandas as pd

from src.data import preprocess


def make_raw():
    # create a small dataset for testing, with missing values and column that has to be dropped
    X = pd.DataFrame({
        "cap-shape": ["x", "b", "x", "f"],
        "stalk-root": ["e", np.nan, np.nan, "c"],
        "veil-type": ["p", "p", "p", "p"],
    })
    y = pd.Series(["e", "p", "p", "e"])
    return X, y


def test_drops_constant_veil_type():
    X, _ = preprocess(*make_raw())
    assert not any(c.startswith("veil-type") for c in X.columns)


def test_missing_values_become_unknown_category_without_dropping_rows():
    X, y = preprocess(*make_raw())
    assert len(X) == 4 and len(y) == 4
    assert "stalk-root_unknown" in X.columns
    assert X["stalk-root_unknown"].tolist() == [False, True, True, False]
    # Rows with missing values are all poisonous here; they must survive.
    assert y[X["stalk-root_unknown"]].tolist() == [1, 1]


def test_target_encoded_poisonous_as_one():
    _, y = preprocess(*make_raw())
    assert y.tolist() == [0, 1, 1, 0]


def test_works_when_veil_type_already_absent():
    X_raw, y_raw = make_raw()
    X, _ = preprocess(X_raw.drop(columns=["veil-type"]), y_raw)
    assert "cap-shape_x" in X.columns


def test_output_is_fully_numeric_with_no_nans():
    X, _ = preprocess(*make_raw())
    assert not X.isna().any().any()
    assert all(pd.api.types.is_bool_dtype(t) or pd.api.types.is_numeric_dtype(t)
               for t in X.dtypes)