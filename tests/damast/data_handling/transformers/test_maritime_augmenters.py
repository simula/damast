
import pandas as pd

from damast.data_handling.transformers.augmenters import (
    AddCombinedLabel,
    InvertedBinariser
)
from damast.domains.maritime.data_specification import ColumnName


def test_inverted_binariser():
    data = [-10, 10, 20, 101]
    df = pd.DataFrame(data, columns=[ColumnName.HEADING])

    inverted_binariser = InvertedBinariser(base_column_name=ColumnName.HEADING,
                                           threshold=100)
    df_transformed = inverted_binariser.transform(df)

    expected_column_name = f"{ColumnName.HEADING}_TRUE"
    assert expected_column_name in df_transformed.columns
    assert df.iloc[0][expected_column_name] == 1
    assert df.iloc[1][expected_column_name] == 1
    assert df.iloc[2][expected_column_name] == 1
    assert df.iloc[3][expected_column_name] == 0


def test_add_combined_label():
    col_a_name = "a"
    col_a_permitted_values = {"0": "a0", "1": "a1", "2": "a2"}

    col_b_name = "b"
    col_b_permitted_values = {"0": "b0", "1": "b1", "2": "b2"}

    data = [["a0", "b0"],
            ["a0", "b1"]]
    df = pd.DataFrame(data, columns=[col_a_name, col_b_name])

    add_combined_label = AddCombinedLabel(
        column_permitted_values={
            col_a_name: col_a_permitted_values,
            col_b_name: col_b_permitted_values,
        },
        column_names=[col_a_name, col_b_name],
        combination_name=ColumnName.STATUS
    )

    df_transformed = add_combined_label.transform(df)
    labels = add_combined_label.label_mapping
    for index, row in df_transformed.iterrows():
        assert [row[col_a_name], row[col_b_name]] == labels[str(row[ColumnName.STATUS])]
