import re
import pathlib
import pandas as pd

from .helper import (
    get_latest_file,
    set_pandas_display
)

set_pandas_display()

class TestSuite:
    def __init__(self, specifications, base_path="labeling/"):
        self.specifications = specifications
        self.base_path = base_path

    def aggregate(self):
        paths = [pathlib.Path(self.base_path) / str(s).zfill(2) for s in self.specifications]

        dfs = list()
        for path in paths:
            filename = get_latest_file(path)
            if not filename:
                continue

            df = pd.read_json(filename, lines=True, orient="records")
            dfs.append(df)

        if dfs == list():
            return None
        else:
            df = pd.concat(dfs).reset_index(drop=True)
            df = df.iloc[df.generated_sample.drop_duplicates().index, :].reset_index(drop=True)

            return df

    def view(self):
        df = self.aggregate()
        if df is None:
            raise ValueError(f"No test cases generated. Please check {self.base_path}")
        else:
            print(df.generated_text.tolist())

    def test(self, clf):
        df = self.aggregate()
        if df is None:
            raise ValueError(f"No test cases generated. Please check {self.base_path}")
        else:
            df["pred"], _ = clf.predict(df.generated_sample.tolist())
            df["error"] = df.pred != df.label
            report_df = df[["specification", "description", "error"]].groupby(["specification", "description"])\
                                                                     .mean()

            print(report_df)
