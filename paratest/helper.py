import os
import time
import pathlib

import pandas as pd

from datetime import datetime
from termcolor import cprint

def set_pandas_display(max_colwidth=100):
    pd.options.display.max_rows = None
    pd.options.display.max_colwidth = max_colwidth
    pd.options.display.max_columns = None

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

def get_system_time():
    return str(time.time()).split(".")[0]

def get_latest_file(folder):
    folder = pathlib.Path(folder)
    if not folder.exists():
        return None

    current_time = time.time()
    filename_dict = {
        filename: current_time - float(filename.stem) for filename in folder.glob("*.json")
        if filename.stem.isdigit() and not os.path.isdir(filename)
    }

    latest_file = min(filename_dict, key=filename_dict.get) if filename_dict != dict() else None

    return latest_file


def save_data(valid_records, invalid_records, desc):
    save_path = pathlib.Path("labeling/{}".format(str(desc)))
    if not save_path.exists():
        save_path.mkdir(parents=True)

    df = pd.concat((
        pd.DataFrame(valid_records).assign(validity=1),
        pd.DataFrame(invalid_records).assign(validity=0)
    ))

    df.to_json(save_path / "{}.json".format(get_system_time()), lines=True, orient="records")


def check_termination_condition(query_count, query_budget):
    if query_count > query_budget:
        cprint(f"#Querys {query_count} Exceeding Budget {query_budget}. Exiting...", "red")
        return True
    else:
        return False