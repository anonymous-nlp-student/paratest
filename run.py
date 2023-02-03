import sys
import openai
import random
import argparse

import pandas as pd

from tqdm import tqdm
from paratest.helper import (
    save_data,
    check_termination_condition
)
from paratest.annotate import annotate
from paratest.generator import ParaTestGenerator
from paratest.classifier import TransformersSequenceClassifier

if __name__ == "__main__":
    # loading user settings
    config = pd.read_json("config.json", orient="index").T.to_dict("records").pop()
    openai.api_key = config["key"]

    SAVE_INTERVAL = config["save_interval"]
    QUERY_BUDGET = config["query_budget"]
    N_DEMONSTRATIONS = config["n_demonstrations"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=int)
    parser.add_argument("--automatic", action="store_true")
    args = parser.parse_args()

    # loading initial test cases
    df = pd.read_json("dataset/initial_test_cases.json", lines=True, orient="records")

    # initial record
    initial_record = df[df.specification == args.specification].to_dict("records").pop()
    initial_test_cases = initial_record["initial_test_cases"]

    folder_name = str(args.specification).zfill(2)

    # loading validity checker
    clf = None
    if args.automatic:
        clf = TransformersSequenceClassifier(
            model_name="checkpoints/{}".format(folder_name),
            model_type="roberta",
            num_labels=2
        )

    test_case = random.choice(initial_test_cases)
    if isinstance(test_case, str):
        n_sentence_per_sample = 1
    else:
        n_sentence_per_sample = 2

    generator = ParaTestGenerator(
        n_sentence_per_sample=n_sentence_per_sample,
        initial_test_cases=initial_test_cases,
        model_name=config["model_name"],
        verbose=False if args.automatic else True
    )

    query_count = 0

    T0 = initial_test_cases
    Ti = list()
    valid_records = list()
    invalid_records = list()

    with tqdm(total=QUERY_BUDGET) as pbar:
        while True:
            if len(Ti) > 5:
                # randomly select demonstrations from top 20% confident cases
                # if empty then select from original cases
                if Ti == list():
                    sorted_T = T0
                else:
                    sorted_T = [k for k, v in sorted(Ti, key=lambda x: x[1])[-int(0.2 * len(Ti)):] if v > 0.8]

                if sorted_T == list():
                    sorted_T = T0

                s = random.choice(sorted_T)
                demonstrations = random.sample(sorted_T + T0, k=N_DEMONSTRATIONS)
            else:
                s = random.choice(T0)
                demonstrations = T0

            d = generator.generate(s, demonstrations=demonstrations)
            ti = d["generated_sample"]

            record = {
                **initial_record,
                "sample": s,
                "demonstrations": demonstrations,
                **pd.json_normalize(d, max_level=1).to_dict("records").pop()
            }

            # update query count
            consumed_budget = 0
            for response in d["response"].values():
                if response is None:
                    continue

                consumed_budget += response["usage"]["total_tokens"]

            query_count += consumed_budget
            pbar.update(consumed_budget)

            # annotation
            validity, prob = annotate(ti, T=initial_test_cases, clf=clf)

            if validity:
                if ti not in Ti:
                    Ti.append((ti, prob))
                    valid_records.append(record)
            else:
                invalid_records.append(record)

            if len(valid_records) % SAVE_INTERVAL == 0:
                save_data(valid_records, invalid_records, folder_name)

            # check termination criteria and exit
            if check_termination_condition(query_count, QUERY_BUDGET):
                save_data(valid_records, invalid_records, folder_name)
                sys.exit()

