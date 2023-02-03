import time

import numpy as np
import pandas as pd

from scipy import special
from collections import Counter
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel
)


class TransformersSequenceClassifier:
    def __init__(self, model_name, model_type, num_labels, model_dir_name=None, verbose=True):
        # current time
        current_time = str(time.time()).split(".")[0]
        if model_dir_name is None:
            model_dir_name = model_name

        # simpletransformers
        model_args = ClassificationArgs()

        model_args.manual_seed = 42
        model_args.silent = not verbose

        model_args.max_seq_length = 256
        model_args.train_batch_size = 4
        model_args.eval_batch_size = 4

        model_args.num_train_epochs = 5
        model_args.save_model_every_epoch = False
        model_args.save_steps = -1
        model_args.evaluate_during_training = False

        model_args.output_dir = f"checkpoints/{model_dir_name}"
        model_args.best_model_dir = f"checkpoints/{model_dir_name}/best_model"
        model_args.tensorboard_dir = "runs/{}".format(current_time)
        model_args.overwrite_output_dir = True

        self.model = ClassificationModel(
            model_type,
            model_name,
            num_labels=num_labels,
            args=model_args,
        )

    def fit(self, train_df):
        train_df = self.process_data(train_df)

        # training
        self.model.train_model(
            train_df,
            eval_df=None
        )

    def predict(self, unlabeled_data):
        if isinstance(unlabeled_data, pd.DataFrame):
            test_df = unlabeled_data
            test_df = self.process_data(test_df)

            preds, _ = self.model.predict(test_df)
            test_df = test_df.assign(predictions=preds)

            return test_df
        else:
            # simpletransformers automatically handles the issues with single/pair sentence
            # or list of single/pair sentences if the format of data passed in is correct
            # see: https://simpletransformers.ai/docs/sentence-pair-classification/
            preds, probs = self.model.predict(unlabeled_data)

            probs = special.softmax(probs, axis=1)

            # only return the predictions associated with the prediction
            return preds, probs[np.arange(len(preds)), preds]

    def process_data(self, df):
        def process_sentence_pair_dataset(df):
            df = pd.DataFrame(
                [
                    {
                        "text_a": tup.text[0],
                        "text_b": tup.text[1],
                        "labels": tup.labels
                    }
                    for tup in df.itertuples()
                ]
            )
            return df

        # there has to be two columns: text and labels so that the simpletransformers could work
        # remove empty input; this should work for both string and tuple
        df = df[df.text.apply(lambda x: len(x) > 1)].reset_index(drop=True)

        # if the majority of the data types are string, then set is_single_sentence_task to True
        is_single_sentence_task = False
        types = df.text.apply(lambda x: isinstance(x, str)).tolist()
        str_type_cnt = Counter(types)
        if str_type_cnt[True] >= 0.9 * len(types):
            is_single_sentence_task = True

        # process dataset into format suitable for simpletransformers to process
        df = df.rename(columns={"label": "labels"})
        if not is_single_sentence_task:
            df = process_sentence_pair_dataset(df)

        return df