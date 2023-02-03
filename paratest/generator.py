import re
import openai

import pandas as pd

from termcolor import cprint

from .gpt import query_gpt


def process_query_response(response):
    output = response["choices"][0]["text"]
    texts = [re.sub("[-}{]", "", t).strip() for t in output.split("\n")]
    return texts


def process_paraphrase_response(response):
    return response["choices"][0]["text"].replace("{", "").replace("}", "")


def process_complete_response(response):
    output = response["choices"][0]["text"]

    searches = re.findall("{(.*?)}", output)
    if searches:
        return searches[0].replace("{", "").replace("}", "")
    else:
        return None


class ParaTestGenerator:
    def __init__(
            self,
            n_sentence_per_sample,
            initial_test_cases,
            model_name="text-curie-001",
            verbose=False
    ):
        self.model_name = model_name
        self.initial_test_cases = initial_test_cases
        self.n_sentence_per_sample = n_sentence_per_sample
        self.verbose = verbose

    def generate(self, sample, demonstrations, method="paratest"):
        paraphrase_response = None
        complete_response = None
        query_response = None

        if self.n_sentence_per_sample == 1:
            if method == "paratest":
                if self.verbose:
                    cprint("RUNNING PARATEST", "red")

                paraphrase_response = self.paraphrase(sample)
                generated_sample = process_paraphrase_response(paraphrase_response)

            elif method == "testaug":
                if self.verbose:
                    cprint("RUNNING TESTAUG", "red")

                query_response = self.query(demonstrations + [sample])
                processed_query_response = process_query_response(query_response)

                generated_sample = processed_query_response[0]

            else:
                generated_sample = None

        elif self.n_sentence_per_sample == 2:
            if method == "paratest":
                if self.verbose:
                    cprint("RUNNING PARATEST", "red")

                paraphrase_response = self.paraphrase(sample[0])
                paraphrasing1 = process_paraphrase_response(paraphrase_response)

                complete_response = self.complete(paraphrasing1, self.initial_test_cases)
                paraphrasing2 = process_complete_response(complete_response)

            elif method == "testaug":
                if self.verbose:
                    cprint("RUNNING TESTAUG", "red")

                query_response = self.query(demonstrations + [sample])
                processed_query_response = process_query_response(query_response)

                paraphrasing1, paraphrasing2 = processed_query_response[:2]

            else:
                paraphrasing1 = None
                paraphrasing2 = None

            generated_sample = (paraphrasing1, paraphrasing2)
        else:
            raise ValueError("ParaTest does not support a sample with more than 2 sentences.")

        return {
            "n_sentence_per_sample": self.n_sentence_per_sample,
            "generated_sample": generated_sample,
            "response": {
                "paraphrase": paraphrase_response,
                "complete": complete_response,
                "query": query_response
            }
        }

    def query(self, examples):
        prompt = self.format_input(examples)

        response = query_gpt(
            model=self.model_name,
            prompt=prompt,
            n_per_query=1
        )

        return response

    def paraphrase(self, d):
        def fill_template(x, style, y):
            return f"Here is some text: {{{x}}}. Here is a rewrite of the text, which is more {style}: {{{y}}}"

        examples = [
            ("I was really sady about the loss", "positive",
             "I was able to accept and work through the loss to move on."),
            ("They asked loudly, over the sound of the train", "intense",
             "They yelled aggressively, over the clanging of the train")
        ]

        examples += [(d, "diverse", "")]
        prompt = "\n".join(fill_template(*example) for example in examples).rstrip("}")

        response = query_gpt(
            model=self.model_name,
            prompt=prompt,
            n_per_query=1
        )

        return response

    def complete(self, p, examples):
        prompt = self.format_input(examples) + f"\n- {{{{{p}}}"
        response = query_gpt(
            model=self.model_name,
            prompt=prompt,
            n_per_query=1
        )

        return response

    def format_input(self, examples):
        if self.n_sentence_per_sample == 1:
            return "\n".join("- {{{}}}".format(example) for example in examples)

        if self.n_sentence_per_sample == 2:
            return "\n".join("- {{{{{}}}\n- {{{}}}}}".format(example[0], example[1]) for example in examples)