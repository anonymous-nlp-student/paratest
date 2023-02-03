# ParaTest: Paraphrasing with LLMs Improves NLP Testing

## Environment and Preparation

In order to run the ParaTest system, a machine with at least one GPU is recommended but not required; ParaTest has a mode that does not rely on running models dependent on GPUs.

- Step 1: No matter GPUs are available or not, you need to create and activate the computation environment using `requirements.txt`. 

    ```bash
    # create a base conda environment
    conda create --name paratest python==3.9

    # install dependencies
    pip install -r requirements.txt
    pip install -e .

    # activate the conda environment
    conda activate paratest
    ```

- Step 2 (Optional): Download the `checkpoints.tar` (validity checkers) from [here](https://drive.google.com/file/d/1_8bKxvxnyUwTlq6xBLeC_i9DfxYd6NJS/view?usp=share_link) (about 7.1 GB) to the `checkpoints/` directory. After unwrapping, the directory should look like the following, where each folder corresponds to each specification mentioned in the paper.

  ```bash
  checkpoints/
  ├── 01
  ...
  ├── 14
  └── 15
  ```

- Step 3: Put your OpenAI key to `config.json`. If you do not have an OpenAI key, you could apply for one [here](https://platform.openai.com/signup); you will have a 20 USD quota for free, which is sufficient for you to replicate the experiments in the paper.

  ```json
  {
    "key": "<OPENAI-API-KEY>",
    ...
  }
  ```
  

## Generating Test Cases

After the preparation step, all you need to do is working with the `run.py` file. For example, if you would like to generate test cases for the specification "Temporal: Before something vs. after something", which is numbered 13, you could:

- Without a GPU: This requires you to annotate the validity the generated test cases with a human annotator in the loop. You will be prompted with a CLI-based window that asks for your label.

  ```bash
  python run.py --specification 13
  ```

- With a GPU: The validity checkers save the step of annotating validity of generated test cases; this makes the pipeline fully automatic (hence `--automatic` flag).

    ```bash
    python run.sh --specification 13 --automatic
    ```

You will find a directory of `labeling/13` automatically created; it stores all the annotated data used for testing NLP models.

## Testing a NLP Model

The test cases generated in the previous step are ready for testing NLP models. The example below tests the most downloaded sentiment classifier `gchhablani/bert-base-cased-finetuned-sst2` on the [HuggingFace model hub](https://huggingface.co/models?sort=downloads&search=sst2) and generates a report of its error rates.

```python
from paratest.suite import TestSuite
from paratest.classifier import TransformersSequenceClassifier

clf = TransformersSequenceClassifier(
    model_name="gchhablani/bert-base-cased-finetuned-sst2",
    model_type="bert",
    num_labels=2,
)

suite = TestSuite(specifications=[1, 3])
suite.test(clf)
```

## Custom Specifications

It is possible to apply the ParaTest system to the specifications that suit your own application yet not considered in the paper. All you need to do is to add a row to the `dataset/initial_test_cases.json`. For example, you would like to test specification 123, you just need to run:

```bash
python run.sh --specification 123
```

