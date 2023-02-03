import openai

from ratelimit import (
    limits,
    sleep_and_retry
)


# rate limit: https://help.openai.com/en/articles/5955598-is-api-usage-subject-to-any-rate-limits
def query_gpt_unreliable(model, prompt, n_per_query):
    response = openai.Completion.create(
                                engine=model,
                                prompt=prompt,
                                n=n_per_query,
                                max_tokens=128,
                                temperature=1,
                                top_p=0.9,
                                frequency_penalty=0,
                                presence_penalty=0
                                )
    return response


@sleep_and_retry
@limits(calls=1, period=2)
def query_gpt(**kwargs):
    return query_gpt_unreliable(**kwargs)