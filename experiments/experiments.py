from langchain_openai import ChatOpenAI

from external_logging import log_dataset_to_wandb
from metrics import metric_fn_mapping
from tasks import summarize


def gen_run_name():
    pass


def process_dataset(
        dataset,
        llm,
        n,
        metrics,
        word_count_target=None,
        otherinstructions=None
):
    print("Processing Dataset!")
    print("Now summarizing data...")
    processed_dataset = dataset.map(
        summarize,
        fn_kwargs={
            'llm': llm,
            'word_count_target': word_count_target,
            'otherinstructions': otherinstructions
        }
    )
    # TODO: Add the LLM Judge as a separate model
    # llm_judge = ChatOpenAI(model_name=LLM_JUDGE_MODEL_ID, temperature=temperature)
    for metric in metrics:
        print(f"Now calculating {str(metric)}...")
        processed_dataset = processed_dataset.map(
            metric_fn_mapping[metric],
            fn_kwargs={
                'llm': llm,
                'n': n
            }
        )
    print("Done!")
    return processed_dataset


def run_experiment(
        model_name,
        temperature,
        dataset,
        number_of_questions,
        metrics,
        word_count_target=None,
        subset_size=None,
        otherinstructions=None,
        log_to_wandb=None
):
      llm = ChatOpenAI(model_name=model_name, temperature=temperature)
      if subset_size is not None:
        dataset = dataset.select(range(subset_size))
      results_subset = process_dataset(dataset, llm, number_of_questions, metrics, word_count_target=word_count_target, otherinstructions=otherinstructions)
      if log_to_wandb is not None and log_to_wandb:
        log_dataset_to_wandb(results_subset, PROJECT_NAME, f"{RUN_PREFIX}{model_name}")
      return results_subset
