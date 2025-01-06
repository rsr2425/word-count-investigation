import json
from json import JSONDecodeError
import enum
import evaluate

rouge_score = evaluate.load("rouge")


class Metric(enum.Enum):
    ROUGE = "ROUGE"
    FACTUAL_CONSISTENCY = "Factual Consistency"

    def __str__(self):
        return self.value


def compute_rouge(record, **kargs):
    return rouge_score.compute(
        predictions=[record['ai_summary']],
        references=[record['summary']]
    )


def generate_questions(text, llm, n):
    messages = [
      ("system", """
        You are a helpful question generating chatbot.  Generate {n} factual questions
        from the text provided by the user. Make sure these questions can be answered
        using the provided text, and that the answers should be yes or no. Make sure there
        are both questions that can be answered with yes and questions that can be answered
        with no. Think through step by step before answering and make sure there are a mix
        of answers to the questions you provide.

        Return the questions as a json containing a list of strings.
        """
      ),
      ("human", f"{text}"),
    ]
    ai_msg = llm.invoke(messages)
    questions = []
    try:
        questions = json.loads(ai_msg.content)['questions']
    except JSONDecodeError as e:
        questions = {'questions': [''] * n}
    return questions

def generate_anwsers(questions, source_text, llm):
    messages = [
      ("system", """
        You are a helpful question answering chatbot.  The user will give you a list of questions and the text off which you
        should answer them. Answer the questions using the provided text. Answer only with "Yes", "No", or "idk". If the
        question cannot be answered using the provided text, answer with "idk". If you are unsure, answer with "idk".
        If the question string is empty, answer with "idk".

        Return the answers as a json containing a list of strings.
        """
      ),
      ("human", f"""
        Please answer the following questions:

          {questions}

        using this text:

          {source_text}
      """),
    ]
    ai_msg = llm.invoke(messages)
    answers = []
    try:
        answers = json.loads(ai_msg.content)['answers']
    except (JSONDecodeError, TypeError) as e:
        answers = ['idk'] * len(questions)
    return answers


def compute_factual_consistency(record, llm, n):
    # TODO figure out why n isn't always respected
    questions = generate_questions(record['text'], llm, n)
    gt_answers = generate_anwsers(questions, record['text'], llm)
    # assert len(gt_answers) == n
    human_summary_answers = generate_anwsers(questions, record['summary'], llm)
    # assert len(human_summary_answers) == n
    ai_summary_answers = generate_anwsers(questions, record['ai_summary'], llm)
    # assert len(ai_summary_answers) == n

    if all(x == 'idk' for x in human_summary_answers):
        hfc = 0
    else:
        hfc = sum([1 if x == y else 0 for x, y in zip(human_summary_answers, gt_answers)]) / float(len(questions))
    if all(x == 'idk' for x in ai_summary_answers):
        afc = 0
    else:
        afc = sum([1 if x == y else 0 for x, y in zip(ai_summary_answers, gt_answers)]) / float(len(questions))

    return {
        'gt_answers': gt_answers,
        'human_summary_answers': human_summary_answers,
        'ai_summary_answers': ai_summary_answers,
        'human_factual_consistency': hfc,
        'ai_factual_consistency': afc,
    }

metric_fn_mapping = {
    Metric.ROUGE: compute_rouge,
    Metric.FACTUAL_CONSISTENCY: compute_factual_consistency,
}
