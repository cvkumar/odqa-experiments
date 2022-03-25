from typing import Optional
import pandas as pd
import transformers
import time
from nq_evaluate import (
    _print_score,
    get_scores,
    read_references,
    read_annotations,
    ANNOTATIONS,
)

from bert_score import score


class QADataset:
    nq = "nq"
    triviaqa = "triviaqa"
    webquestion = "webquestions"


class QAModel(object):
    def __init__(self) -> None:
        self.scores = None
        self.predictions = None

    def _get_predictions(self, nq_test: pd.DataFrame):
        predictions = []
        start = time.time()
        for i in range(len(nq_test)):
            if i % 40 == 0 and i > 0:
                num_left = len(nq_test) - i
                time_taken = round(time.time() - start, 3)
                rate = time_taken / i
                expected_time_left = round((rate * num_left) / 60, 4)  # in minutes
                print(
                    f"At example: {i} after {time_taken} seconds. Expecting to take: {expected_time_left} more minutes"
                )

            example = nq_test.iloc[i]
            prediction = self.predict_answer(example["question"])
            predictions.append({"id": i, "prediction": prediction})
        return predictions

    def predict_answer(self, text: str) -> str:
        raise NotImplementedError("Please Implement this method")

    def evaluate(
        self,
        dataset: str = QADataset.nq,
        nrows: Optional[int] = None,
        get_bert_score=False,
        get_new_predictions=True,
    ):
        annotations = read_annotations(f"data/{dataset}-annotations.jsonl")
        references = read_references(f"data/{dataset}-test.qa.csv")
        test_dataset = pd.read_csv(
            f"data/{dataset}-test.qa.csv", sep="\t", names=["question", "answers"]
        )

        if nrows:
            annotations = annotations[:nrows]
            references = references[:nrows]
            test_dataset = test_dataset[:nrows]

        if get_new_predictions:
            self.predictions = self._get_predictions(test_dataset)

        scores = get_scores(
            self.predictions, references, annotations, get_bert_score=get_bert_score
        )
        for label in ANNOTATIONS:
            _print_score(label, scores[label])
        self.scores = scores


if __name__ == "__main__":
    predictions = [{"id": 1, "prediction": "duck"}, {"id": 2, "prediction": "egg"}]
    references = [
        {"id": 1, "references": ["duck", "cat"]},
        {"id": 2, "references": ["egg", "lamb"]},
    ]
    annotations = [
        {"id": 1, "labels": ["total", "answer_overlap"]},
        {"id": 2, "labels": ["total", "answer_overlap"]},
    ]

    results = get_scores(
        predictions=predictions,
        references=references,
        annotations=annotations,
        get_bert_score=True,
    )
    for label in ANNOTATIONS:
        if label in results:
            _print_score(label, results[label])
    # result = score(["test123"], ["test123"], lang="en", verbose=True, rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

    # print(result)
