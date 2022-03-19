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


class QAModel(object):
    def __init__(self) -> None:
        self.scores = None

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

    def evaluate(self, dataset: str = "nq", nrows: Optional[int] = None):
        if dataset == "nq":
            annotations = read_annotations("data/nq-annotations.jsonl")
            references = read_references("data/nq-test.qa.csv")
            nq_test = pd.read_csv(
                "data/nq-test.qa.csv", sep="\t", names=["question", "answers"]
            )

            if nrows:
                annotations = annotations[:nrows]
                references = references[:nrows]
                nq_test = nq_test[:nrows]

        predictions = self._get_predictions(nq_test)

        scores = get_scores(predictions, references, annotations, get_bert_score=True)
        for label in ANNOTATIONS:
            _print_score(label, scores[label])
        self.scores = scores


if __name__ == "__main__":
    predictions = [{"id": 1, "prediction": "duck"}]
    references = [{"id": 1, "references": ["duck", "cat"]}]
    annotations = [{"id": 1, "labels": ["total", "answer_overlap"]}]

    get_scores(predictions=predictions, references=references, annotations=annotations)