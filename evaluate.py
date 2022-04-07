from tkinter.messagebox import NO
from typing import Optional
import pandas as pd
import transformers
import time
from overlap_evaluate import (
    _print_score,
    get_scores,
    read_references,
    read_annotations,
    ANNOTATIONS,
)
import ast
from bert_score import score


class QADataset:
    nq = "nq"
    triviaqa = "triviaqa"
    webquestion = "webquestions"


class QAModel(object):
    def __init__(
        self, dataset: Optional[str] = QADataset.nq, nrows: Optional[int] = 0
    ) -> None:
        self.scores_per_label = None
        self.all_scores = None
        self.predictions = None
        self.tracker = []

        self.dataset: str = dataset
        self.annotations = read_annotations(f"data/{dataset}-annotations.jsonl")
        self.references = read_references(f"data/{dataset}-test.qa.csv")
        self.test_dataset = pd.read_csv(f"data/{dataset}-test-ctxs.qa.csv")
        self.test_dataset["ctxs"] = self.test_dataset["ctxs"].apply(
            lambda x: ast.literal_eval(x)
        )

        if nrows:
            self.annotations = self.annotations[:nrows]
            self.references = self.references[:nrows]
            self.test_dataset = self.test_dataset[:nrows]

    def get_predictions(self):
        print(f"Retrieving {len(self.test_dataset)} predictions")
        predictions = []
        start = time.time()
        for i in range(len(self.test_dataset)):
            if i % 40 == 0 and i > 0:
                num_left = len(self.test_dataset) - i
                time_taken = round(time.time() - start, 3)
                rate = time_taken / i
                expected_time_left = round((rate * num_left) / 60, 4)  # in minutes
                print(
                    f"At example: {i} after {time_taken} seconds. Expecting to take: {expected_time_left} more minutes"
                )

            example = self.test_dataset.iloc[i]
            prediction = self.predict_answer(example["question"], example["ctxs"])
            predictions.append({"id": i, "prediction": prediction})

        self.predictions = predictions
        return predictions

    def predict_answer(self, text: str, contexts: list) -> str:
        raise NotImplementedError("Please implement this method")

    def evaluate(
        self,
        get_bert_score=False,
        get_predictions=False,
    ):
        if get_predictions:
            print(f"Obtaining predictions:\n")
            self.predictions = self.get_predictions()

        print(f"Obtaining scores:\n")
        scores_per_label, all_scores = get_scores(
            self.predictions,
            self.references,
            self.annotations,
            get_bert_score=get_bert_score,
        )

        try:
            for label in ANNOTATIONS:
                _print_score(label, scores_per_label[label])
        except Exception:
            pass

        self.all_scores = all_scores

        saved_results = []
        # NOTE: We are assuming same order here
        for i in range(len(self.annotations)):
            score = self.all_scores[i]
            saved_results.append(
                {
                    "question": self.test_dataset.iloc[i]["question"],
                    "id": self.annotations[i]["id"],
                    "answers": self.references[i]["references"],
                    "prediction": self.predictions[i]["prediction"],
                    "overlap": self.annotations[i]["labels"],
                    "em": score["exact_match"],
                    "f1": score["f1_score"],
                    "bert_score": score["bert_score"],
                    "meteor_score": score["meteor_score"],
                }
            )

        scores_per_label["no_overlap"] = self.compute_no_overlap_score(saved_results)
        self.scores_per_label = scores_per_label

        return saved_results, scores_per_label

    def compute_answer_overlap_only_scores(self, saved_results):
        answer_overlap_scores = {"em": 0, "f1": 0, "bert_score": 0, "meteor_score": 0}
        answer_overlap_count = 0
        for result in saved_results:
            if "answer_overlap_only" in result["overlap"]:
                answer_overlap_scores["em"] += result["em"]
                answer_overlap_scores["f1"] += result["f1"]
                answer_overlap_scores["meteor_score"] += result["meteor_score"]
                if result["bert_score"] != "NA":
                    answer_overlap_scores["bert_score"] += result["bert_score"]

                answer_overlap_count += 1

        if answer_overlap_count:
            answer_overlap_scores["total"] = answer_overlap_count
            answer_overlap_scores["em"] = round(
                answer_overlap_scores["em"] / answer_overlap_count, 4
            )
            answer_overlap_scores["f1"] = round(
                answer_overlap_scores["f1"] / answer_overlap_count, 4
            )
            answer_overlap_scores["meteor_score"] = round(
                answer_overlap_scores["meteor_score"] / answer_overlap_count, 4
            )

            if answer_overlap_scores.get("bert_score"):
                answer_overlap_scores["bert_score"] = round(
                    answer_overlap_scores["bert_score"] / answer_overlap_count, 4
                )

        return answer_overlap_scores

    def compute_no_overlap_score(self, saved_results):
        no_overlap_scores = {"em": 0, "f1": 0, "bert_score": 0, "meteor_score": 0}
        no_overlap_count = 0
        for result in saved_results:
            if (
                "no_answer_overlap" in result["overlap"]
                and "no_question_overlap" in result["overlap"]
            ):
                no_overlap_scores["em"] += result["em"]
                no_overlap_scores["f1"] += result["f1"]
                no_overlap_scores["meteor_score"] += result["meteor_score"]
                if result["bert_score"] != "NA":
                    no_overlap_scores["bert_score"] += result["bert_score"]

                no_overlap_count += 1

        if no_overlap_count:
            no_overlap_scores["total"] = no_overlap_count
            no_overlap_scores["em"] = round(
                no_overlap_scores["em"] / no_overlap_count, 4
            )
            no_overlap_scores["f1"] = round(
                no_overlap_scores["f1"] / no_overlap_count, 4
            )
            no_overlap_scores["meteor_score"] = round(
                no_overlap_scores["meteor_score"] / no_overlap_count, 4
            )

            if no_overlap_scores.get("bert_score"):
                no_overlap_scores["bert_score"] = round(
                    no_overlap_scores["bert_score"] / no_overlap_count, 4
                )

        return no_overlap_scores


if __name__ == "__main__":
    predictions = [{"id": 1, "prediction": "duck"}, {"id": 2, "prediction": "egg"}]
    references = [
        {"id": 1, "references": ["duck", "cat"]},
        {"id": 2, "references": ["egg", "lamb"]},
    ]
    annotations = [
        {"id": 1, "labels": ["total", "no_answer_overlap"]},
        {"id": 2, "labels": ["total", "answer_overlap"]},
    ]

    results, all_scores = get_scores(
        predictions=predictions,
        references=references,
        annotations=annotations,
        get_bert_score=True,
    )
    for label in ANNOTATIONS:
        if label in results:
            _print_score(label, results[label])

    print(all_scores)

    # result = score(["test123"], ["test123"], lang="en", verbose=True, rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")

    # print(result)
