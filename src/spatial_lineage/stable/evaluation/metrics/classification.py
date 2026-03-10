from __future__ import annotations


def accuracy_score(targets: list[int], predictions: list[int]) -> float:
    if not targets:
        return 0.0
    matches = sum(int(target == prediction) for target, prediction in zip(targets, predictions))
    return matches / len(targets)


def macro_f1_score(targets: list[int], predictions: list[int]) -> float:
    labels = sorted(set(targets + predictions))
    if not labels:
        return 0.0

    per_label: list[float] = []
    for label in labels:
        true_positive = sum(1 for target, prediction in zip(targets, predictions) if target == label and prediction == label)
        false_positive = sum(1 for target, prediction in zip(targets, predictions) if target != label and prediction == label)
        false_negative = sum(1 for target, prediction in zip(targets, predictions) if target == label and prediction != label)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0:
            per_label.append(0.0)
        else:
            per_label.append(2 * precision * recall / (precision + recall))
    return sum(per_label) / len(per_label)
