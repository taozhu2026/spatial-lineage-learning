from __future__ import annotations


def macro_ovr_auroc(targets: list[int], scores: list[list[float]]) -> float:
    labels = sorted(set(targets))
    aucs: list[float] = []
    for label in labels:
        positives = [score[label] for score, target in zip(scores, targets) if target == label]
        negatives = [score[label] for score, target in zip(scores, targets) if target != label]
        if not positives or not negatives:
            continue

        wins = 0.0
        comparisons = 0
        for positive in positives:
            for negative in negatives:
                comparisons += 1
                if positive > negative:
                    wins += 1.0
                elif positive == negative:
                    wins += 0.5
        aucs.append(wins / comparisons if comparisons else 0.0)
    return sum(aucs) / len(aucs) if aucs else 0.0
