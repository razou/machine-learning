from collections import Counter
import numpy as np


def target_distribution(train_y: np.ndarray) -> dict:
    label_counts = Counter(train_y[0])
    total_label_example = sum(label_counts.values())
    label_distribution = {k: np.round(v / total_label_example, 3) for k, v in label_counts.items()}
    return label_distribution
