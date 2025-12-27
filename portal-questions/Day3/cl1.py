# main.py

import warnings
warnings.simplefilter(action='ignore')

import os, sys
import numpy as np


def entropy(labels):
    if labels.size == 0:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(float) / labels.size
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(parent_entropy, child_labels_list):
    total = sum(child.size for child in child_labels_list)
    if total == 0:
        return 0.0
    weighted_child_entropy = 0.0
    for child in child_labels_list:
        if child.size == 0:
            continue
        weight = child.size / total
        weighted_child_entropy += weight * entropy(child)
    return parent_entropy - weighted_child_entropy


def main():
    filename = input().strip()          # e.g., Sample.csv

    if not filename.endswith(".csv"):
        print(f"Error: Unable to read file '{filename}'.")
        return

    try:
        data = np.genfromtxt(
            os.path.join(sys.path[0], filename),
            delimiter=",",
            skip_header=1
        )
    except Exception:
        print(f"Error: Unable to read file '{filename}'.")
        return

    if data.ndim != 2 or data.shape[1] != 6:
        print(f"Error: Unable to read file '{filename}'.")
        return

    fasting_col = 0
    bmi_col = 1
    family_col = 3
    target_col = 5

    y = data[:, target_col]

    parent_H = entropy(y)
    print(f"Parent Node Entropy: {parent_H:.3f}")

    # ---- define helper *inside* main, before using it ----
    def ig_for_feature(col_idx):
        feature = data[:, col_idx]
        mask_pos = feature > 0
        child1 = y[mask_pos]      # feature > 0
        child2 = y[~mask_pos]     # feature <= 0
        return information_gain(parent_H, [child1, child2])
    # ------------------------------------------------------



    ig_fasting = ig_for_feature(fasting_col)
    print(f"Information Gain (Fasting blood): {ig_fasting:.3f}")

    ig_bmi = ig_for_feature(bmi_col)
    print(f"Information Gain (bmi): {ig_bmi:.3f}")

    ig_family = ig_for_feature(family_col)
    print(f"Information Gain (FamilyHistory): {ig_family:.3f}")

    ig_dict = {
        "Fasting blood": ig_fasting,
        "bmi": ig_bmi,
        "FamilyHistory": ig_family,
    }
    best_feature = max(ig_dict, key=ig_dict.get)
    best_ig = ig_dict[best_feature]

    print(f"Best Feature for root node: {best_feature} with Information Gain: {best_ig:.3f}")


if __name__ == "__main__":
    main()