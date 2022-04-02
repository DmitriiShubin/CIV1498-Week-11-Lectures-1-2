import numpy as np
from sklearn.metrics import f1_score
from utils.post_processing import post_processing


def target_metric(y_pred, y_gt):

    threshold_grid = np.arange(0, 1, 0.05)
    scores = []

    for threshold in threshold_grid:
        y_pred_temp, y_gt_temp = post_processing(y_pred.copy(), y_gt.copy(), threshold)
        y_pred_temp = np.eye(2)[y_pred_temp.astype(np.int32)]
        y_gt_temp = np.eye(2)[y_gt_temp.astype(np.int32)]

        scores.append(f1_score(y_gt_temp, y_pred_temp, average='macro'))

    scores = np.array(scores)

    threshold = threshold_grid[np.where(scores == np.max(scores))[0][0]]

    return np.max(scores), threshold
