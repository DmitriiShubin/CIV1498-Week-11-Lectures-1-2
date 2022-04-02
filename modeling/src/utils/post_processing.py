def post_processing(y_pred, y_gt, threshold):
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    return y_pred, y_gt
