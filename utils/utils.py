import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_recall_curve
import copy
# Division by zero is normal in f score calculation
np.seterr(divide='ignore')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_evaluation_stats_kim(ground_truth, predictions, kim_point_adjustment_percentile):
    """
    Implements the evaluation proposed in:
    Kim, Siwon, et al.
    "Towards a rigorous evaluation of time-series anomaly detection."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 7. 2022.
    """

    new_predictions = np.zeros(predictions.shape[0])

    i = 0
    while i < ground_truth.shape[0]:
        if ground_truth[i] == 0:
            new_predictions[i] = predictions[i]
            i += 1
        else:
            anomaly_block_length = 0
            while ground_truth[i + anomaly_block_length] == 1:
                anomaly_block_length += 1
            anomaly_scores_in_block = copy.deepcopy(predictions[i:(i + anomaly_block_length)])
            threshold = np.percentile(anomaly_scores_in_block, 100.0 - kim_point_adjustment_percentile)
            indices_below_threshold = anomaly_scores_in_block < threshold
            anomaly_scores_in_block[indices_below_threshold] = threshold
            new_predictions[i:(i + anomaly_block_length)] = anomaly_scores_in_block
            i += anomaly_block_length

    precision, recall, thresholds = precision_recall_curve(ground_truth, new_predictions)
    f1_scores = 2 / ( (1/precision) + (1/recall))
    index_of_best_f1 = np.argmax(f1_scores)

    return precision[index_of_best_f1], recall[index_of_best_f1], f1_scores[index_of_best_f1]
