"""
    @author: Valentin Thorey
    mail: valentin@rythm.co
    goal: detection object implementation
    Loss function inspired from https://github.com/amdegroot/ssd.pytorch
"""

import torch.nn as nn
from sklearn.base import BaseEstimator

from src.utils.decode import decode
from src.utils.non_maximum_suppression import non_maximum_suppression


class Detection(nn.Module, BaseEstimator):
    """"""

    def __init__(self,
                 number_of_classes,
                 overlap_non_maximum_suppression,
                 top_k_non_maximum_suppression,
                 classification_threshold,
                 use_argmax=False
                 ):
        super(Detection, self).__init__()
        self.number_of_classes = number_of_classes
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.top_k_non_maximum_suppression = top_k_non_maximum_suppression
        self.classification_threshold = classification_threshold
        self.use_argmax = use_argmax

    def forward(self, localizations, classifications, localizations_default):
        batch = localizations.size(0)
        scores = nn.Softmax(dim=2)(classifications)
        results = []

        if self.use_argmax:
            _, idx_label = scores.max(dim=-1)
            for i in range(batch):
                result = []
                localization_decoded = decode(
                    localizations[i].data, localizations_default)

                for class_index in range(1, self.number_of_classes):

                    # check that some events are annotated
                    # ie check that there are labels different from 0
                    if (idx_label[i] == class_index).data.float().sum() == 0:
                        continue

                    (idx_label[i] == class_index)

                    # change appears here
                    mask = (idx_label[i] == class_index).data.long().nonzero()

                    scores_batch_class = scores[i, :, class_index].data
                    scores_batch_class_selected = scores_batch_class[
                        mask.squeeze()]

                    localizations_decoded_selected = localization_decoded[
                                                     mask, :].view(-1, 2)

                    result.extend(
                        [[x[0], x[1], class_index - 1]
                         for x in non_maximum_suppression(
                            localizations_decoded_selected,
                            scores_batch_class_selected,
                            overlap=self.overlap_non_maximum_suppression,
                            top_k=self.top_k_non_maximum_suppression)])
                results.append(result)

        else:
            for i in range(batch):
                result = []
                localization_decoded = decode(
                    localizations[i].data, localizations_default)
                for class_index in range(1, self.number_of_classes):
                    scores_batch_class = scores[i, :, class_index].data
                    scores_batch_class_selected = scores_batch_class[
                        scores_batch_class > self.classification_threshold]
                    if len(scores_batch_class_selected) == 0:
                        continue
                    localizations_decoded_selected = localization_decoded[
                        (scores_batch_class > self.classification_threshold)
                            .unsqueeze(1).expand_as(localization_decoded)
                    ].view(-1, 2)
                    result.extend(
                        [[x[0], x[1], class_index - 1]
                         for x in non_maximum_suppression(
                            localizations_decoded_selected,
                            scores_batch_class_selected,
                            overlap=self.overlap_non_maximum_suppression,
                            top_k=self.top_k_non_maximum_suppression)])
                results.append(result)
        return results
