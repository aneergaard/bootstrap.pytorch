"""
    @author: Stanislas Chambon / Valentin Thorey
    mails: stanislas@rythm.co / valentin@rythm.co
    goal: first spindle detector
    @modified: Alexander Neergaard Olesen, alexno@stanford.edu / alexander.neergaard@gmail.com
"""
import os
from collections import OrderedDict

from torch.autograd import Variable

from src.models.base_detector import BaseDetector
from src.models.detection import Detection
from src.models.detection_loss import *


def get_overlapping_default_events(
        window_size, default_event_sizes, factor_overlap=2):
    window_size = window_size
    default_event_sizes = default_event_sizes
    factor_overlap = factor_overlap
    default_events = []
    for default_event_size in default_event_sizes:
        overlap = default_event_size / factor_overlap
        number_of_default_events = int(window_size / overlap)
        default_events.extend(
            [(overlap * (0.5 + i) / window_size,
              default_event_size / window_size)
             for i in range(number_of_default_events)]
        )
    return torch.Tensor(default_events)


class EventDetector(BaseDetector):

    def __init__(self, n_times=7680, n_channels=1, fs=256,
                 n_classes=1,
                 overlap_non_maximum_suppression=0.4,
                 top_k_non_maximum_suppression=200,
                 classification_threshold=0.7,
                 num_workers=0, shuffle=True, pin_memory=True,
                 batch_size=32, epochs=100,
                 histories_path=None, weights_path=None,
                 threshold_overlap=0.5, factor_negative_mining=3,
                 default_negative_mining=10, negative_mining_mode="worst",
                 lr=1e-4, momentum=0.9, patience=10,
                 lr_decrease_patience=5, lr_decrease_factor=2.,
                 loss="simple",
                 loss_alpha=0.25, loss_gamma=2,
                 k_max=8, max_pooling=2,
                 default_event_sizes=[1 * 256], factor_overlap=4,
                 weight_loc_loss=1,
                 partial_eval=-1, resume=None):

        super(EventDetector, self).__init__()
        self.n_times = n_times
        self.n_channels = n_channels
        self.fs = fs
        self.overlap_non_maximum_suppression = overlap_non_maximum_suppression
        self.top_k_non_maximum_suppression = top_k_non_maximum_suppression
        self.classification_threshold = classification_threshold

        self.threshold_overlap = threshold_overlap
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining
        self.negative_mining_mode = negative_mining_mode
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.max_pooling = max_pooling

        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.lr_decrease_patience = lr_decrease_patience
        self.lr_decrease_factor = lr_decrease_factor

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.epochs = epochs
        self.histories_path = histories_path
        self.weights_path = weights_path
        self.k_max = k_max
        self.factor_overlap = factor_overlap
        self.weight_loc_loss = weight_loc_loss

        self.partial_eval = partial_eval

        # eventness, real events
        self.n_classes = n_classes + 1
        self.history = None
        self.resume = resume

        # matching parameters
        self.matching_parameters = {
            "method": "new",
            "overlap": 0.4
        }

        # loss parameter
        self.loss = loss
        if self.loss == "simple":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
            }

            self.criterion = DOSEDSimpleLoss

        elif self.loss == "worst_negative_mining":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "factor_negative_mining": self.factor_negative_mining,
                "default_negative_mining": self.default_negative_mining,
            }

            self.criterion = DOSEDWorstNegativeMiningLoss

        elif self.loss == "random_negative_mining":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "factor_negative_mining": self.factor_negative_mining,
                "default_negative_mining": self.default_negative_mining,
            }

            self.criterion = DOSEDRandomNegativeMiningLoss

        elif self.loss == "focal":
            self.loss_parameters = {
                "number_of_classes": self.n_classes,
                "alpha": self.loss_alpha,
                "gamma": self.loss_gamma,
            }

            self.criterion = DOSEDFocalLoss

        self.detector = Detection(
            number_of_classes=self.n_classes,
            overlap_non_maximum_suppression=self.overlap_non_maximum_suppression,
            top_k_non_maximum_suppression=self.top_k_non_maximum_suppression,
            classification_threshold=self.classification_threshold)

        self.localizations_default = get_overlapping_default_events(
            window_size=n_times,
            default_event_sizes=default_event_sizes,
            factor_overlap=self.factor_overlap
        )
        print(len(self.localizations_default))

        if self.n_channels != 1:
            self.spatial_filtering = nn.Conv2d(
                1, self.n_channels, (self.n_channels, 1))

        # first block
        self.block_1 = nn.Sequential(
            OrderedDict([
                ("conv_{}".format(1), nn.Conv2d(
                    in_channels=1,
                    out_channels=8,
                    kernel_size=(1, 3))),
                ("padding_{}".format(1),
                 nn.ConstantPad2d([1, 1, 0, 0], 0)),
                ("batchnorm_{}".format(1),
                 nn.BatchNorm2d(4 * (2 ** 1))),
                ("relu_{}".format(1), nn.ReLU()),
                ("max_pooling_{}".format(1),
                 nn.MaxPool2d(kernel_size=(1, self.max_pooling)))]))

        # other blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict([
                        ("conv_{}".format(k), nn.Conv2d(
                            in_channels=4 * (2 ** (k - 1)),
                            out_channels=4 * (2 ** k),
                            kernel_size=(1, 3))),
                        ("padding_{}".format(k),
                         nn.ConstantPad2d([1, 1, 0, 0], 0)),
                        ("batchnorm_{}".format(k),
                         nn.BatchNorm2d(4 * (2 ** k))),
                        ("relu_{}".format(k), nn.ReLU()),
                        ("max_pooling_{}".format(k),
                         nn.MaxPool2d(kernel_size=(1, self.max_pooling)))
                    ])
                ) for k in range(2, self.k_max + 1)
            ]
        )

        # print('in_channels = {}'.format(4 * (2 ** k_max)))
        # print('out_channels = {}'.format(2 * len(self.localizations_default)))
        # print('kernel_size = {}'.format((self.n_channels, n_times // (self.max_pooling ** self.k_max))))
        self.localization = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * (2 ** k_max),
                out_channels=2 * len(self.localizations_default),
                kernel_size=(self.n_channels, n_times // (self.max_pooling ** self.k_max)))
        )
        
        # print('[DEBUG] Event detector __init__ classification.')
        # print('in_channels = {}'.format(4 * (2 ** k_max)))
        # print('out_channels = {}'.format(self.n_classes * len(self.localizations_default)))
        # print('kernel_size = {}'.format((self.n_channels, n_times // (self.max_pooling ** self.k_max))))
        self.classification = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * (2 ** k_max),
                out_channels=self.n_classes * len(self.localizations_default),
                kernel_size=(self.n_channels, n_times // (self.max_pooling ** self.k_max)))
        )
        # print('[DEBUG] Event detector finished __init__.')

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        """forward

        Parameters
        ----------
        x : tensor, shape (n_samples, C, T)
            Input tensor

        Returns:
        --------
        loc : tensor, shape (n_samples, n_default_events * 2)
            Tensor of locations
        clf : tensor, shape (n_samples, n_default_events * n_classes)
            Tensor of probabilities
        """
        batch = x.size(0)

        size = x.size()
        x = x.view(size[0], 1, size[1], size[2])

        # perform spatial filtering if more than 1 channel
        if self.n_channels != 1:
            x = self.spatial_filtering(x)
            x = x.transpose(2, 1)

        z = self.block_1(x)

        for block in self.blocks:
            z = block(z)

        loc = self.localization(z).squeeze().view(batch, -1, 2)
        clf = self.classification(z).squeeze().view(batch, -1, self.n_classes)

        return loc, clf


if __name__ == "__main__":
    n_channels = 1
    n_times = 20 * 128
    n_classes = 2
    model = EventDetector(
        n_times=n_times,
        n_channels=n_channels,
        k_max=8,
        factor_overlap=4,
        n_classes=n_classes, resume=True).cuda()

    x = np.random.randn(10, n_channels, n_times)
    x = Variable(torch.from_numpy(x).float()).cuda()
    # x = torch.from_numpy(x).float().cuda()

    z = model(x)
    print(z[0].shape, z[1].shape)
    # print(z.shape)
    # print(z[0].shape, z[1].shape, z[2].shape)
    # print(z[0].shape, z[1].shape, z[2].shape, z[3].shape)
