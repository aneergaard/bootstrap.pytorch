"""
    @author: Stanislas Chambon / Valentin Thorey
    mails: stanislas@rythm.co / valentin@rythm.co
    goal: base detector based on SS1D object
    @modified: Alexander Neergaard Olesen, alexno@stanford.edu / alexander.neergaard@gmail.com
"""

import copy
import os
from glob import glob

import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
from natsort import natsorted, ns
# torch imports
from torch.utils.data import DataLoader
from tqdm import tqdm

# event detection imports
from src.data.datasets_utils import collate
from src.models.detection_loss import *
from src.utils.matching_utils import *



class BaseDetector(nn.Module):

    def __init__(self,
                 n_classes=1,
                 num_workers=0, shuffle=True, pin_memory=True,
                 batch_size=12, epochs=5,
                 histories_path=None, weights_path=None,
                 threshold_overlap=0.5, factor_negative_mining=3,
                 default_negative_mining=10, negative_mining_mode="worst",
                 lr=1e-4, momentum=0.9, patience=10,
                 lr_decrease_patience=5, lr_decrease_factor=2.,
                 loss="simple",
                 loss_alpha=0.25, loss_gamma=2,
                 weight_loc_loss=1,
                 partial_eval=-1, resume=None):

        super(BaseDetector, self).__init__()
        # print('[DEBUG] Base detector __init__.')
        self.sizes = {}

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.epochs = epochs
        self.histories_path = histories_path
        self.weights_path = weights_path

        self.threshold_overlap = threshold_overlap
        self.factor_negative_mining = factor_negative_mining
        self.default_negative_mining = default_negative_mining
        self.negative_mining_mode = negative_mining_mode
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma

        self.lr = lr
        self.momentum = momentum
        self.patience = patience
        self.lr_decrease_patience = lr_decrease_patience
        self.lr_decrease_factor = lr_decrease_factor

        # eventness, real events
        self.n_classes = n_classes + 1
        self.weight_loc_loss = weight_loc_loss

        # evaluation parameters
        self.partial_eval = partial_eval

        self.localizations_default = []

        self.history = None
        self.model_ = None

        # Load previous saved model
        self.start_epoch = 0

        # matching parameters
        self.matching_parameters = {
            "method": "new",
            "overlap": 0.4
        }
        
        # Update resume based on weights path
        if self.resume is not None and not os.path.isfile(self.resume):
            weight_list = glob(self.resume + '_*.pth.tar')
            self.resume = natsorted(weight_list, key=lambda x: x.lower())[-2]  # We assume there's a 'best' model, which will be at the -1 position. If not, it does not matter much.

            

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
        # print('[DEBUG] Base detector finished __init__.')

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def predict(self, x):
        localizations, classifications = self.forward(x)

        return localizations, classifications

    def predict_generator(
            self, test_gen,
            detection_threshold=[0.9, 0.8, 0.7, 0.6, 0.5]):

        sfreq = test_gen.df.fs.values[0]
        window = test_gen.window

        # binary vector for true labels
        true = np.zeros(
            (self.n_classes - 1,
             test_gen.df.n_times.values[0]))

        # binary vector of pred labels
        # shape n_thres, n_classes, n_times
        pred = np.zeros(
            (len(detection_threshold), self.n_classes - 1,
             test_gen.df.n_times.values[0]))

        for idx_sample, (x, y) in enumerate(tqdm(test_gen)):

            # true vector
            y = y.numpy()
            if y.shape[0] != 0:
                for idx_event in range(y.shape[0]):
                    start = (y[idx_event, 0] + idx_sample) * window
                    end = (y[idx_event, 1] + idx_sample) * window
                    idx_class = int(y[idx_event, 2])

                    idx_start = int(start * sfreq)
                    idx_end = int(end * sfreq)

                    true[idx_class, idx_start:idx_end] = 1

            # pred vector
            x_ = x.unsqueeze(0)
            z_loc, z_clf = self.predict(Variable(x_).cuda())

            # detection here
            for idx_thres, thres in enumerate(detection_threshold):
                self.detector.set_params(classification_threshold=thres)

                z = self.detector(
                    z_loc, z_clf, self.localizations_default.cuda())

                z = np.asarray(z)
                if z.shape[1] != 0:
                    for idx_event in range(z.shape[1]):
                        start = (z[0, idx_event, 0] + idx_sample) * window
                        end = (z[0, idx_event, 1] + idx_sample) * window
                        idx_class = int(z[0, idx_event, 2])

                        idx_start = int(start * sfreq)
                        idx_end = int(end * sfreq)

                        pred[idx_thres, idx_class, idx_start:idx_end] = 1

        return true, pred

    def fit_generator(self, train_gen, val_gen):

        # if params is None:
        #     params = self.parameters()

        self.cuda()

        dataloader_parameters_train = {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "collate_fn": collate,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size
        }
        dataloader_parameters_val = {
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "collate_fn": collate,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size
        }

        train_loader = DataLoader(train_gen, **dataloader_parameters_train)
        val_loader = DataLoader(val_gen, **dataloader_parameters_val)

        # criterion
        device = torch.device("cuda")

        criterion = self.criterion(**self.loss_parameters)

        print("\n\nusing {}".format(self.loss))

        if self.matching_parameters["method"] == "new":
            matching = match_events_localization_to_default_localizations
        else:
            matching = match_events_localization_to_default_localizations_old

        current_lr = self.lr
        optimizer = optim.SGD(
            self.parameters(),
            lr=current_lr, momentum=self.momentum)

        if self.partial_eval != -1:
            print("\nPerforming eval every {} batches".format(
                self.partial_eval))

        history = dict(epoch=[], train_clf=[], train_loc=[], val_clf=[], val_loc=[])
        best_net = copy.deepcopy(self)
        best_loss_val = np.infty
        waiting = 0
        waiting_lr_decrease = 0

        # Maybe restore from checkpoint
        if self.resume:
            # pdb.set_trace()
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                best_net.load_state_dict(checkpoint['model_state_dict'])
                best_loss_val = checkpoint['best_loss_eval']
                history = checkpoint['history']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.load_state_dict(checkpoint['model_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                waiting = checkpoint['waiting']
                waiting_lr_decrease = checkpoint['waiting_lr_decrease']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))

        for epoch in range(self.start_epoch, self.epochs):
            
            # In case we have resumed at a final checkpoint
            if waiting == self.patience:
                break

            # training loop
            print("\nStarting epoch : {} / {}".format(epoch + 1, self.epochs))
            history['epoch'].append(epoch)
            bar_train = tqdm(train_loader)
            self.train()
            # scheduler.step()

            train_epoch_clf = []
            train_epoch_loc = []
            
#             pdb.set_trace()
            for idx_batch, batch in enumerate(bar_train):

                # lr decrease included in training loop
                if waiting_lr_decrease == self.lr_decrease_patience:
                    print("\nReducing lr {} -> {}".format(
                        current_lr, current_lr / self.lr_decrease_factor))

                    current_lr /= self.lr_decrease_factor
                    optimizer = optim.SGD(
                        self.parameters(),
                        lr=current_lr, momentum=self.momentum)
                    waiting_lr_decrease = 0
                    self.train()

                eeg, events = batch
                x = eeg.cuda()

                optimizer.zero_grad()

                # step 1: forward pass
                #                 print('[DEBUG] Forward pass')
                locs, clfs = self.forward(x)

                # step 2: matching
                #                 print('[DEBUG] Matching')
                localizations_target, classifications_target = matching(
                    localizations_default=self.localizations_default,
                    events=events,
                    threshold_overlap=self.matching_parameters["overlap"])
                localizations_target = localizations_target.to(device)
                classifications_target = classifications_target.to(device)

                # step 3: loss
                #                 print('[DEBUG] Loss')
                #                 pdb.set_trace()
                train_clf_pos_loss, train_clf_neg_loss, train_loc_loss = (
                    criterion(locs,
                              clfs,
                              localizations_target,
                              classifications_target))

                loss = (train_clf_pos_loss + train_clf_neg_loss) + self.weight_loc_loss * train_loc_loss
                loss.backward()
#                 with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
                optimizer.step()

                # step 4: monitoring
                train_epoch_clf.append(
                    train_clf_pos_loss.item() + train_clf_neg_loss.item())
                train_epoch_loc.append(train_loc_loss.item())

                train_epoch_clf_ = np.asarray(train_epoch_clf)
                m_clf = np.mean(
                    train_epoch_clf_[np.isfinite(train_epoch_clf_)])
                train_epoch_loc_ = np.asarray(train_epoch_loc)
                m_loc = np.mean(
                    train_epoch_loc_[np.isfinite(train_epoch_loc_)])

                bar_train.set_description(
                    'clf: {:.4f} | loc: {:.4f}'.format(
                        m_clf,
                        m_loc))

                # partial eval (useful when working with long epochs)
                if self.partial_eval != -1:
                    if idx_batch != 0:
                        if idx_batch % self.partial_eval == 0:

                            history["train_loc"].append(m_loc)
                            history["train_clf"].append(m_clf)

                            bar_val = tqdm(val_loader)
                            self.eval()

                            val_epoch_clf = []
                            val_epoch_loc = []

                            for idx_batch_, batch in enumerate(bar_val):
                                eeg, events = batch
                                x = eeg.cuda()

                                # step 1: forward pass
                                locs, clfs = self.forward(x)

                                # step 2: matching
                                localizations_target, classifications_target = matching(
                                    localizations_default=self.localizations_default,
                                    events=events,
                                    threshold_overlap=self.matching_parameters["overlap"])
                                localizations_target = localizations_target.to(device)
                                classifications_target = classifications_target.to(device)

                                # step 3: loss
                                val_clf_pos_loss, val_clf_neg_loss, val_loc_loss = (
                                    criterion(locs,
                                              clfs,
                                              localizations_target,
                                              classifications_target))

                                loss = (val_clf_pos_loss + val_clf_neg_loss) + self.weight_loc_loss * val_loc_loss

                                # step 4: monitoring
                                val_epoch_clf.append(
                                    val_clf_neg_loss.item() + val_clf_pos_loss.item())
                                val_epoch_loc.append(val_loc_loss.item())

                                val_epoch_clf_ = np.asarray(val_epoch_clf)
                                m_clf = np.mean(
                                    val_epoch_clf_[np.isfinite(val_epoch_clf_)])
                                val_epoch_loc_ = np.asarray(val_epoch_loc)
                                m_loc = np.mean(
                                    val_epoch_loc_[np.isfinite(val_epoch_loc_)])

                                bar_val.set_description(
                                    'clf: {:.4f} | loc: {:.4f}'.format(
                                        m_clf, m_loc))

                            history["val_loc"].append(m_loc)
                            history["val_clf"].append(m_clf)

                            # early stopping
                            val_loss_epoch = (self.weight_loc_loss * m_loc) + m_clf
                            if val_loss_epoch < best_loss_val:
                                print("\n\nval loss improved: {:.4f} -> {:.4f}\n".format(
                                    best_loss_val, val_loss_epoch))
                                best_loss_val = val_loss_epoch
                                best_net = copy.deepcopy(self)
                                waiting = 0
                                waiting_lr_decrease = 0
                            else:
                                print("\n\nval loss did not improved: {:.4f} < {:.4f}\n".format(
                                    best_loss_val, val_loss_epoch))
                                waiting += 1
                                waiting_lr_decrease += 1

                            if waiting == self.patience:
                                break

                            self.train()
                            train_epoch_clf = []
                            train_epoch_loc = []

            history["train_loc"].append(m_loc)
            history["train_clf"].append(m_clf)

            bar_val = tqdm(val_loader)
            self.eval()

            with torch.no_grad():
                val_epoch_clf = []
                val_epoch_loc = []

                for idx_batch, batch in enumerate(bar_val):
                    eeg, events = batch
                    x = eeg.cuda()

                    # step 1: forward pass
                    locs, clfs = self.forward(x)

                    # step 2: matching
                    localizations_target, classifications_target = matching(
                        localizations_default=self.localizations_default,
                        events=events,
                        threshold_overlap=self.matching_parameters["overlap"])
                    localizations_target = localizations_target.to(device)
                    classifications_target = classifications_target.to(device)

                    # step 3: loss
                    val_clf_pos_loss, val_clf_neg_loss, val_loc_loss = (
                        criterion(locs,
                                  clfs,
                                  localizations_target,
                                  classifications_target))

                    eval_loss = (val_clf_pos_loss + val_clf_neg_loss) + self.weight_loc_loss * val_loc_loss

                    # step 4: monitoring
                    val_epoch_clf.append(
                        val_clf_neg_loss.item() + val_clf_pos_loss.item())
                    val_epoch_loc.append(val_loc_loss.item())

                    val_epoch_clf_ = np.asarray(val_epoch_clf)
                    m_clf = np.mean(
                        val_epoch_clf_[np.isfinite(val_epoch_clf_)])
                    val_epoch_loc_ = np.asarray(val_epoch_loc)
                    m_loc = np.mean(
                        val_epoch_loc_[np.isfinite(val_epoch_loc_)])

                    bar_val.set_description(
                        'clf: {:.4f} | loc: {:.4f}'.format(
                            m_clf, m_loc))

                history["val_loc"].append(m_loc)
                history["val_clf"].append(m_clf)

                # early stopping
                val_loss_epoch = (self.weight_loc_loss * m_loc) + m_clf
                if val_loss_epoch < best_loss_val:
                    print("Eval loss improved: {:.4f} -> {:.4f}".format(
                        best_loss_val, val_loss_epoch))
                    best_loss_val = val_loss_epoch
                    best_net = copy.deepcopy(self)
                    waiting = 0
                    waiting_lr_decrease = 0

                    # Save best performing model
                    if best_net.weights_path is not None:
                        print('Saving new best model')
                        torch.save({'best_loss_eval': best_loss_val,
                                    'epoch': epoch,
                                    'history': history,
                                    'model_state_dict': self.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'waiting': self.patience,
                                    'waiting_lr_decrease': waiting_lr_decrease}, best_net.weights_path.split('.')[0] + '_best.pth.tar')
                else:
                    print("Eval loss did not improve: {:.4f} < {:.4f}".format(
                        best_loss_val, val_loss_epoch))
                    waiting += 1
                    waiting_lr_decrease += 1

                # Save training checkpoint
                if self.weights_path is not None:
                    torch.save({'best_loss_eval': best_loss_val,
                                'epoch': epoch,
                                'history': history,
                                'model_state_dict': self.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'waiting': waiting,
                                'waiting_lr_decrease': waiting_lr_decrease}, self.weights_path.split('.')[0] + '_{}.pth.tar'.format(epoch))
            
            
            if waiting == self.patience:
                break

        history = pd.DataFrame(history)

        # if self.weights_path is not None:
        #     torch.save(best_net.state_dict(), self.weights_path)

        if self.histories_path is not None:
            history.to_csv(self.histories_path)

        self = best_net
        self.history = history

        return self
