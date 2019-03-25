import os
import resource
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datasets.datasets_utils import collate

resource.setrlimit(
    resource.RLIMIT_NOFILE, (8192, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


class MrOS(data.Dataset):

    def __init__(self,
                 adjust_probs=False,
                 batch_size=100,
                 cohort='mros',
                 dir_data='data',
                 detection_threshold=[x / 10.0 for x in range(9, 0, -1)],
                 downsampling=1,
                 events=['arousal', 'limb'],
                 index_on_events=False,
                 minimum_overlap=0.5,
                 multiclass=False,
                 nb_threads=0,
                 num_records=500,
                 num_records_test=100,
                 num_records_train=None,
                 pin_memory=True,
                 ratio_positive=0.5,
                 selected_channels=['C3', 'C4', 'EOGL', 'EOGR', 'EMG', 'LEG'],
                 split='train',
                 window=2 * 60  # in minutes
                 ):

        self.batch_size = batch_size
        self.cohort = cohort
        self.dir_data = dir_data
        self.events = events
        self.nb_threads = nb_threads
        self.num_records = num_records
        self.num_records_test = num_records_test
        self.num_records_train = num_records_train
        self.pin_memory = pin_memory
        self.split = split

        # Load DataFrame containing cohort studies
        try:
            df = pd.read_csv(
                './data/processed/{}/mm_info.csv'.format(self.cohort), index_col=0)
        except FileNotFoundError:
            df = sorted(
                glob('./data/processed/{}/mm_info_*.csv'.format(self.cohort)))
            df = pd.concat([pd.read_csv(f, index_col=0)
                            for f in df], ignore_index=True).sort_values('record')

        # Select specific events
        df = df[np.isin(
            df.event, self.events)].reset_index(drop=True)

        # Select specific number of subjects
        df = df[np.isin(df.record.values, df.record.unique()
                        [:self.num_records])].reset_index(drop=True)

        # Migrated from datagen.py
        self.selected_channels = selected_channels
        self.class_labels = pd.unique(df.event).tolist()
        self.multiclass = multiclass
        self.number_of_classes = len(self.class_labels) + 1
        self.adjust_probs = adjust_probs
        self.window = window
        self.downsampling = downsampling
        self.fs = pd.unique(df.fs)[0]
        self.fs_downsample = self.fs / self.downsampling
        self.window_size = int(self.window * self.fs)
        self.window_size_downsample = int(self.window * self.fs_downsample)
        self.input_size = self.window_size_downsample
        self.ratio_positive = ratio_positive
        self.index_on_events = index_on_events
        self.minimum_overlap = minimum_overlap
        self.channels_ = ['C3', 'C4', 'EOGL', 'EOGR',
                          'EMG', 'LEG', 'NASAL', 'THOR', 'ABDO']
        self.number_of_channels = len(self.channels_)

        # Split data
        records = sorted(pd.unique(df.record))
        if self.num_records_train is None and self.num_records_test is not None:
            self.num_records_train = len(records) - self.num_records_test
        elif self.num_records_test is None and self.num_records_train is not None:
            self.num_records_test = len(records) - self.num_records_train
        else:
            raise ValueError(
                'Specify either number of training records or number of test records, but not both!')
        r_test = records[0: num_records_test]
        r_ = sorted([r for r in records if r not in r_test])
        r_train, r_eval = train_test_split(
            r_, train_size=0.9,
            test_size=0.1,
            random_state=0)

        if self.split == 'train':
            is_train = True
            self.shuffle = True
            print("Training on {} records".format(len(r_train)))
            self.df = df[df.record.isin(r_train)].reset_index(drop=True)
        elif self.split == 'eval':
            is_train = False
            self.shuffle = False
            print("Validation on {} records".format(len(r_eval)))
            self.df = df[df.record.isin(r_eval)].reset_index(drop=True)
        elif self.split == 'test':
            is_train = False
            self.shuffle = False
            print("Prediction on {} records".format(len(r_test)))
            self.df = df[df.record.isin(r_test)].reset_index(drop=True)
        else:
            raise ValueError()
        self.r_eval = r_eval
        self.r_test = r_test
        self.r_train = r_train

        # Preload memmaps for signal and events
        self.psgs = {k: {'data': None,
                         # 'means': None,
                         # 'stds': None,
                         'number_of_windows': None} for k in self.df.record.unique()}
        # self.psgs = {}for event in pd.unique(self.df.event)
        self.events = {k: {event: {'data': None,
                                   'label': None} for event in pd.unique(self.df.event)} for k in self.df.record.unique()}
        self.events_number_of_event = {
            k: None for k in self.df.record.unique()}

        # Initialize channel indices
        idx_channels = []
        for ic, c in enumerate(self.channels_):
            for c_ in self.selected_channels:
                if c in c_:
                    idx_channels.append(ic)
        self.idx_channels = np.asarray(idx_channels).astype(np.int)
        print('Preloading memmaps...')
        time.sleep(0.5)

        for r in tqdm(pd.unique(self.df.record)):
            # self.psgs[r] = {}

            psg_f = pd.unique(self.df[self.df.record == r].psg_file)[0]
            n_times = pd.unique(self.df[self.df.record == r].n_times)[0]

            # from datetime import datetime
            # elapsed_time = []
            # for k in range(10):
            #     start = datetime.now()
            self.psgs[r]["data"] = np.memmap(
                psg_f,
                dtype='float32',
                mode='r',
                shape=(self.number_of_channels, n_times))
            # elapsed_time.append(datetime.now() - start)
            # print('Mean load time with all channels: {}'.format(np.mean(elapsed_time)))
            # elapsed_time = []
            # for k in range(10):
            #     start = datetime.now()
            #     self.psgs[r]["data"] = np.memmap(
            #         psg_f,
            #         dtype='float32',
            #         mode='r',
            #         shape=(self.number_of_channels, n_times))
            #     self.psgs[r]['data'] = self.psgs[r]['data'][idx_channels, ::self.downsampling]
            #     elapsed_time.append(datetime.now() - start)
            # print('Mean load time with selected channels: {}'.format(np.mean(elapsed_time)))

            # normalization parameters
            # self.psgs[r]['means'] = np.mean(
            #     self.psgs[r]["data"], axis=1, keepdims=True)
            # self.psgs[r]['stds'] = np.std(
            #     self.psgs[r]["data"], axis=1, keepdims=True)

            # self.psgs[r]["idx_channels"] = idx_channels
            self.psgs[r]["number_of_windows"] = (
                self.psgs[r]["data"].shape[1] // self.window_size)

            # self.events[r] = {}
            event_count = 0
            for event in pd.unique(self.df.event):

                if self.df[(self.df.record == r) & (self.df.event == event)].shape[0] != 0:
                    event_f = self.df[(self.df.record == r) & (
                        self.df.event == event)].event_file.values[0]
                    n_events = self.df[(self.df.record == r) & (
                        self.df.event == event)].n_events.values[0]
                    event_label = self.df[(self.df.record == r) & (
                        self.df.event == event)].label.values[0]

                    if not os.path.isfile(event_f):
                        continue
                    event_count += n_events
                    self.events[r][event] = {}
                    self.events[r][event]["data"] = np.memmap(
                        event_f,
                        dtype='float32',
                        mode='r',
                        shape=(2, n_events))
                    self.events[r][event]["label"] = float(event_label)
            self.events_number_of_event[r] = event_count

        # for each index find correct filename
        self.index_to_record = []

        # for each inde give offset in record
        self.index_to_record_index = []
        if self.index_on_events:
            # from datetime import datetime
            # start = datetime.now()
            # self.index_to_record = [record * for record, n_events in self.events_number_of_event.items()]
            for record, n_events in self.events_number_of_event.items():
                self.index_to_record.extend([record] * n_events)
                self.index_to_record_index.extend(range(n_events))  # useless
            # print('Elapsed time: {}'.format(datetime.now() - start))
        else:
            for record, psg in self.psgs.items():
                self.index_to_record.extend(
                    [record] * psg["number_of_windows"])
                self.index_to_record_index.extend(
                    range(psg["number_of_windows"]))

        # set data extractor:
        self.class_probabilities = None
        if self.ratio_positive in [None, False]:
            self.extract_data = self._get_sample
        elif self.ratio_positive not in [None, False] and self.multiclass:
            self.extract_data = self._extract_balanced_multiclass_data

            # Adjust event sampling probabilities
            self.class_probabilities = {
                k: 1 / self.number_of_classes for k in self.class_labels}
        else:
            self.extract_data = self._extract_balanced_data
        print('[INFO] Using {} for sampling'.format(self.extract_data))

    def _extract_balanced_data(self, psg, events, index=None):
        """ Extract a particular index or random
        """
        psg_data, events_data = self._get_sample(psg, events, index=None)
        choice = np.random.choice(
            [0, 1], p=[1 - self.ratio_positive, self.ratio_positive])
        if choice == 0:
            while len(events_data) > 0:
                psg_data, events_data = self._get_sample(
                    psg, events, index=None)
        else:
            while len(events_data) == 0:
                psg_data, events_data = self._get_sample(
                    psg, events, index=None)

        return psg_data, events_data

    def _extract_balanced_multiclass_data(self, psg, events, index=None):
        """ Extract balanced data in a multi-class problem. This function samples the events and shifts the extracted
        window based on a specific sample.
        """
        # Get the class probabilities
        C = self.number_of_classes
        default_prob = 1 / C
        class_probs = self.class_probabilities
        num_events_class = {k: events[k]['data'].shape[1]
                            for k in self.class_probabilities.keys() if events[k]['data'] is not None}
        if len(num_events_class.keys()) + 1 != C:  # Some classes are missing in the PSG
            class_probs = {k: 1 / (len(num_events_class.keys()) + 1)
                           for k in num_events_class.keys()}
            default_prob = 1 / (len(num_events_class.keys()) + 1)

        # Maybe adjust probabilities
        if self.adjust_probs:
            class_probs_adjusted = class_probs[1:]
            class_probs_adjusted = (
                class_probs_adjusted / Nc) / (C * np.sum(class_probs_adjusted / Nc) / (C - 1))
            class_probs = np.insert(
                class_probs_adjusted, 0, self.class_probs[0], axis=0).squeeze()

        # Randomly sample data
        choice = np.random.choice([None] + [k for k in class_probs.keys()],
                                  p=[default_prob] + [v for v in class_probs.values()])
        if choice is not None:
            random_event_idx = np.random.randint(
                events[choice]['data'].shape[1])
            event_midpoint = np.mean(
                events[choice]['data'][:, random_event_idx])
            sample_idx = np.random.randint((event_midpoint - self.window) * self.fs,
                                           (event_midpoint + self.window) * self.fs)
            if sample_idx < 0:
                sample_idx = 0
            elif sample_idx > psg.shape[1] - self.window_size:
                sample_idx = psg.shape[1] - self.window_size
            if (sample_idx + self.window_size) >= psg.shape[1]:
                print('PSG shape: {}'.format(psg.shape))
                print('Sample idx: {}'.format(sample_idx))
                print('Window size: {}'.format(self.window_size))
                print('End idx: {}'.format(self.window_size + sample_idx))
            psg_data, events_data = self._get_sample(
                psg, events, sample_idx=sample_idx)
        else:
            psg_data, events_data = self._get_sample(psg, events, index=None)
#         print(psg_data.shape, len(events_data))
        if psg_data.shape[1] == 0:
            print('PSG shape: {}'.format(psg.shape))
            print('Sample idx: {}'.format(sample_idx))
            print('Window size: {}'.format(self.window_size))
            print('End idx: {}'.format(self.window_size + sample_idx))
            print('Choice: {}'.format(choice))
        return psg_data, events_data

    def _get_sample(self, psg, events, index=None, sample_idx=None):
        if sample_idx is not None:
            index = sample_idx
        elif index is None:
            index = np.random.randint(psg.shape[1] - self.window_size)
        else:
            index = index * self.window_size

        psg_data = np.vstack(
            psg[self.idx_channels, index:index + self.window_size:self.downsampling])

        events_data = []
        for event_name, event in events.items():
            if event['data'] is None:
                continue
            try:
                starts, durations = event["data"][0, :] * \
                    self.fs, event["data"][1, :] * self.fs
            except TypeError:
                print('hej')
            # Relative start stop
            starts_relative = (starts - index) / self.window_size
            durations_relative = durations / self.window_size
            stops_relative = starts_relative + durations_relative

            # Find valid start or stop
            valid_starts_index = np.where(
                (starts_relative > 0) * (starts_relative < 1))[0]
            valid_stops_index = np.where(
                (stops_relative > 0) * (stops_relative < 1))[0]

            # merge them
            valid_indexes = set(
                list(valid_starts_index) + list(valid_stops_index))

            # Annotations contains valid index with minimum overlap requirement
            for valid_index in valid_indexes:
                if (valid_index in valid_starts_index) and (valid_index in valid_stops_index):
                    events_data.append((float(starts_relative[valid_index]),
                                        float(stops_relative[valid_index]), event["label"]))
                elif valid_index in valid_starts_index:
                    if ((1 - starts_relative[valid_index]) / durations_relative[valid_index]) > self.minimum_overlap:
                        events_data.append(
                            (float(starts_relative[valid_index]), 1, event["label"]))

                elif valid_index in valid_stops_index:
                    if ((stops_relative[valid_index]) / durations_relative[valid_index]) > self.minimum_overlap:
                        events_data.append(
                            (0, float(stops_relative[valid_index]), event["label"]))

        return psg_data, events_data

    def __getitem__(self, index):
        r = self.index_to_record[index]
        record_index = self.index_to_record_index[index]
        psg, events = self.extract_data(
            self.psgs[r]["data"], self.events[r], record_index)
        return ToTensor()(psg, events)

    def __len__(self):
        return len(self.index_to_record)

    def make_batch_loader(self):
        data_loader = data.DataLoader(self,
                                      batch_size=self.batch_size,
                                      num_workers=self.nb_threads,
                                      shuffle=self.shuffle,
                                      pin_memory=self.pin_memory,
                                      collate_fn=collate,
                                      drop_last=True)
        return data_loader


class ToTensor:

    def __init__(self):
        pass

    def __call__(self, eeg, events):
        return torch.FloatTensor(eeg), torch.FloatTensor(events)


if __name__ == '__main__':

    train_dataset = MrOS(cohort='mros', split='train')
    # eval_dataset = MrOS(cohort='mros', split='eval')
    # test_dataset = MrOS(cohort='mros', split='test')

    for psg, event in tqdm(train_dataset, total=train_dataset.__len__()):
        pass

    print('')
