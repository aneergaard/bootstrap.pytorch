"""
    @author: Valentin Thorey
    mail: valentin@rythm.co
"""

import torch


def collate(batch):
    """collate fn because unconsistent number of events"""
    batch_events = []
    batch_eegs = []
    for eeg, events in batch:
        batch_eegs.append(eeg)
        batch_events.append(events)
    try:
        return torch.stack(batch_eegs, 0), batch_events
    except RuntimeError:
        print('PSG shape: {}'.format([eeg.shape for eeg in batch_eegs]))
        print('Event shape: {}'.format([len(ev) for ev in batch_events]))
        