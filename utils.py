import torch
import numpy as np
import os
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import argparse


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Recieved {0}".format(s)
        )


def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )



def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))



def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    


def padding_collate_fn(data_batch):
    sentence_list = [] 
    input_ids_list = []
    attention_mask_list = []
    visual_list = []
    label_id_list = []
    segment_list = []
    visual_len_list = []
    source_label_list = []
    for item in data_batch:
        sentence, input_ids, attention_mask, visual, visual_len, source_label, label_id, segment = item

        sentence_list.append(sentence)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        visual_list.append(visual)
        label_id_list.append(label_id)
        segment_list.append(segment)
        visual_len_list.append(visual_len)
        source_label_list.append(source_label)

    sentence = sentence_list
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    visual = pad_sequence(visual_list, batch_first=False)
    visual = visual.permute(1, 0, 2)
    label_id = torch.stack(label_id_list)
    source_label = torch.stack(source_label_list)
    segment = segment_list
    visual_len = torch.tensor(visual_len_list)
    
    return (sentence, input_ids, attention_mask, visual, visual_len, source_label, label_id, segment)



