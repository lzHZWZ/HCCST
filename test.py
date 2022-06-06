import pickle
import sys
import torch
import numpy as np
from network import HashModel, CenterModel

from common import utils
from common.logger import Logger

from evaluate.measure_utils import get_precision_recall_by_Hamming_Radius_optimized, mean_average_precision
import options

def Hash_center_multilables(labels,
                            Hash_center_pre,
                            center_weight):  
    hash_centers = torch.FloatTensor(torch.FloatStorage())
    for (i, label) in enumerate(labels):
        one_labels = (label == 1).nonzero()  
        one_labels = one_labels.squeeze(1)
        Centers = Hash_center_pre[one_labels]
        center_weight_one = center_weight[i][one_labels]
        center_mean = torch.sum(Centers * center_weight_one.view(-1, 1), dim=0)

        hash_centers = torch.cat((hash_centers, center_mean.view(1, -1)), dim=0)
    return hash_centers


def centerQualityCheck(hashCenter_independent, interClass_loss):
    Logger.info("inter class loss {}".format(interClass_loss))
    hashCenter_independent = torch.sign(hashCenter_independent)
    ip = calc_ham_dist(hashCenter_independent, hashCenter_independent)
    ip = ip - torch.triu(ip)
    Logger.info("inter class loss coarse {}".format(ip.sum()))
    zeroStd_num = hashCenter_independent.shape[1] - (torch.std(hashCenter_independent, dim=0) > 0).sum()
    Logger.info("same hash bit num {}".format(zeroStd_num))


def calc_ham_dist(outputs1, outputs2):
    ip = torch.mm(outputs1, outputs2.t())
    mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
    cos = ip / mod.sqrt()
    hash_bit = outputs1.shape[1]
    dist_ham = hash_bit / 2.0 * (1.0 - cos)
    return dist_ham


def interClass_loss(hashCenter_independent):
    ip = calc_ham_dist(hashCenter_independent, hashCenter_independent)
    ip = ip - torch.triu(ip)
    centerQualityCheck(hashCenter_independent, ip.sum())
    return ip.sum()


class Option():
    def __init__(self):
        self.R = 18000
        self.T = 0.


def calculatePrecision_R():
    option = Option()
    state = {}
    test_code, test_lables = utils.loadHashPool(option, state,
                                                '../data/coco/coco_64bit_38e_[0910-16_53_17]_testbase.pkl')

    database_code, database_lables = utils.loadHashPool(option, state,
                                                        '../data/coco/coco_64bit_38e_[0910-16_53_17]_database.pkl',
                                                        type='database')

    map, r, p = mean_average_precision(database_code.detach().numpy(), test_code.detach().numpy(),
                                       database_lables.detach().numpy(), test_lables.detach().numpy(), option)
    print("map: {},r: {},p: {}".format(map, r, p))


