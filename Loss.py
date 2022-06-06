import pickle

import torch.nn as nn
import torch, sys
import torch.nn.functional as F
from common.logger import Logger


class Loss(nn.Module):
    def __init__(self, option, state):
        super(Loss, self).__init__()
        self.option = option
        self.cal_centerLoss = CenterLoss(option, state, option.radius, option.gamma, option.beta)

    def forward(self, hashCode, hashCenter, centerWeight, labels):
        centerLoss = self.cal_centerLoss(hashCode, hashCenter, labels, centerWeight)
        quantizationLoss = torch.mean((torch.abs(hashCode) - 1.0) ** 2)
        temp = centerWeight * torch.log(centerWeight)
        temp = torch.where(torch.isnan(temp), self.useGPU(torch.tensor(0., requires_grad=False)), temp)
        centerWeight_Regulation = torch.sum(temp, dim=1).mean()

        return centerLoss + self.option.lambda_Q * quantizationLoss + self.option.lambda_R * centerWeight_Regulation

    def useGPU(self, x):
        if self.option.use_gpu and torch.cuda.is_available():
            return x.cuda()
        else:
            return x



class HashCenterLoss(nn.Module):

    def __init__(self, option, state):
        super(HashCenterLoss, self).__init__()
        self.option = option
        self.state = state
        self.centerLoss = CenterLoss(self.option, self.state)
        self.is_multi_label = True if option.data_name == 'voc2012' or option.data_name == 'coco' or option.data_name == 'nuswide' else False

    def forward(self, word_embedding, hashCenter_independent, centerWeight, path):
        file = open(path, 'rb')
        center_loss = 0.
        index = 0
        bigBatch_size = 100 
        while True:
            try:
                data = pickle.load(file)
                if index % bigBatch_size == 0:
                    if index != 0:
                        center_loss = center_loss + self.intraClass_loss(hashcode_bigBatch,
                                                                         hashCenter_corresponding_bigBatch,
                                                                         label_bigBatch,
                                                                         centerWeight_bigBatch)
                    hashcode_bigBatch = self.useGPU(data['output'])
                    label_bigBatch = self.useGPU(data['target'])
                    centerWeight_bigBatch = self.useGPU(data['weight'])
                    hashCenter_corresponding_bigBatch = self.useGPU(
                        self.getHashCenters(label_bigBatch, hashCenter_independent, centerWeight_bigBatch))

                else:
                    hashcode_batch = self.useGPU(data['output'])
                    label_batch = self.useGPU(data['target'])
                    centerWeight_batch = self.useGPU(data['weight'])
                    hashCenter_corresponding_batch = self.useGPU(
                        self.getHashCenters(label_batch, hashCenter_independent, centerWeight_batch))

                    hashcode_bigBatch = torch.cat((hashcode_bigBatch, hashcode_batch), dim=0)
                    label_bigBatch = torch.cat((label_bigBatch, label_batch), dim=0)
                    centerWeight_bigBatch = torch.cat((centerWeight_bigBatch, centerWeight_batch), dim=0)
                    hashCenter_corresponding_bigBatch = torch.cat(
                        (hashCenter_corresponding_bigBatch, hashCenter_corresponding_batch), dim=0)

                index += 1
            except Exception:
                center_loss = center_loss + self.intraClass_loss(hashcode_bigBatch,
                                                                 hashCenter_corresponding_bigBatch,
                                                                 label_bigBatch,
                                                                 centerWeight_bigBatch)
                break
        print('UpdateCenters => center_loss:', center_loss)
        print('UpdateCenters => interClass_loss:', self.interClass_loss(hashCenter_independent) )
        print('UpdateCenters => pairwiseLoss:', self.pairwiseLoss(word_embedding, hashCenter_independent))
        loss = self.option.alpha_0 * center_loss / (
                    float(index) / bigBatch_size) - self.option.alpha_1 * self.interClass_loss(
            hashCenter_independent) + self.option.alpha_2 * self.pairwiseLoss(word_embedding, hashCenter_independent)
        return loss

    def getHashCenters(self, labels, hash_centers, center_weight):
        if self.is_multi_label:
            hash_centers = self.Hash_center_multilables(labels, hash_centers, center_weight=center_weight)
        else:
            hash_label = (labels == 1).nonzero()[:, 1]
            hash_centers = hash_centers[hash_label]
        return hash_centers

    def Hash_center_multilables(self, labels,
                                Hash_center_pre,
                                center_weight):  
        hash_centers = self.useGPU(torch.FloatTensor(torch.FloatStorage()))
        for (i, label) in enumerate(labels):
            one_labels = (label == 1).nonzero() 
            one_labels = one_labels.squeeze(1)
            Centers = Hash_center_pre[one_labels]
            center_weight_one = center_weight[i][one_labels]
            center_mean = torch.sum(Centers * center_weight_one.view(-1, 1), dim=0)
            hash_centers = torch.cat((hash_centers, center_mean.view(1, -1)), dim=0)
        return hash_centers

    def intraClass_loss(self, hashCode, hashCenter_corresponding, labels, centerWeight):
        return self.centerLoss(hashCode, hashCenter_corresponding, labels, centerWeight)
        pass

    def centerQualityCheck(self, hashCenter_independent, interClass_loss):
        Logger.info("inter class loss {}".format(interClass_loss))
        hashCenter_independent = torch.sign(hashCenter_independent)
        ip = self.calc_ham_dist(hashCenter_independent, hashCenter_independent)
        ip = ip - torch.triu(ip)
        Logger.info("inter class loss coarse {}".format(ip.sum()))
        zeroStd_num = hashCenter_independent.shape[1] - (torch.std(hashCenter_independent, dim=0) > 0).sum()
        Logger.info("same hash bit num {}".format(zeroStd_num))
        self.state['interClass_loss'].append(interClass_loss)

    def interClass_loss(self, hashCenter_independent):
        ip = self.calc_ham_dist(hashCenter_independent, hashCenter_independent)
        ip = ip - torch.triu(ip)
        self.centerQualityCheck(hashCenter_independent, ip.sum())
        return ip.sum()

    def calc_ham_dist(self, outputs1, outputs2):
        ip = torch.mm(outputs1, outputs2.t())
        mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
        cos = ip / mod.sqrt()
        hash_bit = outputs1.shape[1]
        dist_ham = hash_bit / 2.0 * (1.0 - cos)
        return dist_ham

    def smoothMaximum(self, x, y, beta):
        if torch.cuda.is_available() and self.option.use_gpu:
            e_beta_x = torch.exp(beta * x.to(torch.float64)).cuda()
            e_beta_y = torch.exp(beta * y.to(torch.float64)).cuda()
        else:
            e_beta_x = torch.exp(beta * x.to(torch.float64))
            e_beta_y = torch.exp(beta * y.to(torch.float64))
        return (((torch.mul(x.to(torch.float64), e_beta_x)) + (torch.mul(y.to(torch.float64), e_beta_y))) / (
                e_beta_x + e_beta_y)).to(torch.float32)

    def calc_ham_similarity(self, outputs1, outputs2):
        ip = torch.mm(outputs1, outputs2.t())
        mod = torch.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
        cos = ip / mod.sqrt()
        return 0.5 * (cos + 1.0)

    def pairwiseLoss(self, word_embedding, hashCenter_independent):
        ip = torch.matmul(word_embedding, word_embedding.t())
        length = torch.sqrt(torch.sum(word_embedding ** 2, dim=1))
        length_matrix = torch.matmul(length.view(-1, 1), length.view(-1, 1).t())

        semantic_similarity = (1.0 + ip / length_matrix) * 0.5
        ham_similarity = self.calc_ham_similarity(hashCenter_independent, hashCenter_independent)
        KL_divergence = torch.sum(
            semantic_similarity.float() * torch.log(semantic_similarity.float() / ham_similarity.float()))
        return KL_divergence
        pass

    def useGPU(self, x):
        if self.option.use_gpu and torch.cuda.is_available():
            return x.cuda()
        else:
            return x


class CenterLoss(nn.Module):
    def __init__(self, option, state, R=2.0, gamma=10.0, beta=5.0):
        super(CenterLoss, self).__init__()
        self.option = option
        self.centerLoss_type = option.centerLoss_type
        self.state = state
        self.Radius = self.useGPU(torch.tensor(R)).float()
        self.smooth_beta = self.useGPU(torch.tensor(beta)).float()
        self.gamma = self.useGPU(torch.tensor(gamma)).float()
        if self.centerLoss_type == 'BCELoss':
            self.BCELoss = self.useGPU(nn.BCELoss())

    def smoothMaximum(self, x, y, beta):
        if torch.cuda.is_available() and self.state['use_gpu']:
            e_beta_x = torch.exp(beta * x.to(torch.float64)).cuda()
            e_beta_y = torch.exp(beta * y.to(torch.float64)).cuda()
        else:
            e_beta_x = torch.exp(beta * x.to(torch.float64))
            e_beta_y = torch.exp(beta * y.to(torch.float64))
        return (((torch.mul(x.to(torch.float64), e_beta_x)) + (torch.mul(y.to(torch.float64), e_beta_y))) / (
                e_beta_x + e_beta_y)).to(torch.float32)

    def calc_ham_dist(self, outputs1, outputs2):
        ip = (outputs1 * outputs2).sum(dim=1)
        mod = outputs1.pow(2).sum(dim=1) * (outputs2.pow(2).sum(dim=1))
        cos = ip / mod.sqrt()
        hash_bit = outputs1.shape[1]
        dist_ham = hash_bit / 2.0 * (1.0 - cos)
        return dist_ham

    def calc_ham_dist_by_E(self, outputs1, outputs2):
        hash_bit = outputs1.shape[1]
        ip = (outputs1 - outputs2) ** 2
        ip = ip.sum(dim=1)
        return ip / 4.0

    def cal_BCE_dist(self, hashCode, hashCenters):
        Logger.info(" hash Center is {}".format(hashCenters))
        dist = torch.sum(hashCenters * torch.log(hashCode) + (1. - hashCenters) * torch.log(1. - hashCode), dim=1)
        return dist

    def useGPU(self, x):
        if self.option.use_gpu and torch.cuda.is_available():
            return x.cuda()
        else:
            return x

    def forward(self, hashCode, hashCenters, labels, centerWeight):
        if self.centerLoss_type == 'BCELoss':
            for (i, label) in enumerate(labels):
                one_labels = (label == 1).nonzero()  
                one_labels = one_labels.squeeze(1)
                hashCenters = self.state['hash_center_pre'][one_labels]

                random_center = torch.randint_like(hashCenters[0], 2)
                random_center = random_center.repeat(hashCenters.shape[0], 1)
                hashCenters[hashCenters < 0] = -1
                hashCenters[hashCenters > 0] = 1
                random_center[random_center == 0] = -1  
                hashCenters[hashCenters == 0] = random_center[hashCenters == 0]  
                bce_dist = self.BCELoss(hashCode[i].repeat(hashCenters.shape[0], 1), hashCenters.detach())
                Logger.info("bce_dist {} ".format(bce_dist))
                loss = bce_dist * centerWeight[one_labels]
                Logger.info("bce loss {}".format(loss))

                loss = -torch.log(1. / (1. + torch.exp(self.option.beta_sigmoid * loss)))
                return loss.mean()
        if self.centerLoss_type == 'CELoss':
            q = 0.5 * (hashCode + 1)
            p = 0.5 * (hashCenters + 1)

            loss = -torch.sum(p * torch.log(q) + (1 - p) * torch.log(1 - q), 1)
            loss = loss.mean()

            return loss

        if self.centerLoss_type == 'CauchyLoss':
            dist_ham = self.calc_ham_dist(hashCode, hashCenters)
            loss = -torch.log(
                self.gamma / (self.gamma + dist_ham)
            )
            return loss.mean()

        if self.centerLoss_type == 'MarginLoss':
            dist_ham = self.calc_ham_dist(hashCode, hashCenters)
            loss = -torch.log(
                self.gamma / (
                        self.gamma + self.smoothMaximum(torch.tensor(0.), dist_ham - self.Radius, self.smooth_beta))
            )
            return loss.mean()


