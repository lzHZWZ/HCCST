import os, sys, math

import torch.nn as  nn
import torchvision
import torch
from torch.nn import Parameter
from config_utils import adj2tensor, get_dytxt


class CenterModel(nn.Module):
	def __init__(self, option):
		super(CenterModel, self).__init__()
		self.option = option
		self.class_num_dict = {'voc2012': 20, 'coco': 80, 'nuswide': 21, 'imagenet': 100, 'cifar10': 10}
		self.hash_bit = option.hash_bit
		self.last_layer = nn.Tanh()
		if self.option.center_update:
			self.to_center = nn.Sequential(nn.Linear(option.w2v_dim, 256), nn.ReLU(), nn.Linear(256, self.hash_bit),
										   self.last_layer)

	def forward(self, word_embeddings):
		if self.option.center_update:
			hash_centers = self.to_center(word_embeddings.float())
		else:
			file_path = '../data/' + self.option.data_name + '/' + str(
				self.hash_bit) + '_' + self.option.data_name + '_' + str(
				self.class_num_dict[self.option.data_name]) + '_class.pkl'
			if os.path.exists(file_path):
				center_file = open(file_path, 'rb')
				hash_centers = torch.load(center_file)
			elif os.path.exists(self.option.centers_path):
				center_file = open(self.option.ceters_path, 'rb')
				hash_centers = torch.load(center_file)
		return hash_centers.cuda() if self.option.use_gpu and torch.cuda.is_available() else hash_centers, \
			   word_embeddings

	def getConfig_params(self):
		if self.option.center_update:
			return [
				{'params': self.to_center.parameters(), 'lr': self.option.lr_center},
			]
		else:
			return []


class HashModel(nn.Module):
	def __init__(self, option):
		super(HashModel, self).__init__()
		self.option = option
		self.hash_bit = option.hash_bit
		self.base_model = getattr(torchvision.models, option.model_type)(pretrained=True)
		self.conv1 = self.base_model.conv1
		self.bn1 = self.base_model.bn1
		self.relu = self.base_model.relu
		self.maxpool = self.base_model.maxpool
		self.layer1 = self.base_model.layer1
		self.layer2 = self.base_model.layer2
		self.layer3 = self.base_model.layer3
		self.layer4 = self.base_model.layer4
		self.avgpool = self.base_model.avgpool
		self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
											self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
		self.fc1 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
		self.activation1 = nn.ReLU()
		self.fc2 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
		self.activation2 = nn.ReLU()
		self.fc3 = nn.Linear(self.base_model.fc.in_features, self.hash_bit)
		self.last_layer = nn.Tanh()
		self.dropout = nn.Dropout(0.5)
		self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3,
										self.last_layer)

	def forward(self, images):
		features = self.feature_layers(images)
		features = features.view(features.size(0), -1)
		hash_codes = self.hash_layer(features)
		return hash_codes

	def getConfig_params(self):
		return [
			{'params': self.feature_layers.parameters(), 'lr': self.option.lr * self.option.multi_lr},
			{'params': self.hash_layer.parameters(), 'lr': self.option.lr},
		]



class Swin4Hash(nn.Module):
	def __init__(self, option, model, feature_dim=768):
		super(Swin4Hash, self).__init__()
		self.image_normalization_mean = [0.485, 0.456, 0.406] 
		self.image_normalization_std = [0.229, 0.224, 0.225] 
		self.opt = option
		self.features = model
		self.hash_bit = option.hash_bit

		self.fc1_in_features, self.fc1_out_features = 768, 512  
		self.fc2_out_features = 256

		self.fc1 = nn.Linear(self.fc1_in_features, self.fc1_out_features)
		self.activation1 = nn.ReLU()
		self.fc2 = nn.Linear(self.fc1_out_features, self.fc2_out_features)
		self.activation2 = nn.ReLU()
		self.fc3 = nn.Linear(self.fc2_out_features, self.hash_bit)
		self.last_layer = nn.Tanh()
		self.dropout = nn.Dropout(0.5)
		self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3,
										self.last_layer)

	def forward(self, feature, ):
		feature = self.features.forward_features(feature)
		feature = feature.view(feature.size(0), -1)
		x = self.hash_layer(feature)

		return x

	def getConfig_params(self):
		return [
			{'params': self.features.parameters(), 'lr': self.opt.lr * self.opt.multi_lr},
			{'params': self.hash_layer.parameters(), 'lr': self.opt.lr},
		]
