import math, os, sys
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch.nn.functional as F

def adj2tensor(adj_file):
	import pickle
	result = pickle.load(open(adj_file, 'rb'))
	_adj = result['adj']
	_nums = result['nums']
	return _adj


def get_dytxt(tfile):
	data = []
	try:
		with open(str(tfile), "r") as f:  # 打开文件
			content = f.readlines()  # 读取文件
			for item in content:
				data.append(float(item))
			print("the data list in the .txt is:", data)
	except Exception as e:
		print("ERROR: ", e)
		return []
	return data