from __future__ import print_function
from itertools import count

import numpy as np
import pandas as pd
import torch
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.autograd import Variable

def get_batch(x, onehot_y, batch_size=5):
	row_indexs = np.random.randint(0, len(x), batch_size)
	return Variable(torch.Tensor([x[row] for row in row_indexs])), Variable(torch.Tensor([onehot_y[row] for row in row_indexs]))


def preprocess_data(path, label_name=None):
	data = pd.read_csv(path)
	# some of the age is missing
	feature_list = ['Pclass','Sex', 'SibSp', 'Parch', 'Fare']
	x_data = data.filter(feature_list, axis = 1 )

	x = x_data.values
	sex_index = feature_list.index('Sex')
	for v in x:
		if v[sex_index] == 'male':
			v[sex_index] = 1
		elif v[sex_index] == 'female':
			v[sex_index] = 2

	onehot_y = []

	if label_name:
		y_data = data.filter([label_name], axis = 1)
		y = y_data.values
		for y_val in y:
			if y_val == 1:
				onehot_y.append([0, 1])
			elif y_val == 0:
				onehot_y.append([1, 0])
			else:
				raise "not expected y value"
	return x, onehot_y

x, y = preprocess_data("data/train.csv", "Survived")

FEATURE_COUNT = len(x[0])
CATEGORY_NUM = 2
LEARNING_RATE = 0.1
print('count of features used for training: ' + str(FEATURE_COUNT))

w = Variable(torch.randn(FEATURE_COUNT, CATEGORY_NUM))
b = Variable(torch.randn(CATEGORY_NUM))

fc = torch.nn.Linear(FEATURE_COUNT, CATEGORY_NUM, bias=True)

for batch_idx in count(1):
	batch_x , batch_y = get_batch(x, y, 5)
	
	fc.zero_grad()
		
	output = F.binary_cross_entropy_with_logits(fc(batch_x), batch_y)
	loss = output.data[0]
	output.backward()

	for param in fc.parameters():
		param.data.add_(-1 * LEARNING_RATE * param.grad.data)

	print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
	if loss < 1e-3:
		break


print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))

X, Y = preprocess_data('data/train.csv', "Survived")
success_count=0
index = 0
while index < len(X):
        s_x = X[index]
        print(s_x)
        s_y = Y[index]
        v_x = Variable(torch.Tensor(s_x))
	y_out = fc(v_x)
	val, indices = torch.max(y_out, 0)
	val2, indices2 = torch.max(Variable(torch.from_numpy(s_y)), 0)
	if indices[0] == indices2[0]:
		success_count = success_count + 1
        index = index + 1
print('success rate on training set: {}/{}'.format(success_count, len(X)))
