#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-14
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")
train_data_file = "./zhengqi_train.txt"
test_data_file = "./zhengqi_test.txt"
# 读取数据
X_train = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
y_train = X_train.pop("target")
X_test = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')
# 删除异常值
X_train = X_train[X_train['V9'] > -7.5]

# 归一化
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


distances=[]
for i in range(X_train.shape[1]):
    distances.append(wasserstein_distance(X_train[:,i], X_test[:,i]))

# 决定删除多少个变量 【0,10】

print(distances)
