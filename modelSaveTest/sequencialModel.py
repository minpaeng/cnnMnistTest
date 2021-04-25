import tensorflow as tf
import numpy as np
import keras

#train_data 생성
x_train = np.array([[1], [2]])
y_train = (x_train * 2) + 1

print(x_train)
print(y_train)

print(x_train.shape, x_train.ndim)

#test_data 생성
x_test = np.array([[5], [6]])
y_test = (x_test * 2) + 1

#sequencial model 생성하기
model = keras.models.Sequential()
print(type(model))

#모델의 구조 정의(인풋 아웃풋이 각각 한개인 노드 추가)
model.add(keras.layers.Dense(1, input_dim = 1))

#모델 형태 확인
model.summary()
