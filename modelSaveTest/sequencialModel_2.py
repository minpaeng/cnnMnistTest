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

#모델의 구조 정의
model.add(keras.layers.Dense(2, input_dim = 1)) #output 2개, input 1개

#second layer : four node
model.add(keras.layers.Dense(4))

#output layer : one node
model.add(keras.layers.Dense(1))

#모델 형태 확인
model.summary()

#손실 함수와 optimizer 설정
model.compile('SGD', 'mse')

#fit로 모델 학습시키기
model.fit(x_train, y_train, epochs= 1000, batch_size=2, verbose= 0)

#모델 평가
model.evaluate(x_test, y_test, batch_size=2)

#모델을 통해 정답 예측하기
y_predict = model.predict(x_test, batch_size=2)
print("y_predict = ")
print(y_predict)
print()
print("Y_test = ")
print(y_test)