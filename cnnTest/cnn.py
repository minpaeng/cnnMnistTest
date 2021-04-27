import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_val = x_train[50000:60000]
x_train = x_train[0:50000] #50000장으로 제한
y_val = y_train[50000:60000]
y_train = y_train[0:50000] #50000개로 제한

#50000장의 셈플, 28 * 28 사이즈
print("train data has " + str(x_train.shape[0]) + " samples")
print("every train data is " + str(x_train.shape[1])
      + " * " + str(x_train.shape[2]) + " image(MNIST)")

print(x_train.shape)

#데이터셋 reshape(모델 입력으로 들어갈 수 있도록 (x, 28, 28) -> (x, 28, 28, 1)로 조정)
x_train = np.reshape(x_train, (50000,28,28,1))
x_val = np.reshape(x_val, (10000,28,28,1))
x_test = np.reshape(x_test, (10000,28,28,1))

print(x_train.shape)
print(x_test.shape)

#데이터 표준화(?) 표본화?
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255

x_train /= gray_scale
x_val /= gray_scale
x_test /= gray_scale

#one hot 인코딩 전
print(y_train[0:10])

#one hot 인코딩 후
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print(y_train[0:10])

#input과 output 틀 만들기
#x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
x = x_val
#y_ = tf.placeholder(tf.float32, shape=[None, 10])
y = y_val
#가중치, 편향 함수
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution, pooling 함수
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#첫번째 conv 레이어 : 16개의 5x5 사이즈 필터
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
#Relu함수 사용
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #x는 위에서 만들었던 input 틀

#pooling 레이어로 파라미터 수와 오버피팅을 줄임 : 14x14로 사이즈가 변경됨
h_pool1 = max_pool_2x2(h_conv1)

#두번째 conv 레이어 : 32개의 5x5 사이즈 필터
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#두 번째 pooling 레이어: 7x7로 사이즈가 변경됨
h_pool2 = max_pool_2x2(h_conv2)

#
W_fc1 = weight_variable([7 * 7 * 32, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#
W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

#
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))

#
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
# initialize
init = tf.global_variables_initializer()

# train hyperparameters
epoch_cnt = 3
batch_size = 500
iteration = len(x_train) // batch_size

# Start training
with tf.Session() as sess:
    tf.set_random_seed(777)
    # Run the initializer
    sess.run(init)
    for epoch in range(epoch_cnt):
        avg_loss = 0.
        start = 0;
        end = batch_size

        for i in range(iteration):
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: x_train[start: end], y: y_train[start: end]})
                print("step " + str(i) + ": training accuracy: " + str(train_accuracy))
            train_step.run(feed_dict={x: x_train[start: end], y: y_train[start: end]})
            start += batch_size;
            end += batch_size

            # Validate model
        val_accuracy = accuracy.eval(feed_dict={x, y})
        print("validation accuracy: " + str(val_accuracy))

    test_accuracy = accuracy.eval(feed_dict={x, y})
    print("test accuracy: " + str(test_accuracy))