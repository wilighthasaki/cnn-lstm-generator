import numpy as np
import argparse
import tensorflow as tf
from process.preprocess import Preproccessor
from model.cnn_lstm_generator import CNNLSTMGenerator
import cv2
import pickle
import os

word2id_path = './outputs/word2id.pkl'
gpu_config = '/gpu:1'
model_path = './outputs/models/model1'
with open(word2id_path, 'rb') as word2id_in:
    word2id = pickle.load(word2id_in)


def preprocess(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, 6)

    # 二值化
    img = cv2.adaptiveThreshold(img, 255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=21,
                                   C=20)

    width = 128
    height = 32
    img = cv2.resize(img, (width, height))
    img = np.reshape(img, (height, width, 1))
    return img

# load data and preprocess
test_path = './data/test_data/1'
test_files = os.listdir(test_path)
imgs = []
for test_file in test_files:
    file_path = os.path.join(test_path, test_file)
    img = cv2.imread(file_path, 0)
    img = preprocess(img)
    imgs.append(img)

config = tf.ConfigProto(allow_soft_placement=True)
words_list = []
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config) as sess:
        with tf.device(gpu_config):
            # 指定网络参数初始化方式
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope('model', initializer=initializer, reuse=False):
#                 saver = tf.train.Saver()
                model = CNNLSTMGenerator(32, 128, 20, len(word2id), word2id, batch_size=1)
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
                tf.global_variables_initializer().run()
                for img in imgs:
                    words_list.append(model.predict(sess, img))
