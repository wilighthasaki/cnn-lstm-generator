import numpy as np
import argparse
import tensorflow as tf
from process.preprocess import Preproccessor
from model.cnn_lstm_generator import CNNLSTMGenerator
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', help='the path of the img train data file', default='./data/mkpic/imgs')
parser.add_argument('-t', '--text_path', help='the path of the text train data file', default='./data/mkpic/texts')
parser.add_argument('-w', '--word2id_path', help='the path of the word2id map file', default='./outputs/')
parser.add_argument('-m', '--model_path', help='the path to save model', default='./outputs/models/model')
parser.add_argument('-g', '--gpu_config', help='the config of gpu, the default is /cpu:0', default='/cpu:0')

args = parser.parse_args()

# 获取输入的特征，输出，和保存模型的路径，在这里我们采用之前预处理过的特征，
# 如果有需要也可以修改成为在这里进行预处理，只要在这里初始化一个预处理类即可

print('正在预处理数据')
pre = Preproccessor(args.img_path, args.text_path, args.word2id_path)
imgs = pre.imgs[:1000]
texts = pre.texts[:1000]
word2id = pre.word2id

model_path = args.model_path
gpu_config = args.gpu_config

# 创建输出路径
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')
if not os.path.exists('./outputs/models/'):
    os.mkdir('./outputs/models/')

# 划分训练集和验证集
data_len = imgs.shape[0]
print(data_len)
shuffle_idx = np.arange(data_len)
np.random.shuffle(shuffle_idx)

train_imgs = imgs[shuffle_idx[:int(data_len * 4 / 5)]]
train_texts = texts[shuffle_idx[:int(data_len * 4 / 5)]]
val_imgs = imgs[shuffle_idx[int(data_len * 4 / 5):]]
val_texts = texts[shuffle_idx[int(data_len * 4 / 5):]]

print('正在初始化模型')
# 这个设置用来使没有GPU的时候可以调用CPU
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        # 指定网络参数初始化方式
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope('model', initializer=initializer, reuse=False):
            model = CNNLSTMGenerator(train_imgs.shape[1], train_imgs.shape[2], train_texts.shape[1], len(word2id), word2id)

            print('开始训练')
            tf.global_variables_initializer().run()
            model.train(sess, train_imgs, train_texts, num_epochs=1000)




