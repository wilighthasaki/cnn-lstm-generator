import cv2
import os
import pickle
import numpy as np


class Preproccessor(object):
    '''
    This class is used to preprocess the raw data.
    '''
    def __init__(self, img_path, text_path, model_path):
        '''
        read the imgs and texts in img_path and text text_path
        '''
        self.imgs = []
        self.texts = []
        img_files = os.listdir(img_path)
        raw_texts = []

        for img_file in img_files:
            text_file = img_file.split('.')[0] + '.txt'
            img = cv2.imread(os.path.join(img_path, img_file), 0)
            # resize to same shape
            img = self.process_img(img)
            self.imgs.append(img)
            with open(os.path.join(text_path, text_file), 'r', encoding='utf8') as text_in:
                char_list = []
                text = text_in.readline().rstrip()
                for ch in text:
                    if ch != ' ':
                        char_list.append(ch)
            raw_texts.append(char_list)

        # padding
        padded_texts = self.padding(raw_texts)

        # word2id
        self.texts, self.word2id = self.word2id_transform(padded_texts, model_path)

        # transform imgs to np.array
        self.imgs = np.array(self.imgs)

    def process_img(self, img, width=448, height=224):
        img = 1 - img / 255
        padded = np.zeros((height, width))
        if img.shape[0] * 2 > img.shape[1]:
            resize_width = int(224 * img.shape[1] / img.shape[0])
            img = cv2.resize(img, (resize_width, height))
            padded[:, int(224 - resize_width / 2): int(224 - resize_width / 2) + resize_width] = img
        else:
            resize_height = int(448 * img.shape[0] / img.shape[1])
            img = cv2.resize(img, (width, resize_height))
            padded[int(112 - resize_height / 2): int(112 - resize_height / 2) + resize_height, :] = img

        img = np.reshape(padded, (height, width, 1))
        return img

    def word2id_transform(self, raw_texts, model_path):
        word2id_path = os.path.join(model_path, 'word2id.pkl')
        char_count = []
        if not os.path.exists(word2id_path):
            # if there is no model in the model path
            # firstly count the char num and create the map
            for i in raw_texts:
                for j in i:
                    if j not in char_count:
                        char_count.append(j)
            word2id = {ch: i for i, ch in enumerate(char_count)}
            with open(word2id_path, 'wb') as word2id_out:
                pickle.dump(word2id, word2id_out)
        else:
            with open(word2id_path, 'rb') as word2id_in:
                word2id = pickle.load(word2id_in)

        return np.array([[word2id[j] for j in i] for i in raw_texts]), word2id

    def padding(self, raw_text):
        max_len = 0
        new_text = []
        for i in raw_text:
            if len(i) > max_len:
                max_len = len(i)
        for i in raw_text:
            new_text.append(i + ['<EOS>' for _ in range(max_len - len(i) + 1)])
        return new_text



if __name__ == '__main__':
    pre = Preproccessor('../data/mkpic/imgs', '../data/mkpic/texts', '../outputs')
    print(pre.texts.shape)

