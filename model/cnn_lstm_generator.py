import tensorflow as tf
import numpy as np
from .get_batches import get_batches

class CNNLSTMGenerator(object):
    '''
    这是用CNN + LSTM + Attention + Decoder组成的图片文字识别模型
    '''

    def _max_pooling(self, tensor, name):
        '''
        最大池化层的实现
        '''
        return tf.nn.max_pool(tensor,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def _conv(self, tensor, filter_num, name):
        '''
        卷积层的实现
        '''
        with tf.variable_scope(name):
            # 卷积层参数
            tensor_shape = tensor.get_shape().as_list()
            conv_weights = tf.get_variable(
                'w',
                shape=[3, 3, tensor_shape[-1], filter_num],
                dtype=tf.float32
            )
            conv_biases = tf.get_variable(
                'b',
                shape=[filter_num]
            )
            # 卷积层
            conv = tf.nn.conv2d(tensor,
                                conv_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            return relu

    def __init__(self, height, width, time_steps, num_classes, word2id, batch_size=32, embedding_dim=300, hidden_dim=300, is_train=True):

        self.hidden_dim = hidden_dim
        self.word2id = word2id
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.embedding_dim = embedding_dim

        # 输入输出
        self.image_input = tf.placeholder(tf.float32, [None, height, width, 1])
        self.decode_seqs = tf.placeholder(tf.int64, [None, time_steps])
        self.mask = tf.placeholder(tf.float32, [None, time_steps])
        # default height = 32, width = 128
        self.batch_size = batch_size

        with tf.variable_scope('embedding_layer'):
            self.embedding_matrix = tf.get_variable('emb_matrix',
                                                    shape=[len(word2id), self.embedding_dim],
                                                    dtype=tf.float32)

            self.decode_seqs_emb = tf.nn.embedding_lookup(self.embedding_matrix,
                                                          self.decode_seqs)

        # 网络结构
        with tf.variable_scope('main_model'):
            self.conv1_1 = self._conv(self.image_input, 64, 'conv1_1')
            self.conv1_2 = self._conv(self.conv1_1, 64, 'conv1_2')
            self.pool1 = self._max_pooling(self.conv1_2, 'pool1')

            self.conv2_1 = self._conv(self.pool1, 128, 'conv2_1')
            self.conv2_2 = self._conv(self.conv2_1, 128, 'conv2_2')
            self.pool2 = self._max_pooling(self.conv2_2, 'pool2')

            self.conv3_1 = self._conv(self.pool2, 256, 'conv3_1')
            self.conv3_2 = self._conv(self.conv3_1, 256, 'conv_3_2')
            self.conv3_3 = self._conv(self.conv3_2, 256, 'conv3_3')
            self.pool3 = self._max_pooling(self.conv3_3, 'pool3')

            self.conv4_1 = self._conv(self.pool3, 512, 'conv4_1')
            self.conv4_2 = self._conv(self.conv4_1, 512, 'conv4_2')
            self.conv4_3 = self._conv(self.conv4_2, 512, 'conv4_3')
            self.pool4 = self._max_pooling(self.conv4_3, 'pool4')

            self.conv5_1 = self._conv(self.pool4, 512, 'conv5_1')
            self.conv5_2 = self._conv(self.conv5_1, 512, 'conv5_2')
            self.conv5_3 = self._conv(self.conv5_2, 512, 'conv5_3')
            self.pool5 = self._max_pooling(self.conv5_3, 'pool5')

            with tf.variable_scope('blstm'):
                # reshape
                # input: [batch_size, height / 4, width / 4, 64]
                # output: [batch_size, width / 4, height / 4 * 64]
                self.inputs = tf.transpose(self.pool5, perm=[0, 2, 1, 3])
                print(self.inputs.get_shape().as_list())
                self.inputs = tf.reshape(self.inputs, [-1, int(width / 32), int(height / 32) * 512])

                # Forward direction cell
                self.lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0)
                # Backward direction cell
                self.lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0)

                # Get lstm cell output
                self.outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell,
                                                                    self.lstm_bw_cell,
                                                                    self.inputs,
                                                                    dtype=tf.float32)
                # self.final_state = tf.concat((self.final_state[0][0], self.final_state[1][0]), 1)
                # self.final_state = tf.concat(self.final_state, 2)
                self.final_state = self.final_state[0]

                self.lstm_outputs = tf.concat((self.outputs[0], self.outputs[1]), axis=2)

            with tf.variable_scope('decoder'):
                self.decode_LSTM_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
                print(self.lstm_outputs.get_shape().as_list())
                # add attention mechanism
                self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(600,
                                                                                self.lstm_outputs,
                                                                                memory_sequence_length=[int(width / 32) for _ in range(self.batch_size)])

                self.decode_LSTM_cell = tf.contrib.seq2seq.AttentionWrapper(self.decode_LSTM_cell,
                                                                            self.attention_mechanism,
                                                                            # initial_cell_state=self.final_state,
                                                                            attention_layer_size=len(self.word2id))

                self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_matrix,
                                                                       start_tokens=tf.tile([word2id['<EOS>']], [self.batch_size]),
                                                                       end_token=word2id['<EOS>']
                                                                       )
                self.decoder_init_state =self.decode_LSTM_cell.zero_state(self.batch_size, tf.float32)
                self.decoder_init_state = self.decoder_init_state.clone(cell_state=self.final_state)

                self.decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decode_LSTM_cell,
                    helper=self.helper,
                    initial_state=self.decoder_init_state
                    # initial_state=self.decode_LSTM_cell.zero_state(self.batch_size, tf.float32)
                )
                self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=self.decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.time_steps
                )
            self.softmax_w = tf.get_variable('softmax_w',
                                             [self.num_classes, self.num_classes],
                                             dtype=tf.float32)
            self.softmax_w = tf.reshape(tf.tile(self.softmax_w, [self.batch_size, 1]),
                                        [self.batch_size, self.num_classes, self.num_classes])
            self.softmax_b = tf.get_variable('softmax_b', [self.num_classes], dtype=tf.float32)
            self.logits = tf.matmul(self.outputs[0], self.softmax_w) + self.softmax_b

            self.logits = tf.pad(self.logits,
                                 [[0, 0], [0, self.time_steps - tf.shape(self.logits)[1]], [0, 0]])
            self.y_op = tf.argmax(self.logits, axis=2)
            # 计算损失函数
            self.cost = tf.contrib.seq2seq.sequence_loss(self.logits, self.decode_seqs,
                                                         self.mask,
                                                         average_across_timesteps=True,
                                                         average_across_batch=True)
            self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.cost)

    def sequences_get_mask(self, sequences, pad_val=0):
        """Return mask for sequences.

        Examples
        ---------
        >>> sentences_ids = [[4, 0, 5, 3, 0, 0],
        ...                  [5, 3, 9, 4, 9, 0]]
        >>> mask = sequences_get_mask(sentences_ids, pad_val=0)
        ... [[1 1 1 1 0 0]
        ...  [1 1 1 1 1 0]]
        """
        mask = np.ones_like(sequences)
        for i, seq in enumerate(sequences):
            for i_w in reversed(range(len(seq))):
                if seq[i_w] == pad_val:
                    mask[i, i_w] = 0
                else:
                    mask[i, i_w + 1] = 1
                    break   # <-- exit the for loop, prepcess next sequence
        return mask

    def train(self, sess, imgs, decode_seqs, num_epochs=10, model_path='outputs/models/model1'):
        saver = tf.train.Saver()
        # saver.restore(sess, model_path)

        for epoch in range(num_epochs):
            loss_list = []
            imgs_batches, decode_seqs_batches = get_batches([imgs, decode_seqs], batch_size=32, shuffle=True)
            for batch_idx in range(len(imgs_batches)):
                x = imgs_batches[batch_idx]
                y = decode_seqs_batches[batch_idx]
                mask = self.sequences_get_mask(y, self.word2id['<EOS>'])

                _, cost, y_op = sess.run(
                    [self.train_op,
                     self.cost,
                     self.y_op],
                    feed_dict={self.image_input: x,
                               self.mask: mask,
                               self.decode_seqs: y}
                )
                loss_list.append(cost)
            print('epoch %d finished. Training loss: %.2f' % (epoch, sum(loss_list)/len(loss_list)))
            if epoch % 10 == 0:
                print([[self.id2word[i] for i in j] for j in y_op])
                saver.save(sess, model_path)

    def predict(self, sess, x_input):
        target = sess.run(
            self.y_op,
            feed_dict={
                self.image_input: [x_input]
            }
        )
        target = target[0]
        words = [self.id2word[i] for i in target]
        return words
