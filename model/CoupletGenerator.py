import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from util.get_batches import get_batches


class CoupletGenerator(object):
    def __init__(self, input_dim, num_steps, num_classes, hidden_dim=300, embedding_matrix=None, batch_size=10,
                 encoder_dropout=0.3, word2id=None, is_training=True):
        # 参数初始化
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.encoder_dropout = encoder_dropout
        self.is_training = is_training
        self.word2id = word2id
        self.num_classes = num_classes
        # placeholder
        self.encode_seqs = tf.placeholder(dtype=tf.int32, shape=[None, self.num_steps])
        self.decode_seqs = tf.placeholder(dtype=tf.int32, shape=[None, self.num_steps])
        self.target_seqs = tf.placeholder(dtype=tf.int32, shape=[None, self.num_steps])
        self.mask_seqs = tf.placeholder(dtype=tf.float32, shape=[None, self.num_steps])
        # embedding layer
        with tf.variable_scope('embedding_layer', reuse=False):
            self.embedding_matrix = tf.Variable(embedding_matrix,
                                                dtype=tf.float32)
            self.encode_seqs_emb = tf.nn.embedding_lookup(self.embedding_matrix,
                                                          self.encode_seqs)
            self.decode_seqs_emb = tf.nn.embedding_lookup(self.embedding_matrix,
                                                          self.decode_seqs)

        # encoder
        with tf.variable_scope('encoder'):
            self.LSTM_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            if self.is_training:
                self.LSTM_cell = tf.contrib.rnn.DropoutWrapper(self.LSTM_cell,
                                                               output_keep_prob=1 - self.encoder_dropout)
            self.encoder_outputs, self.final_state = tf.nn.dynamic_rnn(
                self.LSTM_cell,
                self.encode_seqs_emb,
                dtype=tf.float32,
                sequence_length=tl.layers.retrieve_seq_length_op2(self.encode_seqs),
            )
            # self.encoder_outputs = tf.cast(self.encoder_outputs, tf.float32)

        # decoder
        with tf.variable_scope('decoder'):
            self.decode_LSTM_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            # add attention mechanism
            # self.attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            #         512, self.encoder_outputs,
            #         memory_sequence_length=tl.layers.retrieve_seq_length_op2(self.encode_seqs))
            # self.decode_LSTM_cell = tf.contrib.seq2seq.AttentionWrapper(
            #         self.decode_LSTM_cell,
            #         self.attention_mechanism,
            #         initial_cell_state=self.final_state,
            #         attention_layer_size=300)

            # if self.is_training:
            if False:
                self.helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs=self.decode_seqs_emb,
                        sequence_length=tl.layers.retrieve_seq_length_op2(self.decode_seqs))
            else:
                self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_matrix,
                    start_tokens=tf.tile([word2id['<EOS>']], [self.batch_size]),
                    end_token=word2id['<EOS>']
                )

            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decode_LSTM_cell,
                helper=self.helper,
                initial_state=self.final_state
                # initial_state=self.decode_LSTM_cell.zero_state(batch_size, tf.float32)
            )
            self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=50
            )
        self.softmax_w = tf.get_variable('softmax_w',
                                         [self.hidden_dim, self.num_classes],
                                         dtype=tf.float32)
        self.softmax_w = tf.reshape(tf.tile(self.softmax_w, [self.batch_size, 1]),
                                    [self.batch_size, self.hidden_dim, self.num_classes])
        self.softmax_b = tf.get_variable('softmax_b', [self.num_classes], dtype=tf.float32)
        self.logits = tf.matmul(self.outputs[0], self.softmax_w) + self.softmax_b
        self.logits = tf.pad(self.logits,
                             [[0, 0], [0, self.num_steps - tf.shape(self.logits)[1]], [0, 0]])
        self.y_op = tf.argmax(self.logits, axis=2)
        # 计算损失函数
        self.cost = tf.contrib.seq2seq.sequence_loss(self.logits, self.target_seqs,
                                                     self.mask_seqs,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True)
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.cost)

    def train(self, sess, encode_seqs, decode_seqs, target_seqs, num_epoch,
              model_path='models/couplet_generator.model'):
        saver = tf.train.Saver()
        for epoch in range(num_epoch):
            self.is_training = True
            loss_list = []
            encode_seqs_batches, decode_seqs_batches, target_seqs_batches = \
                get_batches([encode_seqs, decode_seqs, target_seqs], batch_size=10, shuffle=True)
            for batch_idx in tqdm(range(len(encode_seqs_batches))):
                x = encode_seqs_batches[batch_idx]
                y = decode_seqs_batches[batch_idx]

                mask = tl.prepro.sequences_get_mask(y)
                _, cost, final_state = sess.run(
                    [self.train_op,
                     self.cost,
                     self.final_state],
                    feed_dict={self.encode_seqs: x,
                               self.target_seqs: target_seqs_batches[batch_idx],
                               self.mask_seqs: mask,
                               self.decode_seqs: y}
                )
                loss_list.append(cost)
            print('epoch %d finished. Training loss: %.2f' % (epoch, sum(loss_list)/len(loss_list)))
            saver.save(sess, model_path)

    def generate(self, sess, encode_seqs, target_seqs, id2word):
        for batch in tl.iterate.minibatches(encode_seqs, target_seqs, batch_size=10, shuffle=False):
            x, y = batch
            y_op = sess.run(
                self.y_op,
                feed_dict={self.encode_seqs: x}
            )
            sent_list = [[str(id2word[i]) for i in sent] for sent in y_op]
            top_couplet_list = [[str(id2word[i]) for i in sent] for sent in x]
            true_list = [[str(id2word[i]) for i in sent] for sent in y]
            for sent_idx in range(len(sent_list)):
                print(''.join(top_couplet_list[sent_idx]).rstrip('<PAD>'))
                print('true:', ''.join(true_list[sent_idx]).rstrip('<PAD>').rstrip('<EOS>'))
                print('predict:', ''.join(sent_list[sent_idx]).rstrip('<EOS>'))
                print()
