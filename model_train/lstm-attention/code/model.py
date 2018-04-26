# -- coding: utf-8 --
# =====================================================================
import tensorflow as tf
from paras import *

class Proofreading_Model(object):
    def __init__(self, is_training, batch_size):
        """
        :param is_training: is or not training, True/False
        :param batch_size: the size of one batch
        """
        # 定义网络参数
        self.learning_rate = tf.Variable(float(LEARNING_RATE), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * LEARNING_RATE_DECAY_FACTOR)
        self.global_step = 0
        self.global_epoch = 0

        # 定义输入层,其维度是batch_size * num_steps
        self.pre_input = tf.placeholder(tf.int32, [batch_size, None])
        self.pre_input_seq_length = tf.placeholder(tf.int32, [batch_size, ])
        self.fol_input = tf.placeholder(tf.int32, [batch_size, None])
        self.fol_input_seq_length = tf.placeholder(tf.int32, [batch_size, ])

        self.candidate_words_input = tf.placeholder(tf.int32, [batch_size, None])
        self.is_candidate = tf.placeholder(tf.float32, [batch_size, None])

        self.one_hot_labels = tf.placeholder(tf.float32, [batch_size, None])

        # 定义预期输出，它的维度和上面维度相同
        self.targets = tf.placeholder(tf.int32, [batch_size, ])
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])  # embedding矩阵
        # pre_context_model
        with tf.variable_scope('Pre'):
            pre_cell = tf.contrib.rnn.BasicLSTMCell(num_units=PRE_CONTEXT_HIDDEN_SIZE, forget_bias=0.0,
                                                    state_is_tuple=True)
            if is_training:
                pre_cell = tf.contrib.rnn.DropoutWrapper(pre_cell, output_keep_prob=KEEP_PROB)
            pre_lstm_cell = tf.contrib.rnn.MultiRNNCell([pre_cell] * PRE_CONTEXT_NUM_LAYERS, state_is_tuple=True)

            pre_input = tf.nn.embedding_lookup(embedding, self.pre_input)  # 将原本单词ID转为单词向量。
            if is_training:
                pre_input = tf.nn.dropout(pre_input, KEEP_PROB)
            self.pre_initial_state = pre_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            pre_outputs, pre_states = tf.nn.dynamic_rnn(pre_lstm_cell, pre_input,
                                                        sequence_length=self.pre_input_seq_length,
                                                        initial_state=self.pre_initial_state, dtype=tf.float32)
            # pre_outputs = pre_outputs[:, -1, :]
            pre_output = pre_states
            self.pre_final_state = pre_states  # 上文LSTM的最终状态

        # fol_context_model
        with tf.variable_scope('Fol'):
            fol_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FOL_CONTEXT_HIDDEN_SIZE, forget_bias=0.0,
                                                    state_is_tuple=True)
            if is_training:
                fol_cell = tf.contrib.rnn.DropoutWrapper(fol_cell, output_keep_prob=KEEP_PROB)
            fol_lstm_cell = tf.contrib.rnn.MultiRNNCell([fol_cell] * FOL_CONTEXT_NUM_LAYERS, state_is_tuple=True)

            fol_input = tf.nn.embedding_lookup(embedding, self.fol_input)  # 将原本单词ID转为单词向量。
            if is_training:
                fol_input = tf.nn.dropout(fol_input, KEEP_PROB)
            self.fol_initial_state = fol_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            fol_outputs, fol_states = tf.nn.dynamic_rnn(fol_lstm_cell, fol_input,
                                                        sequence_length=self.fol_input_seq_length,
                                                        initial_state=self.fol_initial_state,dtype=tf.float32)
            # fol_outputs = fol_outputs[:, -1, :]
            fol_output = fol_states
            self.fol_final_state = fol_states  # 下文lstm的最终状态

        # 简单拼接
        concat_output = tf.concat([pre_output[0][-1], fol_output[0][-1]], axis=-1)

        # LSTM所有输出
        all_outputs = tf.concat([pre_outputs, fol_outputs], axis=1)

        # 1表示有输入，0表示为padding值
        all_input = tf.concat([self.pre_input, self.fol_input], axis=1)
        one_all_input = tf.sign(all_input)

        # attention for all text
        with tf.variable_scope('attention'):
            bilinear_weight = tf.get_variable("bilinear_weight1", [2 * HIDDEN_SIZE, HIDDEN_SIZE])

            # 计算LSTM最后输出与上下文的匹配度
            M = all_outputs * tf.expand_dims(tf.matmul(concat_output, bilinear_weight),
                                                              axis=1)  # M = [batch_size,time_step,hidden_size]

            score = tf.reduce_sum(M, axis=2)  # [batch_size,time_step]
            paddings = tf.ones_like(one_all_input,dtype=tf.float32) * (-2 ** 32 + 1)
            score = tf.where(tf.equal(one_all_input, 0), paddings, score)

            # attention概率(匹配度)
            alpha = tf.nn.softmax(score)

            # attention vector
            attention_output = tf.reduce_sum(all_outputs * tf.expand_dims(alpha, axis=2),axis=1)  # [batch, hidden_size]

        # 拼接
        concat_output = tf.concat([concat_output, attention_output], axis=-1)
        # 双线性attention
        with tf.variable_scope('bilinear'):  # Bilinear Layer (Attention Step)
            candidate_words_input_vector = tf.nn.embedding_lookup(embedding, self.candidate_words_input)
            bilinear_weight = tf.get_variable("bilinear_weight2", [3*HIDDEN_SIZE, HIDDEN_SIZE])

            # 计算候选词与上下文的匹配度
            M = candidate_words_input_vector * tf.expand_dims(tf.matmul(concat_output, bilinear_weight),
                                                              axis=1)  # M = [batch_size,candi_num,hidden_size]
            # attention概率(匹配度)
            alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2))  # [batch_size,candi_num]

        # 非候选词概率置0
        tmp_prob = alpha * self.is_candidate

        # 重算概率
        self.logits = tmp_prob / tf.expand_dims(tf.reduce_sum(tmp_prob, axis=1), axis=1)
        self.logits = tf.clip_by_value(self.logits, 1e-7, 1.0 - 1e-7)

        # 求交叉熵
        loss = -tf.reduce_sum(self.one_hot_labels * tf.log(self.logits), reduction_indices=1)

        # 记录cost
        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean(loss)
            self.ave_cost = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.ave_cost_op = self.ave_cost.assign(tf.divide(
                tf.add(tf.multiply(self.ave_cost, self.global_step), self.cost), self.global_step + 1))
            # global_step从0开始
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('ave_cost', self.ave_cost)
            tf.summary.scalar('learning_rate', self.learning_rate)

        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # self.train_op = optimizer.minimize(self.cost)

        self.merged_summary_op = tf.summary.merge_all()  # 收集节点

