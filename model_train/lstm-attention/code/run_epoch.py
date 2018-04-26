# -- coding: utf-8 --
# =====================================================================
import numpy as np
import time
import random
from paras import *

# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的cost值
def run_epoch(session, model, data, train_op, is_training, batch_size, step_size, char_set,
              file, summary_writer):
    """
    :param session: tf.Session() to compute
    :param model: the proof model already defined
    :param data: data for running
    :param train_op: train operation
    :param is_training: is or not training
    :param batch_size: the size of one batch
    :param step_size: the number of step to run the model
    :param char_set: the dictionary
    :param file: file to write
    :param summary_op: the operation to merge the parameters
    :param summary_writer: output graph writer
    :return: none
    """
    # 总costs
    total_costs = 0.0
    # 获取数据
    dataX1, dataX2, dataX3, dataY = data
    max_cnt = len(dataY)  # 数据长度
    if is_training:
        cnt = random.randint(0, max_cnt - batch_size + 1)  # 现在取第cnt个输入
    else:
        cnt = 0
    correct_num = 0  # 正确个数

    # 训练一个epoch。
    start = time.clock()
    for step in range(step_size):
        if (cnt + batch_size > max_cnt):  # 如果此时取的数据超过了结尾，就取结尾的batch_size个数据
            cnt = max_cnt - batch_size
        x1 = dataX1[cnt:cnt + batch_size]  # 取前文
        x1, x1_seqlen = Pad_Zero(x1)  # 补0

        x2 = dataX2[cnt:cnt + batch_size]  # 取后文
        x2, x2_seqlen = Pad_Zero(x2)  # 补0

        x3 = dataX3[cnt:cnt + batch_size]
        x3, _ = Pad_Zero(x3)

        y = dataY[cnt:cnt + batch_size]  # 取结果

        x4, one_hot = is_candidate(x3, y)

        cost, outputs, _, _, _,\
            = session.run([model.cost, model.logits, train_op, model.learning_rate_decay_op,
                           model.ave_cost_op],
                          feed_dict={model.pre_input: x1, model.fol_input: x2,
                                     model.candidate_words_input: x3,
                                     model.is_candidate: x4,
                                     model.pre_input_seq_length: x1_seqlen,
                                     model.fol_input_seq_length: x2_seqlen,
                                     model.targets: y,
                                     model.one_hot_labels: one_hot
                                     })
        if (is_training):
            model.global_step += 1
            cnt = random.randint(0, max_cnt - batch_size + 1)

        else:
            cnt += batch_size
        if (cnt >= max_cnt):
            cnt = 0
        if not file:
            continue
        total_costs += cost  # 求得总costs
        candidate_classes = np.argmax(outputs, axis=1)

        classes = [x3[i][j] for i,j in enumerate(candidate_classes)]
        target_index = np.array(y).ravel()
        correct_num = correct_num + sum(classes == target_index)

        # 写入到文件以及输出到屏幕
        if (((step + 1) % STEP_PRINT == 0) or (step == 0)) and file:
            end = time.clock()
            print("%.1f setp/s" % (STEP_PRINT / (end - start)))
            start = time.clock()
            print("After %d steps, cost : %.3f" % (step+1, total_costs / (step + 1)))
            file.write("After %d steps, cost : %.3f" % (step+1, total_costs / (step + 1)) + '\n')
            print("outputs: " + ' '.join([char_set[t] for t in classes]))
            print("targets: " + ' '.join([char_set[t] for t in target_index]))
            file.write("outputs: " + ' '.join([char_set[t] for t in classes]) + '\n')
            file.write("targets: " + ' '.join([char_set[t] for t in target_index]) + '\n')

    if file:
        print("After this epoch, cost : %.3f" % (total_costs / (step_size)))
        file.write("After this epoch, cost : %.3f" % (total_costs / (step_size)) + '\n')

    # 收集并将cost加入记录
    if (is_training):
        summary_str = session.run(model.merged_summary_op, feed_dict={model.pre_input: x1, model.fol_input: x2,
                                                         model.candidate_words_input: x3,
                                                         model.is_candidate: x4,
                                                         model.pre_input_seq_length: x1_seqlen,
                                                         model.fol_input_seq_length: x2_seqlen,
                                                         model.targets: y,
                                                         model.one_hot_labels: one_hot
                                                         })
        summary_writer.add_summary(summary_str, model.global_epoch)

    if not is_training and file:
        acc = correct_num * 1.0 / len(dataY)  # 求得准确率=正确分类的个数
        print("acc: %.5f\n" % acc)
        file.write("acc: %.5f\n" % acc)

def Pad_Zero(x):
    x_seqlen = []
    row_len = len(x)
    max_len = 0
    for i in range(row_len):
        col_len = len(x[i])
        x_seqlen.append(col_len)
        max_len = max(max_len, col_len)

    for i in range(row_len):
        col_len = x_seqlen[i]
        for j in range(col_len, max_len):
            x[i].append(0)
    return x, x_seqlen


def is_candidate(x, target):
    x_len = len(x)
    y_len = len(x[0])
    is_candi = np.zeros([x_len, y_len], dtype=np.float32)
    one_hot = np.zeros([x_len, y_len], dtype=np.float32)
    for i in range(x_len):
        for j in range(y_len):
            if(x[i][j] > 0):
                is_candi[i][j] = 1.0
            if(x[i][j] == target[i]):
                one_hot[i][j] = 1.0
    return is_candi, one_hot
