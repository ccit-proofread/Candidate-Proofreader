# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
from model import *
from paras import *
from run_epoch import *

#定义主函数并执行
def main():
    row_num = DATA_SIZE
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index] = int(ele)
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index] = int(ele)
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
        data3 = [one.split() for one in rows]
        for one in data3:
            for index, ele in enumerate(one):
                one[index] = int(ele)
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
        target = []
        for one in rows:
            target.append(int(one))
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        char_set = f.read().split('\n')
    
    train_data=(data1[0:TRAIN_DATA_SIZE],data2[0:TRAIN_DATA_SIZE],data3[0:TRAIN_DATA_SIZE],target[0:TRAIN_DATA_SIZE])
    valid_data=(data1[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                data2[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                data3[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                target[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        train_model = Proofreading_Model(True, TRAIN_BATCH_SIZE)

    with tf.variable_scope("Proofreading_model", reuse=True, initializer=initializer):
        eval_model = Proofreading_Model(False, VALID_BATCH_SIZE)

    saver = tf.train.Saver(max_to_keep=0)

    cdir = RESULT_DIR
    if(not os.path.exists(cdir)):
        os.mkdir(cdir)

    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        # 训练模型。
        i = 0
        if ckpt and ckpt.model_checkpoint_path:
            # 读取模型
            print("loading model...")
            saver.restore(session, ckpt.model_checkpoint_path)
            i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
            file = open(RESULT_PATH, 'a')
            train_model.global_epoch=eval_model.global_epoch=i
            train_model.global_step=i*TRAIN_STEP_SIZE
        else:
            print("new training...")
            tf.global_variables_initializer().run()
            file = open(RESULT_PATH, 'w')

        # 要使用tensorboard，首先定义summary节点，不定义会出错
        summary_writer = tf.summary.FileWriter(COST_PATH, session.graph)

        while i < NUM_EPOCH:
            print("In iteration: %d " % i)
            file.write("In iteration: %d\n" % i)
            
            print("In training:")
            file.write("In training:\n")
            run_epoch(session, train_model, train_data, train_model.train_op, True,
                      TRAIN_BATCH_SIZE, TRAIN_STEP_SIZE, char_set, file, summary_writer)
            
            #保存模型
            print("saving model...")
            saver.save(session, CKPT_PATH+MODEL_NAME, global_step = i)
            
            #验证集
            print("In evaluating:")
            file.write("In evaluating:\n")
            run_epoch(session, eval_model, valid_data, tf.no_op(), False,
                      VALID_BATCH_SIZE, VALID_STEP_SIZE, char_set, file, False)
            i += 1
            train_model.global_epoch += 1                   

        file.close()

if __name__ == "__main__":
    main()
