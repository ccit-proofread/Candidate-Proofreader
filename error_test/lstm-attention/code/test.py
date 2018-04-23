# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import math
from model import *
from paras import *
from run_epoch import *

#定义主函数并执行
def main():
    row_num = TEST_DATA_SIZE
    
    with open(DATA0_PATH, 'r', encoding='utf-8') as f:
        #rows = f.read(row_num).strip().split('\n')
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        data0 = [] 
        for one in rows:
            data0.append(int(one))
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        #rows = f.read(row_num).strip().split('\n')
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        #rows = f.read(row_num).strip().split('\n')
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        #rows = f.read(row_num).strip().split('\n')
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        data3 = [one.split() for one in rows]
        for one in data3:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        #rows = f.read(row_num).strip().split('\n')
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        target = [] 
        for one in rows:
            target.append(int(one))
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        global char_set
        char_set = f.read().split('\n')
    
    test_data=(data0,data1,data2,data3,target)
    
    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        test_model = Proofreading_Model(False, TEST_BATCH_SIZE)

    saver = tf.train.Saver()

    cdir = RESULT_DIR
    if(not os.path.exists(cdir)):
        #print(cdir)
        os.mkdir(cdir)

    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        # 训练模型。
        if ckpt and ckpt.model_checkpoint_path:
            # 读取模型
            print("loading model...")
            saver.restore(session, ckpt.model_checkpoint_path)
            i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
            test_model.global_epoch=i
            test_model.global_step=i*TEST_STEP_SIZE
        else:
            print("model doesn't exist!")
            return
        # 测试模型。
        file = open(TEST_RESULT_PATH, 'w')
        print("In testing with model of epoch %d: " % (i-1))
        run_epoch(session, test_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_STEP_SIZE, char_set, file,False,False)
        file.close()

if __name__ == "__main__":
    main()
