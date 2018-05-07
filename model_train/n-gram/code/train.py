# -*- coding: utf-8 -*-
#=====================================================================
import os
import numpy as np
import math
from paras import *

def main():
    row_num = TRAIN_DATA_SIZE
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        train_data1 = [one.split() for one in rows]
        
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        train_data2 = [one.split() for one in rows]
        
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        train_data3 = [one.split() for one in rows]
        
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        train_target = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                train_target.append(line.strip())
            else:
                break
            cnt += 1
        
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        char_set = f.read().split('\n')

    # 训练
    if(os.path.exists(COUNT_SAVE_PATH)):
        print("loading count.dict...")
        with open(COUNT_SAVE_PATH, 'r', encoding='utf-8') as f:
            dict = f.read()
            count = eval(dict)
    else:
        count = { "none":0 }
    print("training...")
    pre_bi_right = 0 # 前一个右2元组
    pre_tri = 0 # 前一个3元组
    pre_tri_right = 0  # 前一个右3元组
    pre2_tri_right = 0 # 前两个右3元组
    for i in range(TRAIN_DATA_SIZE):
        if(((i+1) % 10000 == 0) or (i == 0)):
            print("training %d row" % (i+1))
        # 二元组统计
        bi_left_word = train_data1[i][-1] + ' ' + train_target[i]
        # print(bi_left_word)
        if(bi_left_word != pre_bi_right):
            if bi_left_word not in count:
                count[bi_left_word] = 1
            else:
                count[bi_left_word] += 1

        bi_right_word = train_target[i] + ' ' + train_data2[i][-1]
        pre_bi_right = bi_right_word
        if bi_right_word not in count:
            count[bi_right_word] = 1
        else:
            count[bi_right_word] += 1
        
        # 三元组统计
        if(len(train_data1[i]) >= 2):
            tri_left_word = train_data1[i][-2] + ' ' + train_data1[i][-1] + ' ' + train_target[i]
            if((tri_left_word != pre2_tri_right) and (tri_left_word != pre_tri)):
                if tri_left_word not in count:
                    count[tri_left_word] = 1
                else:
                    count[tri_left_word] += 1
            
        
        pre2_tri_right = pre_tri_right
        if(len(train_data2[i]) >= 2):
            tri_right_word = train_target[i] + ' ' + train_data2[i][-1] + ' ' + train_data2[i][-2] 
            if tri_right_word not in count:
                count[tri_right_word] = 1
            else:
                count[tri_right_word] += 1
            pre_tri_right = tri_right_word            
        else:
            pre_tri_right = 0

        tri_word = train_data1[i][-1] + ' ' + train_target[i] + ' ' + train_data2[i][-1]
        pre_tri = tri_word
        if(tri_word != pre_tri_right):
            if tri_word not in count:
                count[tri_word] = 1
            else:
                count[tri_word] += 1
            
    print("saving count dict...")
    with open(COUNT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write(str(count))

if __name__ == "__main__":
    main()
