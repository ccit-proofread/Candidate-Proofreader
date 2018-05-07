# -*- coding: utf-8 -*-
# =====================================================================
import os
import numpy as np
import math
from paras import *

def main():
    row_num = TEST_DATA_SIZE
    with open(DATA0_PATH, 'r', encoding='utf-8') as f:
        test_data0 = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                test_data0.append(line.strip())
            else:
                break
            cnt += 1

    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        test_data1 = [one.split() for one in rows]
        
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        test_data2 = [one.split() for one in rows]
        
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                rows.append(line)
            else:
                break
            cnt += 1
        test_data3 = [one.split() for one in rows]
        
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        test_target = []
        cnt = 0
        for line in f:
            if(cnt<row_num): 
                test_target.append(line.strip())
            else:
                break
            cnt += 1
        
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        char_set = f.read().split('\n')

    with open(COUNT_SAVE_PATH, 'r', encoding='utf-8') as f:
        dict = f.read()
        count = eval(dict)

    # for ele in count.items():
        #print(ele)
    
    print("testing...")
    # 测试
    for i in range(TEST_DATA_SIZE):

        # 计算特征出现次数和
        sum_bi_left = 0
        sum_bi_right = 0
        sum_tri_left = 0
        sum_tri_right = 0
        sum_tri = 0
        for ele in test_data3[i]:
            bi_left_word = test_data1[i][-1] + ' ' + ele
            if bi_left_word in count:
                sum_bi_left += count[bi_left_word]

            bi_right_word = ele + ' ' + test_data1[i][-1]
            if bi_right_word in count:
                sum_bi_right += count[bi_right_word]

            if (len(test_data1[i]) < 2):
                tri_left_word = "none"
            else:
                tri_left_word = test_data1[i][-2] + ' ' + test_data1[i][-1] + ' ' + ele
            if tri_left_word in count:
                sum_tri_left += count[tri_left_word]

            if (len(test_data2[i]) < 2):
                tri_right_word = "none"
            else:
                tri_right_word = ele + ' ' + test_data2[i][-1] + ' ' + test_data2[i][-2]
            if tri_right_word in count:
                sum_tri_right += count[tri_right_word]

            tri_word = test_data1[i][-1] + ' ' + ele + ' ' + test_data2[i][-1]
            if tri_word in count:
                sum_tri += count[tri_word]


        # 计算概率和分数
        p_bi_left = []
        p_bi_right = []
        p_tri_left = []
        p_tri_right = []
        p_tri = []
        score = []
        original_word_score = 0
        for ele in test_data3[i]:
            ele_score = 0
            bi_left_word = test_data1[i][-1] + ' ' + ele
            if bi_left_word not in count:
                prob = 0
            else:
                prob = count[bi_left_word]/sum_bi_left
            ele_score += 0.1*prob
            p_bi_left.append(prob)

            bi_right_word = ele + ' ' + test_data1[i][-1]
            if bi_right_word not in count:
                prob = 0
            else:
                prob = count[bi_right_word]/sum_bi_right
            ele_score += 0.1 * prob
            p_bi_right.append(prob)

            if (len(test_data1[i]) < 2):
                tri_left_word = "none"
            else:
                tri_left_word = test_data1[i][-2] + ' ' + test_data1[i][-1] + ' ' + ele
            if (tri_left_word not in count) or (count[tri_left_word] == 0):
                prob = 0
            else:
                prob = count[tri_left_word] / sum_tri_left
            ele_score += (0.8/3) * prob
            p_tri_left.append(prob)

            if (len(test_data2[i]) < 2):
                tri_right_word = "none"
            else:
                tri_right_word = ele + ' ' + test_data2[i][-1] + ' ' + test_data2[i][-2]
            if (tri_right_word not in count) or (count[tri_right_word] == 0):
                prob = 0
            else:
                prob = count[tri_right_word] / sum_tri_right
            ele_score += (0.8 / 3) * prob
            p_tri_right.append(prob)

            tri_word = test_data1[i][-1] + ' ' + ele + ' ' + test_data2[i][-1]
            if tri_word not in count:
                prob = 0
            else:
                prob = count[tri_word] / sum_tri
            ele_score += (0.8 / 3) * prob
            p_tri.append(prob)

            score.append(ele_score)
            if(ele == test_data0[i]):
                original_word_score = ele_score


        original_word = int(test_data0[i])
        output_word = original_word
        target_word = int(test_target[i])

        max_score = -1 # 最大分数
        max_cset_word = -1 # 最大分数对应的字
        for j,ele in enumerate(score):
            if(max_score < ele):
                max_score = ele
                max_cset_word = test_data3[i][j]

        # print(original_word_score,max_score)
        if(original_word_score < max_score):
            if(original_word_score == 0 and max_score > 0):
                output_word = int(max_cset_word)
            elif(original_word_score > 0 and original_word_score < 0.01*max_score):
                output_word = int(max_cset_word)

        # 输出
        if (((i + 1) % STEP_PRINT == 0) or (i == 0)):
            print("testing %d row" % (i+1))
            print("outputs: " + str(char_set[output_word]))
            print("targets: " + str(char_set[target_word]))
        
        # 统计评估
        statistics_evaluation(original_word, output_word, target_word)

def statistics_evaluation(original_word, output_word, target_word):
    global TP, FP, TN, FN, P, N, TPW, TPR

    if (output_word != original_word):  # 修改的文本
        if (original_word != target_word):  # 错改对或错改错
            TP = TP + 1
            if (output_word == target_word):  # 错改对
                TPR = TPR + 1
            if (output_word != target_word):  # 错改错
                TPW = TPW + 1
        elif (original_word == target_word) and (output_word != target_word):  # 对改错
            FP = FP + 1
    else:  # 不修改的文本
        if (original_word == target_word):
            TN = TN + 1
        else:
            FN = FN + 1

def print_evaluation():
    global TP, FP, TN, FN, P, N, TPW, TPR
    P = TP + FN
    N = TN + FP
    if(not os.path.exists('../results/')):
        os.mkdir('../results/')
    file = open(TEST_RESULT_PATH, 'w')

    print("P : %d\t N : %d" % (P,N))
    file.write("P : %d\t N : %d\n" % (P,N))
    print("TP : %d\t FP : %d" % (TP, FP))
    file.write("TP : %d\t FP : %d\n" % (TP, FP))
    print("TN : %d\t FN : %d" % (TN, FN))
    file.write("TN : %d\t FN : %d\n" % (TN, FN))
    print("TPR : %d\t TPW : %d" % (TPR, TPW))
    file.write("TPR : %d\t TPW : %d\n" % (TPR, TPW))

    Accuracy = (TP+TN)/(P+N)
    Error_Rate = 1-Accuracy
    Recall = TP/P
    Precision = TP/(TP+FP)
    F1_Score = 2*Precision*Recall/(Precision+Recall)
    Correction_Rate = TPR / TP
    Specificity = TN / N
    Delta = (P-(FP+FN+TPW)) / P 
    print("Accuracy : %.5f " % Accuracy)
    file.write("Accuracy : %.5f \n" % Accuracy)
    print("Error_Rate : %.5f " % Error_Rate)
    file.write("Error_Rate : %.5f \n" % Error_Rate)
    print("Recall : %.5f " % Recall)
    file.write("Recall : %.5f \n" % Recall)
    print("Precision : %.5f " % Precision)
    file.write("Precision : %.5f \n" % Precision)
    print("F1_Score : %.5f " % F1_Score)
    file.write("F1_Score : %.5f \n" % F1_Score)
    print("Correction_Rate : %.5f " % Correction_Rate)
    file.write("Correction_Rate : %.5f \n" % Correction_Rate)
    print("Specificity : %.5f " % Specificity)
    file.write("Specificity : %.5f \n" % Specificity)
    print("Delta : %.5f " % Delta)
    file.write("Delta : %.5f \n" % Delta)
    file.close()


if __name__ == "__main__":
    main()
    print_evaluation()




