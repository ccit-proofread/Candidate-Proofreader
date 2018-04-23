# -- coding: utf-8 --
# =====================================================================
import tensorflow as tf
import numpy as np
from paras import *

def statistics_evaluation(classes,target_index,x0, prob_logits,output_classes, original_classes):
    global TP, FP, TN, FN, P, N, TPW, TPR
    for i,output_word in enumerate(classes):
        original_word = x0[i]
        target_word = target_index[i]
        
        if(prob_logits[i][original_classes[i]] > PROOFREAD_BIAS * prob_logits[i][output_classes[i]]):
            output_word = original_word
        '''
        if(prob_logits[i][output_classes[i]] < PROOFREAD_BIAS):
              output_word = original_word
        '''
        # print("output:%.3lf, original:%.3lf" % (prob_logits[i][output_classes[i]],prob_logits[i][original_classes[i]]))
 
        if (output_word != original_word):  # 修改的文本
            if (original_word != target_word):#错改对或错改错
                TP = TP + 1
                if (output_word == target_word): #错改对
                    TPR = TPR +1
                if (output_word != target_word): #错改错
                    TPW = TPW +1
            elif (original_word == target_word) and (output_word != target_word): #对改错
                FP = FP + 1
        else:  # 不修改的文本
            if (original_word == target_word):
                TN = TN + 1
            else:
                FN = FN + 1


def print_evaluation(file):
    global TP, FP, TN, FN, P, N, TPW, TPR
    P = TP + FN
    N = TN + FP
    print("P : %d\t N : %d" % (P,N))
    file.write("P : %d\t N : %d\n" % (P,N))
    print("TP : %d\t FP : %d" % (TP, FP))
    file.write("TP : %d\t FP : %d\n" % (TP, FP))
    print("TN : %d\t FN : %d" % (TN, FN))
    file.write("TN : %d\t FN : %d\n" % (TN, FN))
    print("TPR : %d\t TPW : %d" % (TPR, TPW))
    file.write("TPR : %d\t TPW : %d" % (TPR, TPW))

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

