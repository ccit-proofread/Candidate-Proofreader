#coding=utf-8
import time
from paras import *
from train import *
alpha = 0.9
beta = 0.2

ngram = BackoffModel('./origin_train_data/', './ngram_data/')

def wordchangeprob(orig, updated, confusion_set):
    prob = 0.0
    if len(orig)==len(updated):
        for i in range(0, len(orig)-1):
            #compute dynamic alpha
            if i == 0:
                alpha_d = alpha + ( alpha * beta * ngram.unigram_mle(orig[i]) - 0.5)
            else:
                alpha_d = alpha + ( alpha * beta * ngram.bigram_back_off_model(orig[i], orig[i+1])-0.5 )

            if alpha_d > 1:
                alpha_d = 1
            if orig[i] == updated[i]:
                prob = prob + math.log10(alpha_d)
            elif alpha_d != 1:
                noofcandidates = len(confusion_set) - 1   #要减去本身word
                prob = prob + math.log10((1 - alpha_d) / noofcandidates)
        return prob
    else:
        print("Original and updated sentences differ in length")
        return prob

#get probability of sentence in log base 10
def getsentenceprob(sentence):
    prob = 0.0
    for i in range(0, len(sentence)-2):
        if i == 0:
            prob = prob + math.log10(ngram.unigram_mle(sentence[i]))
        elif i == 1:
            prob = prob + math.log10(ngram.bigram_back_off_model(sentence[i], sentence[i+1]))
        else:
            prob = prob + math.log10( ngram.trigram_back_off_model(sentence[i], sentence[i+1], sentence[i+2]) )
    return prob

def getbestfitsentence(sentence, confusion_set, i):
    maxprob = wordchangeprob(sentence, sentence, []) + getsentenceprob(sentence)
    best_lst = sentence[:]
    for j in confusion_set:
        tmp_lst = sentence[:]
        tmp_lst[i] = j
        tmp_prob = wordchangeprob(sentence, tmp_lst, confusion_set) + getsentenceprob(tmp_lst)
        if tmp_prob > maxprob:
            maxprob = tmp_prob
            best_lst = tmp_lst[:]
    return best_lst[i]

def  main():
    start_reading_from_files = time.time()
    row_num = TEST_DATA_SIZE
    with open(DATA0_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
            if cnt%100000 == 0: print("data0: %d round" % cnt)
        test_data0 = [one.split() for one in rows]

    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
            if cnt % 100000 == 0: print("data1: %d round" % cnt)
        test_data1 = [one.split() for one in rows]

    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
            if cnt % 100000 == 0: print("data2: %d round" % cnt)
        test_data2 = [one.split() for one in rows]

    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                rows.append(line)
            else:
                break
            cnt += 1
            if cnt % 100000 == 0: print("data3: %d round" % cnt)
        test_data3 = [one.split() for one in rows]

    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        test_target = []
        cnt = 0
        for line in f:
            if (cnt < row_num):
                test_target.append(line)
            else:
                break
            cnt += 1
            if cnt % 100000 == 0: print("target: %d round" % cnt)

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        char_set = f.read().split('\n')
    end_reading_from_files = time.time()
    print("===> time for reading from files: ~'%s's" % round(end_reading_from_files - start_reading_from_files))
    cnt = 0
    start = time.time()
    for i in range(TEST_DATA_SIZE):
        data1 = test_data1[i]
        len1 = len(data1)
        data0 = test_data0[i]
        data2 = test_data2[i]
        len2 = len(data2)
        rev_data2 = data2[::-1]
        data2 = rev_data2[0:len2 - 1]
        len2 = len(data2)
        sentence = data1[1:len1] + data0 + data2[0:len2]
        original_word = int(test_data0[i][0])
        output_word = int(getbestfitsentence(sentence, test_data3[i], len1-1))
        target_word = int(test_target[i])
        # 统计评估
        statistics_evaluation(original_word, output_word, target_word)
        cnt += 1
        end = time.time()
        if cnt%1000 == 0 :
            print("testing round: %d   speed is %.1f step/s" % (cnt, (cnt/(end-start)) ) )
            print_evaluation(TEST_RESULT_PATH)

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

def print_evaluation(filename):
    global TP, FP, TN, FN, P, N, TPW, TPR
    P = TP + FN
    N = TN + FP
    file = open(filename, 'a', encoding= 'utf-8')
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
    Correct_Recall = TPR / P
    Correct_Precison = TPR / (TP+FP)
    Correct_F1_Score = 2*Correct_Precison*Correct_Recall/(Correct_Precison+Correct_Recall)
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
    print("Correct_Recall : %.5f " % Correct_Recall)
    file.write("Correct_Recall : %.5f \n" % Correct_Recall)
    print("Correct_Precison : %.5f " % Correct_Precison)
    file.write("Correct_Precison : %.5f \n" % Correct_Precison)
    print("Correct_F1_Score : %.5f " % Correct_F1_Score)
    file.write("Correct_F1_Score : %.5f \n" % Correct_F1_Score)
    print("Delta : %.5f " % Delta)
    file.write("Delta : %.5f \n\n" % Delta)
    file.close()

if __name__ == '__main__':
    main()





