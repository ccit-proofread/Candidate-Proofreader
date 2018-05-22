import math

VOCAB_SIZE = 10000

# 测试数据参数
DATA_SIZE = 6949708
TEST_DATA_SIZE = 6949708
num = 0
#文件路径

DATA0_PATH = './spilt_test_data_0.001/s_error_'+str(num)
DATA1_PATH = './spilt_test_data_0.001/s_data1_'+str(num)
DATA2_PATH = './spilt_test_data_0.001/s_data2_'+str(num)
DATA3_PATH = './spilt_test_data_0.001/s_nears_'+str(num)
TARGET_PATH = './spilt_test_data_0.001/s_target_'+str(num)
VOCAB_PATH = './spilt_test_data_0.001/vocab.10000'
'''
DATA0_PATH = './data/data0.'+str(DATA_SIZE)
DATA1_PATH = './data/data1.'+str(DATA_SIZE)
DATA2_PATH = './data/data2.'+str(DATA_SIZE)
DATA3_PATH = './data/candidate.'+str(DATA_SIZE)
TARGET_PATH = './data/target.'+str(DATA_SIZE)
VOCAB_PATH = './data/vocab.10000'
'''
COUNT_SAVE_PATH = './count.dict'
TEST_RESULT_PATH = './results/test_results.txt'

STEP_PRINT = 1000 # 输出步频

TP = FP = TN = FN = P = N = 0
TPR = TPW = 0