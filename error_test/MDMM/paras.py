VOCAB_SIZE = 10000

# 测试数据参数
DATA_SIZE = 6949708
TEST_DATA_SIZE = int(6949708)

# 文件路径

DATA0_PATH = '../../model_data/error_origins.'+str(DATA_SIZE)
DATA1_PATH = '../../model_data/data1.'+str(DATA_SIZE)
DATA2_PATH = '../../model_data/data2.'+str(DATA_SIZE)
DATA3_PATH = '../../model_data/nears.'+str(DATA_SIZE)
TARGET_PATH = '../../model_data/target.'+str(DATA_SIZE)
VOCAB_PATH = '../../model_data/vocab.10000'

COUNT_SAVE_PATH = '../count.dict'
TEST_RESULT_PATH = '../results/test_results.txt'

STEP_PRINT = 1000 # 输出步频

TP = FP = TN = FN = P = N = 0
TPR = TPW =0