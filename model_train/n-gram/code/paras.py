VOCAB_SIZE = 10000

# 训练数据参数
DATA_SIZE = int(24378315 * 0.4)-1
TRAIN_DATA_SIZE = int(DATA_SIZE)

# 文件路径
DATA1_PATH = '../../model_data/data1.24378315'
DATA2_PATH = '../../model_data/data2.24378315'
DATA3_PATH = '../../model_data/nears.24378315'
TARGET_PATH = '../../model_data/target.24378315'
VOCAB_PATH = '../../model_data/vocab.10000'

COUNT_SAVE_PATH = '../count.dict'
TEST_RESULT_PATH = '../results/test_results.txt'

STEP_PRINT = 1000 # 输出步频

TP = FP = TN = FN = P = N = 0
TPR = TPW =0
