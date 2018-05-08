import math

VOCAB_SIZE = 10000 #词典规模
LEARNING_RATE = 0.001 #学习率
LEARNING_RATE_DECAY_FACTOR =  1 #控制学习率下降的参数
KEEP_PROB = 0.8 #节点不Dropout的概率
HIDDEN_SIZE = 128 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 1 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS= 1 #下文lstm的深度
VALID_BATCH_SIZE = TEST_BATCH_SIZE = 128 #测试数据batch的大小
TRAIN_BATCH_SIZE = 128 #训练数据batch的大小

DATA_SIZE = 6949708
TEST_DATA_SIZE = DATA_SIZE
TEST_STEP_SIZE=math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)

PROOFREAD_BIAS = 0.0001
STEP_PRINT = 1000
TP = FP = TN = FN = P = N = 0
TPW = TPR = 0

#文件路径
DATA0_PATH = '../../model_data/error_origins.6949708'
DATA1_PATH = '../../model_data/data1.6949708'
DATA2_PATH = '../../model_data/data2.6949708'
DATA3_PATH = '../../model_data/nears.6949708'
TARGET_PATH = '../../model_data/target.6949708'
VOCAB_PATH = '../../model_data/vocab.10000'

CKPT_PATH = '../ckpt/'
MODEL_NAME = 'model.ckpt'
COST_PATH = '../logs/cost&lr_logs'
RESULT_DIR='../results'
RESULT_PATH = RESULT_DIR+'/results.txt'
TEST_RESULT_PATH = RESULT_DIR+'/test_results.txt'
