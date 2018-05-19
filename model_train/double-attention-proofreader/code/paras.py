import math

VOCAB_SIZE = 10000 #词典规模
LEARNING_RATE = 0.001 #学习率
LEARNING_RATE_DECAY_FACTOR =  1.0 #控制学习率下降的参数
KEEP_PROB = 0.8 #节点不Dropout的概率
HIDDEN_SIZE = 128 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 1 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS= 1 #下文lstm的深度
VALID_BATCH_SIZE = TEST_BATCH_SIZE = 128 #测试数据batch的大小
TRAIN_BATCH_SIZE = 128 #训练数据batch的大小

DATA_SIZE = int(8708681)
TRAIN_DATA_SIZE = int(6957038)
VALID_DATA_SIZE = int(DATA_SIZE - TRAIN_DATA_SIZE)
TEST_DATA_SIZE = 1000000

TRAIN_STEP_SIZE=math.ceil(TRAIN_DATA_SIZE / TRAIN_BATCH_SIZE)
VALID_STEP_SIZE=math.ceil(VALID_DATA_SIZE / VALID_BATCH_SIZE)
TEST_STEP_SIZE = math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)


STEP_PRINT = 2000
NUM_EPOCH = 15 # 迭代次数

#文件路径
DATA1_PATH = '../../model_data/data1.8708681'
DATA2_PATH = '../../model_data/data2.8708681'
DATA3_PATH = '../../model_data/nears.8708681'
TARGET_PATH = '../../model_data/target.8708681'
VOCAB_PATH = '../../model_data/vocab.10000'

CKPT_PATH = '../ckpt/'
MODEL_NAME = 'model.ckpt'
COST_PATH = '../logs/cost&lr_logs'
RESULT_DIR='../results'
RESULT_PATH = RESULT_DIR+'/results.txt'
TEST_RESULT_PATH = RESULT_DIR+'/test_results.txt'
