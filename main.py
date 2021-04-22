import tensorflow as tf
from model import train
'''
任务：利用前 time_steps 个小时的天气条件和污染程度数据
预测以下两种情况：
1.接下来 predict_time_steps 小时段的污染程度
2.第 time_steps 小时的污染程度

本代码实现的是第二种，若要预测第一种情况，只需要简单的对 series_to_supervised 函数生成的监督数据集的列进行选择取舍
即可得到想要的数据集，然后加以训练即可实现第一种情况，预测结果还不错，mse和mae比第二种情况高很多，但总体效果也还不错
'''
algorithm_params={
    ### 任务要预测的情况
    'task':2,

    ### 有关数据集的一些参数如下：
    # 历史序列长度(时间步)
    'time_steps' : 18,
    # 要预测的序列长度
    'predict_time_steps' : 1,
    # 每一时间步的特征数：可从下面代码中得到的预处理后的数据集得到该特征数，此处初始为 None
    'features_n' : None,
    # 划分训练集
    'train__valid_test_split':0.3,
    # 划分验证集和测试集
    'valid__test_split':0.5,

    ### 有关LSTM模型结构的参数如下：
    # LSTM隐层节点数
    'n_hidden_nodes':128,
    # LSTM隐层层数
    'n_hidden_layers':2,
    # 输出层节点数
    'n_output_nodes':1,

    ### 有关LSTM模型训练的参数如下：
    # 优化器的选择:Adam、Adagrad、SGD
    'optimizer':'Adam',
    # 学习率
    'learning_rate':1e-3,
    # 损失函数
    'loss':'mse',
    # 训练迭代次数
    'epochs':50,
    # 数据集批量大小
    'batch_size':256,

    ### 有关进化计算和注意力层的参数如下：
    # 记录所有已训练过的个体 individual 的 rmse
    'key_to_rmse':{},
    # 目前在验证集上表现最好的注意力层权重
    'best_weight':None,
    # 迭代次数
    'iterations':10,
    # 每一代的种群大小
    'pop_size':20,
    # 每个个体的每个特征的编码长度
    'encode_length':6,
    # 每次迭代后挑选出用于产生后代的个体数目,该参数就等于将种群划分的组的个数。
    'n_selected':4
}
def main():
    # 配置GPU申请内存的方式是按需增长或者限制内存使用大小
    print('is gpu available:', tf.test.is_gpu_available())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    train(algorithm_params)
if __name__=='__main__':
    main()