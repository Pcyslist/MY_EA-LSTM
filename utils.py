from datetime import datetime
import numpy as np
from pandas import read_csv, DataFrame,concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 对原始数据进行预处理：
def preprocess_raw_dataset(csvfile_path):
    '''
    对来自 http://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
    的原始数据集下载并重命名为 raw_dataset.csv ,然后进行预处理，产生的数据集 pollution.csv
    更清晰、更易于进一步处理。
    :param csvfile_path: 待处理的csv文件的路径。
    :return: None 直接在运行目录下产生处理后的新数据集 pollution.csv 。
    '''
    # 处理时间的值：将四个字段整合成一个
    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
    # 加载数据
    dataset = read_csv(csvfile_path,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    # 删除第一列‘No’
    dataset.drop('No', axis=1, inplace=True)
    # 为每列指定更清晰的名称
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    # 将合并后的时间信息作为DataFrame的索引。
    dataset.index.name = 'date'
    # 将NA值替换为“0”值
    dataset['pollution'].fillna(0, inplace=True)
    # 删除前二十四个小时，因为该二十四个小时无法被填充
    dataset = dataset[24:]
    # 保存成新的文件
    dataset.to_csv('pollution.csv')
    print('原始数据 raw_dataset.csv 处理完成，生成 pollution.csv 文件以供后续处理 ！')
    return

# 对pollution数据集进行预处理（编码、归一化）
def preprocess_pollution(csvfile_path):
    '''
    对处理原始数据得到的数据集进一步的处理
    :param csvfile_path: 待处理的csv格式的数据集路径
    :return: scaler 归一化的操作算子；scaled 处理完成得到的数据
    '''
    # 加载预处理原始数据后的数据集
    dataset = read_csv(csvfile_path, header=0, index_col=0)
    values = dataset.values
    # 对风向进行整型编码/亦可进行one-hot编码
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # 确保所有的数据是浮点型
    values = values.astype('float32')
    # 对数据进行归一化处理以消除量纲的影响
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print('pollution.csv 数据集处理完成！')
    return scaler,scaled

# 将时间序列数据集转换为监督学习数据集的函数定义
# https://blog.csdn.net/u012735708/article/details/82772388
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 得到LSTM模型可直接使用的数据集
def get_usable_data(algorithm_params):
    '''
    通过该函数能够按照 algorithm_params 参数对原始数据集进行配置处理，得到可供
    LSTM模型直接使用的数据集。
    :param algorithm_params:  算法参数
    :return: train_X,valid_X,test_X,train_y,valid_y,test_y
    '''
    # 数据预处理
    preprocess_raw_dataset('dataset/raw_dataset.csv')
    scaler, scaled = preprocess_pollution('pollution.csv')
    algorithm_params['features_n']=scaled.shape[1]  # 得到每一时间步的特征数

    # 将时间序列数据集转换为监督学习数据集
    reframed = series_to_supervised(scaled, algorithm_params['time_steps'],
                                    algorithm_params['predict_time_steps'])
    # 删除不需要预测的标签列
    index_col = algorithm_params['time_steps'] * algorithm_params['features_n']
    if algorithm_params['task'] == 1:
        reframed.drop( reframed.columns[ [ index_col + i for i in range( 1 , algorithm_params['features_n'] ) ] ], axis=1, inplace=True )
    if algorithm_params['task'] == 2:
        reframed.drop( reframed.columns[ [ index_col + i for i in range( algorithm_params['features_n'] ) ] ], axis=1, inplace=True )

    # 将监督学习数据集拆分为输入输出
    values=reframed.values
    if algorithm_params['task'] == 1:
        X, y = values[:, : index_col], values[:, -1]
    if algorithm_params['task'] == 2:
        X , y= values[ : , : index_col ] , values[ : , -algorithm_params['features_n'] ]

    # 拆分为训练集、验证集、测试集 划分比例为（7：1.5：1.5）
    train_X, X, train_y, y = train_test_split(X, y, test_size=algorithm_params['train__valid_test_split'])
    valid_X, test_X, valid_y, test_y = train_test_split(X, y, test_size=algorithm_params['valid__test_split'])

    # 将数据集变为 LSTM 输入需要的形状 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], algorithm_params['time_steps'], algorithm_params['features_n']))
    valid_X = valid_X.reshape((valid_X.shape[0], algorithm_params['time_steps'], algorithm_params['features_n']))
    test_X = test_X.reshape((test_X.shape[0], algorithm_params['time_steps'], algorithm_params['features_n']))
    return train_X,valid_X,test_X,train_y,valid_y,test_y,scaler
# 将注意力层的权重应用于数据集的特征上（对于LSTM而言 此处的特征指的是相应时间步上的所有特征，共用一个注意力权重）
def apply_weight(series_X, weight):
    weight = np.array(weight)
    weighted_series_X = series_X * np.expand_dims(weight, axis=1)
    return weighted_series_X
# 判断rmse是否是所有的rmse中的最小值
def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False