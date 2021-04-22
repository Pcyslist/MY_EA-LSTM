import math
import pickle
from numpy import concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error
from algorithm import pop_weights_init,individual_to_key,select,reconstruct_population,pop_to_weights
from utils import get_usable_data,apply_weight,is_minimum


# 定义模型结构
def make_model(algorithm_params):
    model = Sequential()
    for i in range( algorithm_params['n_hidden_layers'] - 1 ):
        model.add(LSTM(algorithm_params['n_hidden_nodes'], input_shape=(algorithm_params['time_steps'], algorithm_params['features_n']), return_sequences=True))
    model.add(LSTM(algorithm_params['n_hidden_nodes'], return_sequences=False))
    model.add(Dense(algorithm_params['n_output_nodes']))
    return model
# 训练模型
def train(algorithm_params):
    # 得到LSTM模型可以直接使用的数据集
    train_X, valid_X, test_X, train_y, valid_y, test_y,scaler=get_usable_data(algorithm_params)
    # 最好的模型的初始化就是一个简单的LSTM模型
    best_model=make_model(algorithm_params)
    # 最好的权重初始化为每个权重都是1.0
    algorithm_params['best_weight']=[1.0] * algorithm_params['time_steps']
    best_model.compile(loss = algorithm_params['loss'], optimizer = algorithm_params['optimizer'])
    print('在竞争随机搜索之前初始训练一个基本的、注意力层全为1.0的 LSTM 模型......')
    # 将注意力权重应用于训练集上
    weighted_train_X=apply_weight(train_X,algorithm_params['best_weight'])
    best_model.fit( weighted_train_X,train_y,epochs=algorithm_params['epochs'],
                   batch_size=algorithm_params['batch_size'], shuffle=True ,verbose=2)
    print('训练完成，开始竞争随机搜索......')
    # 产生第一代种群及其对应的十进制权重
    pop,weights=pop_weights_init(algorithm_params['pop_size'],algorithm_params['time_steps'],algorithm_params['encode_length'])
    # 对竞争随即搜索的代数进行迭代
    for iteration in range(algorithm_params['iterations']):
        # 对每代中的个体及其权重进行迭代
        for index, (indiv, weight) in enumerate(zip(pop, weights)):
            print('iteration: [%d/%d] indiv_no: [%d/%d]' % (iteration + 1, algorithm_params['iterations'], index + 1, algorithm_params['pop_size']))
            key = individual_to_key(indiv)
            # 只有该个体没有参与过训练才让它去参与模型训练
            if key not in algorithm_params['key_to_rmse'].keys():
                model=make_model(algorithm_params)
                model.compile(loss = algorithm_params['loss'], optimizer = algorithm_params['optimizer'])
                # model.set_weights(best_model.get_weights())
                weighted_train_X=apply_weight(train_X,weight)
                model.fit( weighted_train_X,train_y,epochs=algorithm_params['epochs'],
                   batch_size=algorithm_params['batch_size'], shuffle=True ,verbose=2)
                pred_y = model.predict( apply_weight(valid_X, weight) )
                # 反归一化预测值
                reshaped_valid_X = valid_X.reshape((valid_X.shape[0], algorithm_params['time_steps'] * algorithm_params['features_n']))
                inv_pred_y = concatenate((pred_y, reshaped_valid_X[:, -(algorithm_params['features_n']-1):]), axis=1)
                inv_pred_y=scaler.inverse_transform(inv_pred_y)
                inv_pred_y=inv_pred_y[:,0]
                # 反归一化真实值
                reshaped_valid_y = valid_y.reshape((len(valid_y), 1))
                inv_valid_y=concatenate((reshaped_valid_y, reshaped_valid_X[:, -(algorithm_params['features_n']-1):]), axis=1)
                inv_valid_y=scaler.inverse_transform(inv_valid_y)
                inv_valid_y=inv_valid_y[:,0]
                print('验证集预测值：',inv_pred_y)
                print('验证集真实值：',inv_valid_y)
                rmse = math.sqrt(mean_squared_error(inv_valid_y, inv_pred_y))
                mae = mean_absolute_error(inv_valid_y, inv_pred_y)
                print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))
                if is_minimum(rmse, algorithm_params['key_to_rmse']):
                    best_model.set_weights(model.get_weights())
                    algorithm_params['best_weight'] = weight.copy()
                algorithm_params['key_to_rmse'][key] = rmse
        pop_selected, fitness_selected = select(pop, algorithm_params['n_selected'], algorithm_params['key_to_rmse'])
        pop = reconstruct_population(pop_selected, algorithm_params['pop_size'])
        weights = pop_to_weights(pop, algorithm_params['time_steps'], algorithm_params['encode_length'])
    # 显示挑选出的最佳权重作用在验证集上的最小的rmse和mae。
    pred_y = best_model.predict(apply_weight(valid_X, algorithm_params['best_weight']))
    # 反归一化预测值
    reshaped_valid_X = valid_X.reshape((valid_X.shape[0], algorithm_params['time_steps'] * algorithm_params['features_n']))
    inv_pred_y = concatenate((pred_y, reshaped_valid_X[:, -(algorithm_params['features_n'] - 1):]), axis=1)
    inv_pred_y = scaler.inverse_transform(inv_pred_y)
    inv_pred_y = inv_pred_y[:, 0]
    # 反归一化真实值
    reshaped_valid_y = valid_y.reshape((len(valid_y), 1))
    inv_valid_y = concatenate((reshaped_valid_y, reshaped_valid_X[:, -(algorithm_params['features_n'] - 1):]), axis=1)
    inv_valid_y = scaler.inverse_transform(inv_valid_y)
    inv_valid_y = inv_valid_y[:, 0]
    print('验证集预测值：', inv_pred_y)
    print('验证集真实值：', inv_valid_y)
    rmse = math.sqrt(mean_squared_error(inv_valid_y, inv_pred_y))
    mae = mean_absolute_error(inv_valid_y, inv_pred_y)
    print("用best_weight作用于验证集上得到的最小RMSE: %.4f, 最小MAE: %.4f" % (rmse, mae))

    print('在测试集上进行评估：')
    pred_y = best_model.predict(apply_weight(test_X, algorithm_params['best_weight']))
    # 反归一化预测值
    reshaped_test_X=test_X.reshape((test_X.shape[0], algorithm_params['time_steps'] * algorithm_params['features_n']))
    inv_pred_y = concatenate((pred_y, reshaped_test_X[:, -(algorithm_params['features_n'] - 1):]), axis=1)
    inv_pred_y = scaler.inverse_transform(inv_pred_y)
    inv_pred_y = inv_pred_y[:, 0]
    # 反归一化真实值
    reshaped_test_y = test_y.reshape((len(test_y), 1))
    inv_test_y =concatenate((reshaped_test_y, reshaped_test_X[:, -(algorithm_params['features_n']-1):]), axis=1)
    inv_test_y = scaler.inverse_transform(inv_test_y)
    inv_test_y = inv_test_y[:, 0]
    print('测试集预测值：', inv_pred_y)
    print('测试集真实值：', inv_test_y)
    rmse = math.sqrt(mean_squared_error(inv_test_y, inv_pred_y))
    mae = mean_absolute_error(inv_test_y, inv_pred_y)
    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))
    print('best weight : ', algorithm_params['best_weight'])
    best_model.save('best_model.h5')
    best_weight_file = open('best_weight.data', 'wb')
    pickle.dump(algorithm_params['best_weight'], best_weight_file)
    best_weight_file.close()
    print('训练结束，已在运行目录下产生最好的模型以及与其对应的最好的权重文件。')
    return
