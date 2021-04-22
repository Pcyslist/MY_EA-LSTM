import math
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from main import algorithm_params
from utils import get_usable_data, apply_weight
from numpy import concatenate
from sklearn.metrics import mean_squared_error,mean_absolute_error

def visualize_model_performance(algorithm_params):
    train_X, valid_X, test_X, train_y, valid_y, test_y,scaler=get_usable_data(algorithm_params)
    best_model=load_model('best_model.h5')
    with open('best_weight.data','rb') as f:
        best_weight=pickle.load(f)
    pred_y = best_model.predict(apply_weight(test_X, best_weight))
    # 反归一化预测值
    reshaped_test_X = test_X.reshape((test_X.shape[0], algorithm_params['time_steps'] * algorithm_params['features_n']))
    inv_pred_y = concatenate((pred_y, reshaped_test_X[:, -(algorithm_params['features_n'] - 1):]), axis=1)
    inv_pred_y = scaler.inverse_transform(inv_pred_y)
    inv_pred_y = inv_pred_y[:, 0]
    # 反归一化真实值
    reshaped_test_y = test_y.reshape((len(test_y), 1))
    inv_test_y = concatenate((reshaped_test_y, reshaped_test_X[:, -(algorithm_params['features_n'] - 1):]), axis=1)
    inv_test_y = scaler.inverse_transform(inv_test_y)
    inv_test_y = inv_test_y[:, 0]
    print('测试集预测值：', inv_pred_y)
    print('测试集真实值：', inv_test_y)
    print('注意力层：',best_weight)
    print('权重最高的时间步：',best_weight.index(max(best_weight))+1)
    rmse = math.sqrt(mean_squared_error(inv_test_y, inv_pred_y))
    mae = mean_absolute_error(inv_test_y, inv_pred_y)
    print("RMSE: %.4f, MAE: %.4f" % (rmse, mae))
    plt.plot(inv_pred_y[100:200], label='predict')
    plt.plot(inv_test_y[100:200], label='true')
    plt.legend()
    plt.savefig('model_performance.png')
    plt.show()

if __name__=='__main__':
    visualize_model_performance(algorithm_params)