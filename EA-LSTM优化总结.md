# EA-LSTM优化总结

## 数据预处理

- 对日期信息的合并和设为索引的处理

- 对缺失值的填充处理

- 对分类变量的数值离散化处理

- 对所有特征的归一化处理

- 预测任务：

  1. 前time_steps时间序列的所有特征，预测第time_steps时间步的pm2.5

  2. 前time_steps时间序列的所有特征，预测第time_steps+1时间步的pm2.5

  3. 前time_steps时间序列的所有特征，预测后predict_time_steps的pm2.5

- 训练集、验证集、测试集的划分比例

## 竞争随机搜索CRS的参数

- 迭代次数 iterations
- 每一代种群的大小pop_size
- 每个种群的分组个数n_group也即每次迭代挑选出的个体数目n_selected
- 种群中每个个体的单位编码长度code_length

## LSTM模型参数

- 输入序列时间步数time_steps
- 隐层节点个数n_hidden_node
- 隐层个数n_hidden_layer
- 训练的迭代次数epoch
- 批量梯度下降的batch_size大小
- 优化算法的选择adam或者sgd
- 学习率learning_rate
- 对每个Attention权重训练模型时，是否加载best_model的weights作为新model的初始值

## Attention层的问题

- 每个时间步的所有特征是否共享一个权重，若不共享，则注意力层种群的每个个体的基因序列会很长。
  - 共享时基因序列长度为：time_steps * encode_length
  - 不共享时基因序列长度：time_steps * features * encode_length
  - 共享时权重个数：time_steps
  - 不共享时权重个数：time_steps * features

- 权重之和应该是1，生成每个个体对应的权重后，通过softmax来使得权重之和为1
- 注意力层位置：
  - Input_layer -> Attention_layer -> LSTM_layer->output_layer
  - input_layer -> LSTM_layer ->Attention_layer ->output_layer

## 代码性能问题

- 训练模型时内存暴增，已解决，避免了迭代训练每个权重的模型时重复定义模型
- 预测任务1时取得的预测rmse和mae都非常小，预测效果很好
- 预测任务2、3时取得的预测rmse稳居25以上，预测效果一般
  - 尝试的解决办法：
    1. 在数据集上做文章：
       1. 精简特征个数，挑选有用的特征，这里的有用，指的是从经验的角度，认为该特征有利于待预测的特征的预测，它们之间的相关系数应该很高。
       2. 增加或者减小历史时间步time_steps
    2. 在竞争随机搜索上做文章：
       1. 增加迭代的次数
       2. 增加每代的个体数
       3. 减少？或者增加从每代中选出用于重建种群的个体数
    3. 在LSTM模型上做文章
       1. 提高LSTM模型的复杂度（LSTM层数，每层的节点个数）
       2. 采用其他的优化算法
       3. 模型的训练epoch数（其实意义不大，因为每个注意力个体都会继承当前最好的模型的权重来继续训练，意味着前epoch训练的成果能够得到记忆）
    4. 在Attentions层上做文章：
       1. 增加权重维度
       2. 改变注意力层位置或者增加注意力层