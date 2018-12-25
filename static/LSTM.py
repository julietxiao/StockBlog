# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tushare as ts
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

rnn_unit = 10  # 隐层神经元的个数
lstm_layers = 2  # 隐层层数
input_size = 7
output_size = 1
lr = 0.0006  # 学习率
train_times = 2000  #训练次数

# 获取训练集
# batch_size:每批次训练样本数
# time_step:时间步
# train_begin,train_end:训练集的数量
def get_train_data(data,batch_size, time_step, train_begin, train_end):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        # 这里要注意7是要训练的属性个数
        x = normalized_train_data[i:i + time_step, :7]
        # y是label,所以选取一个当label的属性即可,这里我选取的是最高价属性
        y = normalized_train_data[i:i + time_step, 7, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

# 获取测试集
# 把样本中除了训练集中剩下的数据当成训练集
def get_test_data(data,time_step=20, test_begin=400):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
    return mean, std, test_x, test_y

# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置、dropout参数

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# ——————————————————定义神经网络变量——————————————————
def lstmCell():
    # basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

# ————————————————训练模型————————————————————

def train_lstm(code, data,batch_size=40, time_step=20, train_begin=0, train_end=400):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(data,batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_times):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                 keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, './model/model_save'+code+'/model.ckpt'))
        # I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        # if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")

# ————————————————预测模型————————————————————
def prediction(code,data,time_step,test_begin, datelist):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(data,time_step,test_begin)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('./model/model_save'+code)
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
        acc = round(acc, 4)
        print("The accuracy of this predict:", acc)
        # 以折线图表示结果
        plt.figure()
        l = test_predict.shape[0]
        xdatelist = datelist[0:l]
        xdatelist = xdatelist[::-1]

        xs1 = [datetime.strptime(d, '%Y-%m-%d').date() for d in xdatelist]
        # plt.plot(list(range(len(test_predict))), test_predict, color='b', )

        plt.plot(xs1, test_predict, color='b', )

        l2 = test_y.shape[0]
        xdatelist2 = datelist[0:l2]
        xdatelist2 = xdatelist2[::-1]
        xs2 = [datetime.strptime(d, '%Y-%m-%d').date() for d in xdatelist2]


        # plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.plot(xs2, test_y, color='r')

        plt.xlabel(u'日期')
        plt.ylabel(u'ma5(5日均价)')
        plt.title(u'股价预测精度为:' +str(acc) )
        # plt.legend([L1, L2], ['test1', 'test2'], loc='upper right')
        # X坐标，将str类型的数据转换为datetime.date类型的数据，作为x坐标
        # xs = [datetime.strptime(d, '%Y/%m/%d').date() for d in l_time]
        plt.gcf().autofmt_xdate()
        # plt.show()
        plt.savefig("./pic/"+code+"pre.png")




def LSTM(code):
    ori_data = ts.get_hist_data(code)
    # 选取其中8个属性,7个属性作为预测,1个属性作为label
    data = ori_data.iloc[:, 0:8].values
    # 按照日期先后翻转一下
    data = data[::-1]
    # 该股票共有N条数据
    N = data.shape[0]
    # 前2/3数据用于训练,后1/3数据用于预测
    batch_size = int(N/20);
    time_step = 10
    train_begin = 0
    train_end = int((N/3)*2)

    datelist = ori_data.index
    # datelist = datelist[::-1]

    train_lstm(code, data,batch_size, time_step, train_begin, train_end)
    prediction(code,data,time_step,train_end, datelist)

# codelist = []
# for line in open("data.txt","r"): #设置文件对象并读取每一行文件
#     line = line[:-1]
#     codelist.append(line)
# for code in codelist:
#     LSTM(code)

LSTM('300532')
# round(000020matrix[i,j], decPts)
