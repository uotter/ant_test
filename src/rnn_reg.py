import os as os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd  # 导入pandas
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import src.iolib as il
import src.features_genarator as fg
import datetime as datetime

# load data


BATCH_START = 0  # 建立 batch data 时候的 index
TIME_STEPS = 10  # backpropagation through time 的 time_steps
BATCH_SIZE = 5

OUTPUT_SIZE = 1
CELL_SIZE = 10  # RNN 的 hidden unit size
LR = 0.006  # learning rate

strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
holiday_df = pd.read_csv(il.project_root_path + r"/data/holidays.csv")
minute_data = il.read_raw_data()
day_data = il.get_matrix_by_day(minute_data)
date_feature_df_out = fg.get_date_features(holiday_df)
market_feature_df_out = fg.get_rnn_market_features()
combine_features_out = fg.get_combine_features([date_feature_df_out, market_feature_df_out])
dataset_total = fg.generate_rnn_dataset(combine_features_out, day_data)
dataset = dataset_total.ix[:strptime("2017-07-23", "%Y-%m-%d"), :]
X = dataset.ix[:len(dataset) - 4, :len(dataset.columns) - 1]
Y = dataset.ix[:len(dataset) - 4, len(dataset.columns) - 1:]
predict_X = dataset_total.ix[9:, :len(dataset_total.columns) - 1]
train_x = X.ix[:-int(TIME_STEPS * BATCH_SIZE), :]
train_y = Y.ix[:-int(TIME_STEPS * BATCH_SIZE), :]
test_x = X.ix[-int(TIME_STEPS * BATCH_SIZE):, :]
test_y = Y.ix[-int(TIME_STEPS * BATCH_SIZE):, :]
predict_x = predict_X.ix[-int(TIME_STEPS * BATCH_SIZE):, :]


x_train = train_x.values
y_train = train_y.values
x_test = test_x.values
y_test = test_y.values
ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(x_train)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(y_train.reshape(-1, 1))
test_x = ss_x.transform(x_test)
test_y = ss_y.transform(y_test.reshape(-1, 1))
train_x = pd.DataFrame(train_x)
train_y = pd.DataFrame(train_y)
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(test_y)
INPUT_SIZE = len(train_x.columns)


def get_last_batch_ant():
    global train_x, train_y, seq_test
    last_batch_seq = train_x[-int(TIME_STEPS * BATCH_SIZE):].values.reshape((-1, TIME_STEPS, INPUT_SIZE))
    last_batch_res = train_y[-int(TIME_STEPS * BATCH_SIZE):].values.reshape((-1, TIME_STEPS, 1))
    # last_batch_predict_x = test_x[-int(TIME_STEPS * BATCH_SIZE):].values.reshape((-1, TIME_STEPS, INPUT_SIZE))
    return last_batch_seq, last_batch_res  # , last_batch_predict_x


def get_batch_ant():
    global train_x, train_y, BATCH_START, TIME_STEPS, test_x, test_y, predict_x
    x_part1 = train_x[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    y_part1 = train_y[BATCH_START: BATCH_START + TIME_STEPS * BATCH_SIZE]
    # print('时间段=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)

    seq = x_part1.values.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    # print('seq shape=', seq.shape)
    res = y_part1.values.reshape((BATCH_SIZE, TIME_STEPS, 1))
    # print('res shape=', res.shape)
    seq_test = test_x.values.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    res_test = test_y.values.reshape((BATCH_SIZE, TIME_STEPS, 1))
    seq_predict = predict_x.values.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
    BATCH_START += TIME_STEPS

    # returned seq, res and xs: shape (batch, step, input)
    return seq, res, seq_test, res_test, seq_predict


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        '''
        :param n_steps: 每批数据总包含多少时间刻度
        :param input_size: 输入数据的维度
        :param output_size: 输出数据的维度 如果是类似价格曲线的话，应该为1
        :param cell_size: cell的大小
        :param batch_size: 每批次训练数据的数量
        '''
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')  # xs 有三个维度
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')  # ys 有三个维度
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # 增加一个输入层
    def add_input_layer(self, ):
        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度1000的时间线上，每批次50个样本，每个样本20个时刻
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # -1 表示任意行数
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    # 多时刻的状态叠加层
    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # time_major=False 表示时间主线不是第一列batch
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    # 增加一个输出层
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # 预测结果

    def compute_cost(self):
        # losses = tf.contrib.legacy_seq2seq.sequence_loss(
        #     [tf.reshape(self.pred, [-1], name='reshape_pred')],
        #     [tf.reshape(self.ys, [-1], name='reshape_target')],
        #     [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='losses'
        # )
        losses = tf.losses.mean_squared_error(
            tf.reshape(self.pred, [-1], name='reshape_pred'),
            tf.reshape(self.ys, [-1], name='reshape_target'),
            weights=1.0
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    iternum = 200
    train_cost_list = []
    test_cost_list = []
    mae_list = []
    final_pred_y_list = []
    last_batch_seq, last_batch_res = get_last_batch_ant()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    for j in range(iternum):  # 训练200次
        pred_res = None
        # for i in range(int(np.ceil(len(train_x) / TIME_STEPS))):
        for i in range(int(np.ceil(len(train_x) / (BATCH_SIZE * TIME_STEPS)))):
            seq, res, seq_test, res_test, seq_predict = get_batch_ant()

            if i == 0:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)
            # print('{0} cost inside: '.format(i), round(cost, 4))

            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
        # pred_res = sess.run(model.pred, feed_dict={model.xs: seq})
        train_cost_list.append(round(cost, 4))
        pred_test_res, test_cost = sess.run([model.pred, model.cost],
                                            feed_dict={model.xs: seq_test, model.ys: res_test})
        test_res = ss_y.inverse_transform(res_test.flatten()[-50:])
        pred_test_res = ss_y.inverse_transform(pred_test_res.flatten()[-50:])
        mean_absolute_error = tf.reduce_mean(tf.abs(tf.subtract(test_res, pred_test_res)))
        mae = sess.run(mean_absolute_error)
        mae_list.append(mae)
        test_cost_list.append(round(test_cost, 4))
        final_pred_y = sess.run(model.pred, feed_dict={model.xs: seq_predict})
        final_pred_y_list.append((ss_y.inverse_transform(final_pred_y.flatten()[-50:]))[-5:])
        print('{0} cost: '.format(j), round(cost, 4), round(test_cost, 4), mae)

        BATCH_START = 0  # 从头再来一遍

    # 画图
    last_batch_pred_res = sess.run(model.pred,
                                   feed_dict={model.xs: last_batch_seq.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))})
    final_pred_test_res, final_test_cost = sess.run([model.pred, model.cost],
                                                    feed_dict={model.xs: seq_test, model.ys: res_test})

    # print("Predict result: ", final_pred_y[-5:], final_pred_y.shape)
    # print("结果:", last_batch_pred_res.shape)
    # print('实际', last_batch_res.flatten().shape)
    last_batch_pred_res = ss_y.inverse_transform(last_batch_pred_res.flatten()[-50:])
    # pred_res = ss_y.inverse_transform(pred_res.flatten()[-50:])
    last_batch_res = ss_y.inverse_transform(last_batch_res.flatten()[-50:])
    final_pred_test_res = ss_y.inverse_transform(final_pred_test_res.flatten()[-50:])
    test_res = ss_y.inverse_transform(res_test.flatten()[-50:])
    r_size = BATCH_SIZE * TIME_STEPS
    ###画图###########################################################################
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 12))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(3, 1, 1)
    # line1, = axes.plot(X[-int(TIME_STEPS * BATCH_SIZE):].index.tolist(), pred_res, 'b--',
    #                    label='rnn result')
    line2, = axes.plot(X[-int(TIME_STEPS * BATCH_SIZE):].index.tolist(), last_batch_pred_res,
                       'r--',
                       label='rnn train')
    line3, = axes.plot(Y[-int(TIME_STEPS * BATCH_SIZE):].index.tolist(), last_batch_res, 'r',
                       label="true train y")
    plt.legend(handles=[line2, line3])
    axes2 = fig.add_subplot(3, 1, 2)
    line1, = axes2.plot(test_x.index.tolist(), final_pred_test_res,
                        'r--',
                        label='rnn test')
    line6, = axes2.plot(test_x.index.tolist(), test_res, 'r',
                        label="true test y")
    plt.legend(handles=[line1, line6])
    axes3 = fig.add_subplot(3, 1, 3)
    line4, = axes3.plot(range(1, iternum + 1), train_cost_list, 'r', label="train_learning curve")
    line5, = axes3.plot(range(1, iternum + 1), test_cost_list, 'r--', label="test_learning curve")
    axes.grid()
    axes2.grid()
    axes3.grid()
    fig.tight_layout()
    plt.legend(handles=[line4, line5])
    plt.show()
    cost_list = [train_cost_list, test_cost_list, mae_list]
    df = pd.DataFrame(cost_list, index=['train_cost', 'test_cost', 'mae'], columns=range(1, iternum + 1))
    df = df.T
    final_pred_df = pd.DataFrame(final_pred_y_list, index=range(1, iternum + 1),
                                 columns=["0724", "0725", "0726", "0727", "0728"])
    final_df = pd.concat([df, final_pred_df], axis=1)
    final_df.to_excel(il.rnn_result_path_xls)
