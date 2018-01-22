#coding:utf-8
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


time_step = 1
input_size = 1
n_hidden_units = 32
batch_size = 128
learning_rate = 0.01


#构造数据
x = np.linspace(0,np.pi*5,1000,dtype=np.float32)
noise = np.random.normal(0,0.1,size=x.shape)
y = np.sin(x) + noise


xs = tf.placeholder(tf.float32,[None,time_step,input_size])
#输入和输出形状相同
ys = tf.placeholder(tf.float32,[None,time_step,input_size])

#定义weights和biases
weights = {
    'in':tf.Variable(tf.random_normal([input_size,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,input_size]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
    'out':tf.Variable(tf.constant(0.1,shape=[input_size]))
}

#RNN输出二维数据
def RNN(x,weights,biases):
    #输入数据形状(batch_size,time_step,input_size)------>RNN输入形状(batch_size,input_size)
    x = tf.unstack(x,time_step,1) #将time_step维度去除
    lstm_cell = rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0)
    #返回outputs,主线state,分线state
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']

output = RNN(xs,weights,biases)
#将数据转为三维
output_reshape = tf.reshape(output,[-1,time_step,input_size])
cost = tf.losses.mean_squared_error(labels=ys,predictions=output_reshape)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    a = plt.figure(1,figsize=(6,5))
    #开始交互模式
    plt.ion()
    for step in range(5000):
        k = 0
        while k < x.shape[0]:
            batch_x = x[k:k+batch_size]
            batch_y = y[k:k+batch_size]
            #将输入转为三维数据
            batch_x = batch_x.reshape(-1,time_step,input_size)
            batch_y = batch_y.reshape(-1,time_step,input_size)
            k = k+batch_size
            _,c = sess.run([train,cost],feed_dict={xs:batch_x,ys:batch_y})
        outputs = sess.run(output,feed_dict={xs:x.reshape(-1,time_step,input_size),ys:y.reshape(-1,time_step,input_size)})
        if step % 500 ==0:
            a.clear()
            plt.plot(x,y,'r-',label='real data')
            plt.legend(loc='best')
            plt.plot(x,outputs,'b-',label='regression data')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()
            plt.pause(0.1)
    plt.ioff()