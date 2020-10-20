import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from load import load_data

if __name__ == "__main__":
    file = open("parameter.txt", "w")
    # 定义神经网络的参数
    learning_rate = 0.0009  # 学习率
    training_step = 10000  # 训练迭代次数
    testing_step = 6000  # 测试迭代次数
    display_step = 1000  # 每多少次迭代显示一次损失
    # 定义输入和输出
    x = tf.placeholder(tf.float32, shape=(None, 76), name="X_train")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="Y_train")
    # 定义模型参数
    wh = tf.Variable(tf.random_normal([76, 200], stddev=1.0, seed=1))
    bh = tf.Variable(tf.random_normal([200], stddev=1.0, seed=1))
    w = tf.Variable(tf.random_normal([200, 1], stddev=1.0, seed=1))
    b = tf.Variable(tf.random_normal([1], stddev=1.0, seed=1))
    # 定义神经网络的前向传播过程
    Modelh = tf.nn.sigmoid(tf.matmul(x, wh) + bh)
    Model = tf.nn.sigmoid(tf.matmul(Modelh, w) + b)
    keep_prob = tf.placeholder(tf.float32)
    # Model = tf.nn.tanh(tf.matmul(x, w) + b)
    # Model = tf.nn.relu(tf.matmul(x,w) + b)
    """
    对模型进行优化，将Model的值加0.5之后进行取整，
    方便测试准确率(若Model>0.5则优化后会取整为1，反之会取整为0)
    """
    model = Model + 0.5
    model = tf.cast(model, tf.int32)
    y_ = tf.cast(y, tf.int32)
    # Dropout操作：用于防止模型过拟合
    keep_prob = tf.placeholder(tf.float32)
    Model_drop = tf.nn.dropout(Model, keep_prob)
    # 损失函数：交叉熵
    cross_entropy = -tf.reduce_mean(
        y * tf.log(tf.clip_by_value(Model, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - Model, 1e-10, 1.0)))
    """
    优化函数
    即反向传播过程
    主要测试了Adam算法和梯度下降算法，Adam的效果较好
    """
    # 优化器：使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # 优化器：梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # 加载数据and数据预处理

    X_train, X_test, Y_train, Y_test = load_data('D:/source/cicids2018/fri.csv', 50000)
    # X_test_Mn = StandardScaler().fit_transform(X_test)
    b = MinMaxScaler()
    X_test_cen = b.fit_transform(X_test)
    # 1、标准化
    # X_train_Mn = StandardScaler().fit_transform(X_train)
    # 2、正则化 norm为正则化方法：'l1','l2','max'
    # X_train_nor = Normalizer(norm='max').fit_transform(X_train)
    # 3、归一化(centering)
    a = MinMaxScaler()
    X_train_cen = a.fit_transform(X_train)
    # 计算准确度
    correct_prediction = tf.equal(model, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求所有correct_prediction的均值

    # 创建会话运行TensorFlow程序
    with tf.Session() as sess:
        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)
        with pd.ExcelWriter('parameter.xlsx') as writer:
            data1 = pd.DataFrame(sess.run(wh))
            data1.to_excel(writer, sheet_name='wh', float_format='%.6f')
            data2 = pd.DataFrame(sess.run(bh))
            data2.to_excel(writer, sheet_name='bh', float_format='%.6f')
            data3 = pd.DataFrame(sess.run(w))
            data3.to_excel(writer, sheet_name='w', float_format='%.6f')
        # 训练测试集
        for i in range(training_step):
            # 训练模型运行语句（采用矩阵运算将训练时间减少至十几秒）
            sess.run(optimizer, feed_dict={x: X_train_cen, y: Y_train, keep_prob: 0.5})
            # 每迭代1000次输出一次日志信息
            # display = (i % 10)
            if i % display_step == 0:
                # 输出交叉熵之和
                total_cross_entropy_train = sess.run(cross_entropy, feed_dict={x: X_train_cen, y: Y_train})
                print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy_train))
                # 输出准确度
                # 每10轮迭代计算一次准确度
                accuracy_rate = sess.run(accuracy, feed_dict={x: X_train_cen, y: Y_train, keep_prob: 1.0})
                print('第' + str(i) + '轮,Training的准确度为：' + str(accuracy_rate))
        # 测试数据集
        for i in range(testing_step):
            # 通过选取样本训练神经网络并更新参数
            sess.run(optimizer, feed_dict={x: X_test_cen, y: Y_test})
            # 每迭代1000次输出一次日志信息
            # display1 = (i % 10)
            if i % display_step == 0:
                # 计算所有数据的交叉熵
                total_cross_entropy_test = sess.run(cross_entropy, feed_dict={x: X_test_cen, y: Y_test})
                print("After %d testing step(s),cross entropy on all data is %g" % (i, total_cross_entropy_test))
                # if (display1 == 0) and (i<len(X_test)):
                accuracy_rate1 = sess.run(accuracy, feed_dict={x: X_test_cen, y: Y_test, keep_prob: 1.0})
                print('第' + str(i) + '轮,Testing的准确度为：' + str(accuracy_rate1))
        # 输出预测的结果和期望的结果
        pred_Y_test = sess.run(Model, feed_dict={x: X_test_cen})
        print('测试结果如下：')
        for pred, real in zip(pred_Y_test, Y_test):
            print(pred, real)
        print('\n\n')
        # 将预测值加0.5取整，便于绘制混淆矩阵
        pred_Y_test = pred_Y_test + 0.5
        pred_Y = np.trunc(pred_Y_test)
        # 输出混淆矩阵
        Confused = confusion_matrix(y_true=Y_test, y_pred=pred_Y)
        print('Confusion_matrix is down right here:')
        print(Confused, end='\n\n')
        # 绘制混淆矩阵
        flg, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(Confused, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(Confused.shape[0]):
            for j in range(Confused.shape[1]):
                ax.text(x=j, y=i, s=Confused[i, j], va='center', ha='center')
        plt.xlabel('pred_Y_test')
        plt.ylabel('Y_teat')
        plt.show()
