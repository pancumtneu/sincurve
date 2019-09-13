import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_sin=np.linspace(-10,10,200)
c,s=np.cos(x_sin),np.sin(x_sin)
plt.subplot(2,2,2)
plt.plot(x_sin,c,color="blue",linestyle="-",label="COS",alpha=0.5)
plt.plot(x_sin,s,"r*",label="SIN")

x_data=np.linspace(-10,10,300)
out=tf.sigmoid(x_data)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y=sess.run(out)
    plt.subplot(2, 2, 1)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Sigmoid Activation function')

    plt.plot(x_data,y)
    plt.show()


#
# a=tf.constant(2,shape=[2,])
#
# b=tf.constant(3,shape=[2,])
# c=tf.add(a,b)
# with tf.Session() as sess:
#
#     print (sess.run(c))
