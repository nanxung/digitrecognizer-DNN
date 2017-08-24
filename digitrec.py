import tensorflow as tf
import numpy as np
import pandas as pd

move_average_decay=0.99

learning_rate_decay=0.99


learning_rate_base = 0.8


regularization = 0.0001



batch_size=64

train_data=pd.read_csv("./data/train.csv")

test_data=pd.read_csv("./data/test.csv")

labels1=train_data.label.values

labels=[]

for i in labels1:
    z=np.zeros((1,10))

    z[0][i]=1

    labels.append(z[0])

num_data=train_data.shape[0]

train_x_=train_data.loc[:,'pixel0':].values

dataSize=train_x_.shape[0]

test_x=test_data.loc[:,'pixel0':].values
train_x=[]

def convert2gray(img):
    if len(img.shape)>2:
        gray=np.mean(img,-1)
        return gray
    else:
        return img

for x in train_x_:
    x=x.reshape(28,28)
    image=convert2gray(x)
    image1=image.flatten()/255
    train_x.append(image1)


def inf(x,avgclass,w1,w2,b1,b2):
    if avgclass==None:
        y1=tf.nn.relu(tf.matmul(x,w1)+b1)
        return tf.matmul(y1,w2)+b2
    else:
        y1=tf.nn.relu(tf.matmul(x,avgclass.average(w1))+avgclass.average(b1))
        return tf.matmul(y1,avgclass.average(w2))+avgclass.average(b2)

x=tf.placeholder(tf.float32,shape=[None,784],name='x-input')
y_=tf.placeholder(tf.float32,shape=[None,10],name='y-input')

w1=tf.Variable(tf.truncated_normal(shape=[784,500],stddev=0.1,dtype=tf.float32))

w2=tf.Variable(tf.truncated_normal(shape=[500,10],stddev=0.1,dtype=tf.float32))

b1=tf.Variable(tf.constant(0.1,shape=[500]))

b2=tf.Variable(tf.constant(0.1,shape=[10]))

global_step=tf.Variable(0,trainable=False)

learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,dataSize/batch_size,learning_rate_decay,staircase=False)

# a=tf.nn.relu(tf.matmul(x,w1)+b1)

# y__=tf.matmul(a,w2)+b2

y__=inf(x,None,w1,w2,b1,b2)



variable_averages=tf.train.ExponentialMovingAverage(
    move_average_decay,global_step
)
variable_averages_op=variable_averages.apply(tf.trainable_variables())


y=inf(x,variable_averages,w1,w2,b1,b2)

entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y__))+tf.contrib.layers.l2_regularizer(regularization)(w1)+tf.contrib.layers.l2_regularizer(regularization)(w2)

train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(entropy,global_step)

# train_step=tf.train.AdamOptimizer(learning_rate).minimize(entropy)

with tf.control_dependencies([train_step,variable_averages_op]):
    train_op=tf.no_op(name='train')

cor=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
aur=tf.reduce_mean(tf.cast(cor,tf.float32))
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(5000):
        if i%100==0:
            auc=sess.run(aur,feed_dict={x:train_x[-100:],y_:labels[-100:]})
            print("第{}次，准确率为{}".format(i+100,auc))
        start=(i*batch_size)%(dataSize-100)
        end=min(start+batch_size,dataSize-100)
        sess.run(train_op,feed_dict={x:train_x[start:end],y_:labels[start:end]})

    yy = sess.run(y__, feed_dict={x: test_x})
    yl = sess.run(tf.argmax(yy, 1))
    wr = open('res2.csv', 'w')
    print('ImageId,Label', file=wr)
    for i in range(len(yl)):
        print(i + 1, yl[i], sep=',', file=wr)
    wr.close()