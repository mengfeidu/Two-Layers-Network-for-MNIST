import numpy as np
import random
import copy
import os
## class and functions

# Active functions
def LeakyRelu(input_data,gradient = False):
    if not gradient:
        output = copy.deepcopy(input_data)
        output[output <= 0] = 0.01 * output[output <= 0]
        return output
    else:
        output = copy.deepcopy(input_data)
        output[output > 0] = 1
        output[output <= 0] = 0.01
        return output
    
def Nofun(input_data,gradient = False):
    # use for the output layer
    if gradient:
        return np.ones(input_data.shape)
    else:
        return input_data
    
def Relu(input_data,gradient = False):
    if not gradient:
        output = copy.deepcopy(input_data)
        output[output <= 0] = 0
        return output
    else:
        output = copy.deepcopy(input_data)
        output[output > 0] = 1
        output[output <= 0] = 0
        return output

def Sigmoid(input_data,gradient = False):
    if not gradient:
        return 1.0 / (1.0 + np.exp(-input_data))
    else:
        return input_data * (1.0 - input_data)

# Linear layer
class LinearLayer:
    def __init__(self, input_dim=256, hidden=[10], active_func=LeakyRelu
                 , dropout=0, isbias=True):
        # hidden layers ' structure
        self.hidden = hidden
        self.dropout = dropout
        self.active_func = active_func
        self.isbias = isbias
        self.length = len(hidden)

        self.weight = []
        self.bias = []
        self.weight_grad = [0] * self.length
        self.out_grad = [0] * self.length
        self.bias_grad = [0] * self.length
        self.last_weight = []
        self.last_bias = []
        # outputs of each layer
        self.outputs = [0] * self.length
        self.bias_matrix = []

        # initial for first layer
        tmp = np.ones(shape=[input_dim,1])
        self.bias_matrix.append(tmp)
        # He initial
        self.weight.append(np.random.normal(size=[input_dim, hidden[0]]
                                       , scale=(2/(input_dim*(1+0.01**2)))**0.5))
        self.last_weight.append(self.weight[0])
        self.bias.append(np.zeros(shape=[hidden[0]]))
        self.last_bias.append(self.bias[0])
        # initial for 2 to length - 1 layer
        for i in range(len(hidden)-1):
            tmp = np.ones(shape=[hidden[i],1])
            self.bias_matrix.append(tmp)
            self.weight.append(np.random.normal(size=[hidden[i], hidden[i+1]]
                                                , scale=(2/(hidden[i]*(1+0.01**2)))**0.5))
            self.last_weight.append(self.weight[-1])
            self.bias.append(np.zeros(shape=[hidden[1+i]]))
            self.last_bias.append(self.bias[-1])

        self.ifdrop = True
        if (dropout > 0):
            self.create_dropout()

    def forward(self, input_data):
        if self.dropout > 0 and self.ifdrop:
            #self.create_dropout()
            self.outputs[0] = self.active_func(np.dot(input_data * self.dropout_vec[0], self.weight[0]) + self.bias[0])
            for i in range(self.length - 1):
                self.outputs[i] = self.outputs[i]*self.dropout_vec[i+1]
                z = np.dot(self.outputs[i], self.weight[i + 1]) + self.bias[i + 1]
                self.outputs[i + 1] = self.active_func(z)
            self.final_output = self.outputs[self.length - 1]
            return self.final_output
        else:
            # normal
            self.outputs[0] = self.active_func(np.dot(input_data, self.weight[0]) + self.bias[0])
            for i in range(self.length - 1):
                z = np.dot(self.outputs[i], self.weight[i + 1]) + self.bias[i + 1]
                self.outputs[i + 1] = self.active_func(z)
            self.final_output = self.outputs[self.length - 1]
            return self.final_output



    def backprop(self, input_data, out_grad):
        if self.dropout > 0:
            grad = out_grad
            # update for l-1 to 2 layer
            for i in range(self.length - 1,0,-1):
                grad = grad * self.active_func(self.outputs[i],True)
                self.weight_grad[i] = np.dot(self.outputs[i - 1].T, grad)
                shape = self.outputs[i - 1].shape

                self.bias_grad[i] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
                self.out_grad[i] = np.dot(grad, self.weight[i].T)*self.dropout_vec[i]
                grad = self.out_grad[i]
            # update for first layer
            grad = grad * self.active_func(self.outputs[0],True)
            self.weight_grad[0] = np.dot((input_data*self.dropout_vec[0]).T, grad)
            shape = self.outputs[0].shape
            self.bias_grad[0] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
            self.out_grad[0] = np.dot(grad, self.weight[0].T)*self.dropout_vec[0]
            return self.weight_grad
        else:
            grad = out_grad
            # update for l-1 to 2 layer
            for i in range(self.length - 1,0,-1):
                j = self.length - i - 1
                grad = grad * self.active_func(self.outputs[i],True)
                self.weight_grad[i] = np.dot(self.outputs[i - 1].T, grad)
                shape = self.outputs[i - 1].shape
                self.bias_grad[i] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
                self.out_grad[i] = np.dot(grad, self.weight[i].T)
                grad = self.out_grad[i]
            # update for first layer
            grad = grad * self.active_func(self.outputs[0],True)
            self.weight_grad[0] = np.dot(input_data.T, grad)
            shape = self.outputs[0].shape
            self.bias_grad[0] = np.dot(grad.T, np.ones(shape=[shape[0], 1])).T
            self.out_grad[0] = np.dot(grad, self.weight[0].T)

            return self.weight_grad



    def get_input_gradient(self):
        return self.out_grad[0]

    def update(self, lr=0.001, momentum=0, regularization=0):
        for i in range(len(self.weight)):
            dw = self.weight[i]-self.last_weight[i]
            db = self.bias[i] - self.last_bias[i]
            if not self.isbias:
                self.weight[i] = self.weight[i] - lr * self.weight_grad[i] + momentum * dw - lr * regularization * self.weight[i]
                self.bias[i] = self.bias[i] - lr * self.bias_grad[i] + momentum * db
            else:
                self.bias[i] = self.bias[i] + (- lr * self.bias_grad[i] + momentum * db)
                self.weight[i] = self.weight[i] + (- lr * self.weight_grad[i] + momentum * dw - lr * regularization * self.weight[i]) * \
                            self.bias_matrix[i]

            self.last_weight[i] = self.weight[i]
            self.last_bias[i] = self.bias[i]

    def create_dropout(self):
        # randomly drop neurons in each layer
        # the normal situation is dropout = 0, so the real prob is 1- self.dropout
        p = 1 - self.dropout
        input_dim,_ = self.weight[0].shape
        self.dropout_vec = []
        self.dropout_vec.append(np.random.binomial(1,p,size=[input_dim,]))
        for i in range(self.length-1):
            length,_ = self.weight[i + 1].shape
            # construct the drop out verctor with prob = dropout
            self.dropout_vec.append(np.random.binomial(1,p,size=[length,]))

    def fit_dropout(self):
        # fit dropout
        p = 1- self.dropout
        for i in range(self.length):
            self.weight[i] = self.weight[i] * p
            self.bias[i] = self.bias[i] * p
        self.ifdrop = False

    def change_weight_to_train(self):
        # change weight back to trainning
        p = 1- self.dropout
        for i in range(self.length):
            self.weight[i] = self.weight[i]/p
            self.bias[i] = self.bias[i]/p
        self.ifdrop = True
    

# Loss function
class Mse:
    def loss_value(self, input_data, target):
        self.input = input_data
        self.target = target
        self.batch_size , _ = input_data.shape
        return 0.5*np.sum((self.input - self.target)**2)/self.batch_size
    def gradient(self):
        g = self.input - self.target
        g /= self.batch_size
        return g
    
class NagetiveLog:
    def loss_value(self, input_data, target):
        self.target = target
        self.input = input_data
        self.batch_size, _ = self.target.shape
        exp = np.exp(input_data)

        self.softmax = exp / exp.sum(axis=1, keepdims=True)
        value = np.log(self.softmax)
        self.loss = - (np.sum(value * self.target) / self.batch_size)
        return self.loss
    def gradient(self):
        # one hot type!!!
        p = (self.target * self.softmax).sum(axis=1, keepdims=True)
        self.batch_size, _ = self.target.shape
        g = -(self.target - p) / self.batch_size
        return g

# Network structure
class network:
    # 模型：全连接神经网络，可以选择第一层是否加入卷积层
    # 可以选择是否dropout， 是否将每一层的某一个神经元置为常数（参数bias=True）

    def __init__(self, input_dim, output_dim, hidden, act_fun=LeakyRelu, dropout=0, isbias=False):

        self.hidden_layer = LinearLayer(input_dim, hidden, active_func=act_fun, dropout=dropout, isbias = isbias)
        self.output_layer = LinearLayer(hidden[-1], [output_dim], active_func=Nofun, dropout=dropout, isbias = isbias)

        self.Layers = [self.hidden_layer, self.output_layer]

    def forward(self, input_data):
        self.input = input_data
        in_data = out_data = input_data
        for layer in self.Layers:
            out_data = layer.forward(in_data)
            in_data = out_data
        return out_data

    def backprop(self, grad_of_loss):
        length = len(self.Layers)
        grad_out = grad_of_loss
        for i in range(length-1,0,-1):
            layer = self.Layers[i]
            before_layer = self.Layers[i-1]
            _ = layer.backprop(before_layer.final_output ,grad_out)
            grad_out = layer.get_input_gradient()
        first_layer = self.Layers[0]
        _ = first_layer.backprop(self.input, grad_out)

    def update(self, lr=0.001, momentum=0, regularization=0):
        for layer in self.Layers:
            layer.update(lr, momentum, regularization)


    def train_stop(self):
        for i in self.Layers:
            i.fit_dropout()

    def train_start(self):
        for i in self.Layers:
            i.change_weight_to_train()
    
    def save(self,dir):
        model_dict = {
            'hidden_layer':{'weight':self.hidden_layer.weight,'bias':self.hidden_layer.bias},
            'output_layer':{'weight':self.output_layer.weight,'bias':self.output_layer.bias}
        }
        np.save(dir,model_dict)
        
    def load(self,dir):
        para_dict = np.load(dir,allow_pickle=True).item()
        for i in range(self.hidden_layer.length):
            self.hidden_layer.weight[i] = para_dict['hidden_layer']['weight'][i]
            self.hidden_layer.bias[i] = para_dict['hidden_layer']['bias'][i]
        for i in range(self.output_layer.length):
            self.output_layer.weight[i] = para_dict['output_layer']['weight'][i]
            self.output_layer.bias[i] = para_dict['output_layer']['bias'][i]
        

# One hot codeing
def Onehot(input_data):
    #input = label  - 1
    input_data = input_data.reshape((len(input_data),))
    code = np.zeros((len(input_data), 10))
    code[range(len(input_data)), input_data] = 1
    return code


# Data Loader
class MNISTLoader():
    def __init__(self,train=False,valid = False,normalization = True,shuffle = True
                 ,vague = False):
        self.train = train
        self.normalization = normalization
        self.shuffle = shuffle
        self.index = 0

        if self.train:
            kind = 'train'
            labels_path = os.path.join('data','%s-labels-idx1-ubyte'% kind, '%s-labels.idx1-ubyte'% kind)
            images_path = os.path.join('data','%s-images-idx3-ubyte'% kind, '%s-images.idx3-ubyte'% kind)

            with open(labels_path, 'rb') as lbpath:
                #magic, n = struct.unpack('>II',lbpath.read(8))
                y = np.fromfile(file=lbpath,dtype=np.uint8)[8:]
            with open(images_path, 'rb') as imgpath:
                X = np.fromfile(file=imgpath,dtype=np.uint8)[16:].reshape(len(y), 784)
            self.data = X/255
            self.labels = y

            # one-hot
            self.size = len(self.labels)
            self.labels = self.labels.reshape((self.size,))
            self.labels = Onehot(self.labels)

            
        else:
            kind = 't10k'
            labels_path = os.path.join('data','%s-labels-idx1-ubyte'% kind, '%s-labels.idx1-ubyte'% kind)
            images_path = os.path.join('data','%s-images-idx3-ubyte'% kind, '%s-images.idx3-ubyte'% kind)

            with open(labels_path, 'rb') as lbpath:
                y = np.fromfile(file=lbpath,dtype=np.uint8)[8:].reshape((10000))
            with open(images_path, 'rb') as imgpath:
                X = np.fromfile(file=imgpath,dtype=np.uint8)[16:].reshape(len(y), 784).astype(float)
            
            self.data = X/255
            self.labels = y
        
            # one-hot
            self.size = len(self.labels)
            self.labels = self.labels.reshape((self.size,))
            self.labels = Onehot(self.labels)

        # normalization
        if normalization:
            # mu = np.mean(self.data) # 0.3081
            # sigma = np.std(self.data) # 0.1307
            self.data = (self.data - 0.3081) / 0.1307
        # shuffle
        if shuffle:
            shuffle = np.arange(self.size)
            np.random.shuffle(shuffle)
            self.data = self.data[shuffle]
            self.labels = self.labels[shuffle]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def batch_get(self,batch_size):
        flag = False
        if self.index * batch_size > self.size:
            data = self.data[self.index :self.size, :]
            label = self.labels[self.index :self.size, :]
            self.index = 0
            flag = True
        else:
            data = self.data[self.index:self.index + batch_size, :]
            label = self.labels[self.index:self.index + batch_size, :]
            self.index = (self.index + batch_size) % self.size
        #self.index = (self.index + batch_size) % self.size
        if flag:
            # when an epoch is over, shuffle the dataset
            shuffle = np.arange(self.data.shape[0])
            np.random.shuffle(shuffle)
            self.data = self.data[shuffle]
            self.labels = self.labels[shuffle]

        return data, label





## visulize weight

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

    
def svd_visual1(model_dir, layer_name, idx, is_svd):
    para_dict = np.load(model_dir,allow_pickle=True).item()
    if is_svd:
        U, S, V = np.linalg.svd(para_dict[layer_name]['weight'][idx])
        matrix = U[:100,:100] * S[:100] * V[:100,:100]
    else:
        matrix = para_dict[layer_name]['weight'][idx]
    sns.set(font_scale=1.25)
    ## 主要用sns
    sns.heatmap(matrix,
                cbar=True,
                cmap="RdBu_r",
                #square=True,
                fmt='.2f',
                #  annot_kws={'size': 2}
                )
    plt.savefig('parameter_'+layer_name+str(idx)+'_plot.png')




## loss
from mpl_toolkits.axes_grid1 import host_subplot
def plot_acc_loss(train_loss, test_loss, acc):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()   # 共享x轴
    # set title
    plt.title('Curve of Loss and Accuracy',fontsize=20)
    # set labels
    host.set_xlabel("epochs",fontsize=15)
    host.set_ylabel("loss",fontsize=15)
    par1.set_ylabel("test accuracy",fontsize=15)
 
    # plot curves
    p1, = host.plot(range(len(train_loss)), train_loss, marker='o', markersize=3, linewidth= 1, label="train loss")
    p2, = host.plot(range(len(test_loss)), test_loss, marker='o', markersize=3, linewidth= 1, label="test loss")
    p3, = par1.plot(range(len(acc)), acc, marker='+', markersize=4, linewidth= 1, label="test accuracy")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    # host.axis["left"].label.set_color(p1.get_color())
    # host.axis["left"].label.set_color(p2.get_color())
    # par1.axis["right"].label.set_color(p3.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.show()
    plt.savefig('loss_plot.png')

# data_loss = np.load('loss\loss_batch4_epoch100_decay001_l2001.npy',allow_pickle=True).item()
# a = 1
# plot_acc_loss(data_loss['train_loss'], data_loss['test_loss'], data_loss['test_acc'])
# a = 1