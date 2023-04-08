import numpy as np
import pandas as pd
import scipy.io
from utils import *





# train and test function
def test(model,dataloader,batch_size,loss_function):
    correct = 0
    model.train_stop()
    labels = []

    for i in range(1):
        input, label = dataloader.batch_get(batch_size)
        # forward
        model_out = model.forward(input)
        
        loss_value = loss_function.loss_value(model_out,label)
        labels = np.argmax(label, axis=1)
        preds = np.argmax(model_out, axis=1)

        correct = sum(labels == preds)
        
    model.train_start()
    return correct/batch_size, loss_value
def train(model, loss_function, train_dataloader, test_dataloader, batch_size, epoch, lr_start=0.001, momentum=0.0, regularization=0.0, lr_decay=0.009):
    train_loss = 0
    best_acc = 0
    
    save_train_loss =[]
    save_test_loss = []
    save_test = []
    best_model = copy.deepcopy(model)
    best_epoch = 0
    lr = lr_start
    iter_num = int(len(train_dataloader)/batch_size)+1
    for i in range(epoch):
        for _ in range(iter_num):
            input_data, label = train_dataloader.batch_get(batch_size)
            # forward
            model_out = model.forward(input_data)
            loss_value = loss_function.loss_value(model_out,label)
            # backprop an update
            loss_grad = loss_function.gradient()
            model.backprop(loss_grad)
            model.update(lr=lr, momentum=momentum, regularization=regularization)

            train_loss += loss_value

        print('[',(i+1),'/',epoch,']','train loss:', train_loss/iter_num)
        save_train_loss.append(train_loss/iter_num)
        train_loss = 0

        # test data
        test_acc, test_loss = test(model, dataloader=test_dataloader, batch_size=len(test_dataloader), loss_function=loss_function)
        save_test.append(test_acc)
        save_test_loss.append(test_loss)
        print("Accuracy of the model on the test data:",test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = i+1
            # best_labels = labels
            # best_pred = pred
        print("best acc:", best_acc,"       -->     best epoch:", best_epoch)
        ## lr decay
        #(1)
        lr = lr_start * 1.0 / (1.0 + lr_decay * i)
        #(2)
        # lr = lr_start *  np.exp(lr_decay, i)

    return best_model, save_train_loss, save_test_loss, save_test

if __name__ == '__main__':
    
    # Data loader
    train_dataloader = MNISTLoader(train = True)
    test_dataloader = MNISTLoader(train = False)
    # initial network
    nn = network(784,10,[784,100])
    # nn = network(784,10,[784])
    ## loss function
    loss = Mse()
    ## training and save
    # best_model, train_loss, test_loss, test_acc = train(nn, loss, train_dataloader, test_dataloader, batch_size=16, epoch=150, lr_start=0.001, momentum=0.9, regularization=0.001, lr_decay=0.001)
    # best_model.save('models\model_dict_batch16_epoch150_decay001.npy')

    # # loss save
    # loss_dict = {
    #         'train_loss':train_loss,
    #         'test_loss':test_loss,
    #         'test_acc':test_acc,
    #         'batch_size':16,
    #         'lr':0.001,
    #         'momentum':0.9,
    #         'l2':0.001,
    #         'lr_decay':0.001,
    #     }
    # np.save('loss_batch16_epoch150.npy',loss_dict)
    
    
    # load model and test
    # nn.load('models\model_dict_batch16_epoch100_decay001.npy')
    # test_acc,_ = test(nn, dataloader=test_dataloader, batch_size=10000, loss_function=loss)
    
    # loss visualize
    # data_loss = np.load('loss\loss_batch16_epoch100_decay001.npy',allow_pickle=True).item()
    # plot_acc_loss(data_loss['train_loss'], data_loss['test_loss'], data_loss['test_acc'])
    
    # parameter visualize
    svd_visual1('models\model_dict_batch16_epoch100_decay001.npy', 'hidden_layer', 1, is_svd=False)
    a = 1