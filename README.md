# Two-Layers-Network-for-MNIST

This is a homework of DATA620004 Netural Network and Deep Learning(2023) in the School of Data Science , Fudan University. 

## Introduction
* The details of this homework can be seen in the file **report.pdf**.
* The codes of network structure, loss function, active function, dataloader and visualization can be seen in **utils.py**.
* The codes of network train and test can be seen in **main.py**.

## Set up
* If you need to train the network from the beginning, please download the MNIST dataset into the directory **data**.
* If you need to load the trained model and validate the performance , you can directly run the python file **main.py** following the guide bellow.

* For train the network from the beginning, please run the code:
    ```python
    ## training and save
    best_model, train_loss, test_loss, test_acc = train(nn, loss, train_dataloader, test_dataloader, batch_size=16, epoch=150, lr_start=0.001, momentum=0.9, regularization=0.001, lr_decay=0.001)
    best_model.save('models\model_dict_batch16_epoch150_decay001.npy')

    # loss save
    loss_dict = {
            'train_loss':train_loss,
            'test_loss':test_loss,
            'test_acc':test_acc,
            'batch_size':16,
            'lr':0.001,
            'momentum':0.9,
            'l2':0.001,
            'lr_decay':0.001,
        }
    np.save('loss_batch16_epoch150.npy',loss_dict)
    ##
    ```

## Performance 
### The performance of system1 
| Name       | Test Loss   | Test Acc  |
| --------   | -----  | ----  |
| Model1     | **0.032** |   **0.976**     |
| Model2     | 0.055 |  0.960    |
