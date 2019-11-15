import numpy as np

import os


def load_other_batch_data(data_path='/home/mjacks3/monize/tahdith/datasets/train'):
    #Intended just for training our data
    full_x = []

    for r, d, f in os.walk(data_path):
        for file in f:
            if 'embedding' in file:
                with open(data_path +'/'+ file[0:-14]+'/'+ file) as fi:
                    data = fi.readlines()

                    data = data[1:-1]
                    data = [list(map(str, line.split())) for line in data]
                    data = np.array(data)
                    x, y = data[:, 1:], data[:, 0]

                    for ind in range (len(x)):
                        x[ind] = x[ind].astype(float)

                    full_x  = np.append(full_x, x)


    return full_x, 0

def load_other_data(data_path='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.embedding'):
    # for demo training and for tesrt
    print(data_path)

    with open(data_path) as f:
        data = f.readlines()

    data = data[1:-1]
    data = [list(map(str, line.split())) for line in data]
    data = np.array(data)
    x, y = data[:, 1:], data[:, 0]

    for ind in range (len(x)):
        x[ind] = x[ind].astype(float)
   # x = [list(map(float , x)) for num in x]
    #x = x.reshape([-1, 16, 16, 1])
    return x, y


    

def load_usps(data_path='./data/usps'):
    if not os.path.exists(data_path+'/usps_train.jf'):
        raise ValueError("No data for usps found, please download the data from links in \"./data/usps/download_usps.txt\".")

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y


  

def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_mnist_test():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y



def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()
    elif dataset == "demo":
        return load_other_data()
    else:
        raise ValueError('Not defined for loading %s' % dataset)


def load_data(dataset):
    x, y = load_data_conv(dataset)
    return x.reshape([x.shape[0], -1]), y

"""
if __name__ == "__main__":
    x, y = load_other_batch_data()
    print(x)
    x, y = load_other_data('/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.embedding')
    #print (x)
"""