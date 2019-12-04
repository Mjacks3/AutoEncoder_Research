

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




def load_data(dataset):
    x, y = load_other_batch_data(dataset)
    return x.reshape([x.shape[0], -1]), y

"""
if __name__ == "__main__":
    x, y = load_other_batch_data()
    print(x)
    x, y = load_other_data('/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.embedding')
    #print (x)
"""