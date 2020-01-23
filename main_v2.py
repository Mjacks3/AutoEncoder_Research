  # -*- coding: utf-8 -*-
# coding: utf-8
#Arg commands kick over very differnernt processes, so imports are moved to functions they are used to save time
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.optimizers import SGD
from FcDEC import FcDEC
import matplotlib.pyplot as plt

#import tensorflow as tf
sub_epoch_count = "" 

def graph_embedding(args):

    import networkx as nx
    from ge import DeepWalk

    print("Beginning Deepwalk Embedding \n")
    print("Displaying Additional Parameters \n")

    data = []

    if args.wk_params and len(args.wk_params) % 2 == 0:
        
        for param, p_type in zip(args.wk_params[0::2],args.wk_params[1::2]) :

            if p_type == 'int':
                data.append((param,int))
            elif p_type == 'float':
                data.append((param,float))
            else :
                data.append((param,str))         

    else:
        print("No Additional Parameters or Uneven Argument Pairs Given. Proceeding...")

    print(data)


    for r, d, f in os.walk(args.dp_loc):
        if len(f) == 2 and os.stat(r+"/"+f[0]).st_size != 0 : #Some Empty EdgeLists!?
            for file in f:
                if '.txt' in file and 'embedding' not in file and 'louvain' not in file:
                    print("\n\n" + file + "\n\n")
                    
                    #data : bool or list of (label,type) tuples
                    G = nx.read_edgelist(args.dp_loc +'/'+ file[0:-4]+'/'+ file , create_using=nx.DiGraph(), nodetype=None, data=data)
                    model = DeepWalk(G, walk_length=20, num_walks=150, workers=10)
                    model.train(window_size=10, iter=100)
                    embeddings = model.get_embeddings()

                    f = open(args.dp_loc +'/'+ file[0:-4]+'/'+ file[0:-4]+".embedding",'w+')

                    # Close opend file
                    f.write("Variable_N "+ str(len(list(embeddings.items())[0][1]))+"\n")

                    for array in embeddings.items():
                        f.write(array[0]+ " ")
                        for number in array[1]:
                            f.write(str(number) + " ")
                        f.write("\n")
                    f.close()  
 
    return 0

def load_other_batch_data(data_path):
    #data_ path should be our trai or test directory
    x = []
    y = []
    data = []
    for r, d, f in os.walk(data_path):
        for file in f:
            if 'embedding' in file:
                with open(r+'/'+ file) as fi:
                    temp = fi.readlines()
                    for line in temp[1:]:
                        #print(line)
                        data.append(line)

    data = [list(map(str, line.split())) for line in data]
    data = np.array(data)
    #print(type(data))
    x, y = data[:, 1:], data[:, 0]
    x = x.astype('float64')
    #x = x.reshape([-1, 4, 4, 1])
    #x = x.reshape([-1, 28, 28, 1]) / 255.

    return x, y.astype(str)

  

def load_data(dataset):
    x, y = load_other_batch_data(dataset)
    return x.reshape([x.shape[0], -1]), y




def _get_data_and_model(args,dataset):
    # prepare dataset
    x, y = load_other_batch_data(dataset)

    # prepare the model
    n_clusters = args.num_clusters
    #print("NUM CLUSTERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(args.num_clusters) #add a loop or some kind of batch
    #print("num clusters")

    model = FcDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
    model.compile(optimizer=SGD(args.lr, 0.9), loss='kld')

    return (x, y), model


def train(args):
    # get data and model
    (x, y), model = _get_data_and_model(args, args.train_dataset)
    model.model.summary()

    # pretraining
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrained_weights)
    else:  
        pretrain_optimizer = SGD(1.0, 0.9)
        i_epoch = 0
        
        for epochs_this_iteration in range(0,510,10):
            temp_save_dir = args.save_dir+ "/"+ str(epochs_this_iteration)
            print(temp_save_dir)

            if not os.path.exists(temp_save_dir):
                os.makedirs(temp_save_dir)

            for r, d, f in os.walk(temp_save_dir):
                print(f)
                if len(f)  < 4:
                    
                    model.pretrain(x, y = None, optimizer=pretrain_optimizer, epochs=epochs_this_iteration - i_epoch, batch_size=args.batch_size,
                                save_dir=temp_save_dir, verbose=args.verbose, aug_pretrain=args.aug_pretrain)
                    #Then fit
                    model.fit(x, y = None,  maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
                                    save_dir=temp_save_dir, aug_cluster=args.aug_cluster)
            i_epoch = epochs_this_iteration
                    

    # clustering
    #y_pred = model.fit(x, y = None,  maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
    #                   save_dir=args.save_dir, aug_cluster=args.aug_cluster)
    
    return 0



def test(args):
    print("Begining with Test Dataset")
    assert args.weights is not None
    (x, y), model = _get_data_and_model(args, args.test_dataset)  
    model.model.summary()

    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)

    y_pred = model.predict_labels(x)
    #print(x)
    #print(y)
    #print(y_pred)


    node_clusters = []

    name = args.test_dataset.split("/")[-1]
    print(name)

    with open(args.test_dataset +"/"+name + "_" + str(args.num_clusters)+ "-"+ str(sub_epoch_count)+ ".clustering",'w+') as f:
        for node_id, cluster in zip(y, y_pred):
            f.write( str(node_id) + " " + str(cluster) +"\n")

        #node_clusters.append( str(node_id) + " " + str(cluster))

    
    #node_clusters
    return 0
    



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='main')


    # Parameters for pretraining
    parser.add_argument('--training', action='store_true', help="Training")
    parser.add_argument('--train_dataset', default='experiment/train',help="Dataset Directory")
    parser.add_argument('-d', '--save-dir', default='experiment/results',help="Dir to save the results")               
    parser.add_argument('--num_clusters', default=3, type=int,help="Number Clusters") #may range this
    parser.add_argument('--aug-pretrain', action='store_true',
                        help="Whether to use data augmentation during pretraining phase")
    parser.add_argument('--pretrained-weights', default=None, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=300, type=int,
                        help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose for pretraining")

    # Parameters for clustering
    parser.add_argument('-a', '--analysis', action='store_true', help="Display clustering metrics")
    parser.add_argument( '--edge_list',default="", help="File to Edge List")

    parser.add_argument('-t', '--testing', action='store_true',help="Testing the clustering performance with provided weights")
    parser.add_argument('--test_dataset', default='datasets/demo_1/acmeair',help="Test Dataset Directory")


    #Add new test locs
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help="Model weights, used for testing")
    parser.add_argument('--aug-cluster', action='store_true',
                        help="Whether to use data augmentation during clustering phase")
    parser.add_argument('--optimizer', default='adam', type=str,
                        help="Optimizer for clustering phase")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="learning rate during clustering")
    parser.add_argument('--batch-size', default=256, type=int,
                        help="Batch size")
    parser.add_argument('--maxiter', default=1000, type=int,
                        help="Maximum number of iterations")
    parser.add_argument('-i', '--update-interval', default=140, type=int,
                        help="Number of iterations to update the target distribution")
    parser.add_argument('--tol', default=0.001, type=float,
                        help="Threshold of stopping training")

    parser.add_argument('--experiment',  action='store_true',
                        help="Ignore all other arguments and run experiment")

    #Parameters for Graph Embedding 
    parser.add_argument('--deepwalk', action='store_true',help="Graph Embedding Using Deepwalk") #dataset to look for embedding
    parser.add_argument('--dp_loc',  default='experiment/train',help="Graph Embedding Data Location") #dataset to look for embedding
    parser.add_argument('--wk_params', action='store',nargs="*", help='Tuple param type for extra edgelist parameters.Types are int/float/str. ') #Use nargs = '*' for multiple arguments  # need sets of two





    args = parser.parse_args()
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)

if args.experiment:

    #Graph Embedding
    """
    
    
    args.dp_loc = "xperiment/train"
    args.wk_params = ["weight", "int"]
    graph_embedding(args)
    
    #End Graph Embedding
    """

    
    #Training
    
    models = [10]
    for num in models:
        args.num_clusters = int(num)
        args.save_dir = "experiment/" + str(args.num_clusters)

        print("NUM CLUSTERS THIS ITR: " + str (args.num_clusters))
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        train(args)
    
    #End Training
    
    
    
    #Test
    #loop to get num clusters +
    # num clusters to loop 
    # 
    # 
    # 

    #args.weights = "experiment/3/model_final.h5"
    #test_dataset = "datasets/demo_1/acmeair"
    #edge_list_loc = "datasets/demo_1/acmeair/acmeair.txt"
    """

    
    cluster_qValue_map = { 
                        "2" : [],
                        "3" : [],
                        "4" : [],
                        "5" : [],
                        "6" : [],
                        "7" : [],
                        "8" : [],
                        "9" : [],
                        "10": [],
                        "11": [],
                        "12": [],
                        "13": [],
                        "14": [],
                        "15": [],
                        "16": [],
                        "17": [],
                        "18": [],
                        "19": []
                        }
    
    models = [5,10]
    for num_clusters in models:
        for sub_epoch_count in range(10,510,10):
            print(sub_epoch_count)
            for r, d, f in os.walk("experiment/test"): 
                if len(f) >= 4 and r.split("/")[-1] +  "_" + str(args.num_clusters)+ "-"+ str(sub_epoch_count)+ ".clustering" not in f  :
                    args.num_clusters = num_clusters
                    args.weights = "experiment/"+str(num_clusters)+"/"+str(sub_epoch_count)+"/model_final.h5"
                    args.test_dataset = r
                    test(args)



    



    models = [5,10]
    for num_clusters in models:
        for r0, d0, f0 in os.walk("experiment/"+ str(num_clusters)):
            print(r0)
            if os.path.exists(r0+"/model_final.h5"):
                args.weights = r0+"/model_final.h5"
                args.num_clusters = num_clusters

                sub_epoch_count = r0.split("/")[-1]
                print(sub_epoch_count)

                for r, d, f in os.walk("experiment/test"): # for each file
                    if len(f) >= 4 and r.split("/")[-1] +  "_" + str(args.num_clusters)+ "-"+ str(sub_epoch_count)+ ".clustering" not in f  :
                        print(num_clusters)
                        for file_name in f: 
                            if ".embedding" in file_name:
                                args.test_dataset = r
                                test(args)
                            else:
                                pass

                        #print(args.test_dataset)
                        #test(args)
                
    """
    #End Test


    
    #Training
#NEXT. VERY IMPORTANT
    #for num_clusters in range (2,20):

        

        #for each num_clusters
        #train
        #save
        #Test 
        #metrics
   

else:

        #Graph Embedding
        if args.deepwalk:
            graph_embedding(args)


        if args.training or  args.testing:
            from tensorflow.keras.optimizers import SGD
            from FcDEC import FcDEC
        #Training
        if args.training:
            train(args)

        # testing
        if args.testing:
            node_clusters = test(args)
            if args.analysis:
                calculate_metrics(node_clusters,args.edge_list)
                pass

