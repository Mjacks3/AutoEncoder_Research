# -*- coding: utf-8 -*-
# coding: utf-8
#Arg commands kick over very differnernt processes, so imports are moved to functions they are used to save time
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
#import tensorflow as tf


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
        if len(f) == 1:
            for file in f:
                #if '.txt' in file and 'embedding' not in file:
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
                print(file)
                
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
        model.pretrain(x, y = None, optimizer=pretrain_optimizer, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                       save_dir=args.save_dir, verbose=args.verbose, aug_pretrain=args.aug_pretrain)
  

    # clustering
    y_pred = model.fit(x, y = None,  maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
                       save_dir=args.save_dir, aug_cluster=args.aug_cluster)
    
    return 0



def test(args):
    print("test")
    assert args.weights is not None
    (x, y), model = _get_data_and_model(args, args.test_dataset)  
    model.model.summary()

    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)

    y_pred = model.predict_labels(x)
    print(x)
    print(y)
    print(y_pred)


    node_clusters = []
    for node_id, cluster in zip(y, y_pred):
        print( str(node_id) + " " + str(cluster))
        node_clusters.append( str(node_id) + " " + str(cluster))

    return node_clusters




def calculate_metrics(clusters, edge_list='' ):
    #cluster_pairs='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.clusters'
    if not os.path.exists(edge_list):
        print("Cannot Find Edge List. Exiting")
        return 0

    #Assemble Mappings for formula calculations
    node_partition_mapping = {}
    partion_node_counting = {}

    for pair in clusters:
        node_partion = pair.split()
        node_partition_mapping[node_partion[0]] = node_partion[1]

    for a in set(node_partition_mapping.values()):
        partion_node_counting[a] = 0
        for b in node_partition_mapping.items():
            if str(b[1]) == str(a) :
                partion_node_counting[a] += 1

    #Calculate M_Norm, Sum of weights in graph
    m_norm = 0

    with open(edge_list) as f:
        for edge in f:
            m_norm += 1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Modularity Q Calculation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    modQ = 0 

    for partition in partion_node_counting.keys():
        internal = 0 
        incident = 0
        with open(edge_list) as f:
            for edge in f:
                source_dest = edge.split()

                if node_partition_mapping[source_dest[0]] == str(partition) and \
                node_partition_mapping[source_dest[1]] == str(partition):
                    internal += 1
                    incident += 1

                elif node_partition_mapping[source_dest[0]] == str(partition) or \
                node_partition_mapping[source_dest[1]] == str(partition):

                    incident += 1

        modQ += ( (internal/(2.0*m_norm)) -   (incident/(2.0*m_norm))**2)

    print("Modularity Q Value")
    print(modQ)
    """



    for partition_a in partion_node_counting.keys():
        for partition_b in partion_node_counting.keys():
            if partition_a == partition_b:

                
                actual_edges =  0 
                possible_edges =   (partion_node_counting[partition_a] - 1) * partion_node_counting[partition_b]

                with open(edge_list) as f:
                        for edge in f:
                            source_dest = edge.split()

                            if source_dest[0] not in node_partition_mapping:
                                #print("unmapped node: " , source_dest[0])
                                pass #A bug if hit;  each node should be mapped

                            elif source_dest[1] not in node_partition_mapping:
                                #print("unmapped node: " , source_dest[1])
                                pass #A bug if hit;  each node should be mapped


                            elif (str(node_partition_mapping[source_dest[0]]) == partition_a and \
                                str(node_partition_mapping[source_dest[1]]) == partition_b):

                                actual_edges += 1
                #print(partition_a + " to " + partition_b)
                #print(str(actual_edges)+ "/" + str(possible_edges))
                #print()



                

            else:
                #print(partition_a+" "  partition_b " ", end="")
                actual_edges =  0 
                possible_edges =  2* partion_node_counting[partition_a] * partion_node_counting[partition_b]

                with open(edge_list) as f:
                        for edge in f:
                            source_dest = edge.split()

                            if source_dest[0] not in node_partition_mapping:
                                #print("unmapped node: " , source_dest[0])
                                pass #A bug if hit;  each node should be mapped

                            elif source_dest[1] not in node_partition_mapping:
                                #print("unmapped node: " , source_dest[1])
                                pass #A bug if hit;  each node should be mapped


                            elif (str(node_partition_mapping[source_dest[0]]) == partition_a and \
                                str(node_partition_mapping[source_dest[1]]) == partition_b)  or \
                                (str(node_partition_mapping[source_dest[0]]) == partition_b and \
                                str(node_partition_mapping[source_dest[1]]) == partition_a):

                                actual_edges += 1
                #print(partition_a + " to " + partition_b)
                #print(str(actual_edges)+ "/" + str(possible_edges))
                #print()
    """
                
    #print(partion_node_counting)  

    #plt.plot([10], [modQ], 'ro')
    #plt.show()



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--train_dataset', default='datasets/demo',help="Dataset Directory")
                        

    parser.add_argument('-d', '--save-dir', default='results/demo',
                        help="Dir to save the results")
    parser.add_argument('--num_clusters', default=2, type=int,
                        help="Number Clusters") #may range this

           

    # Parameters for pretraining
    parser.add_argument('--training', action='store_true', help="Training")

    parser.add_argument('--aug-pretrain', action='store_true',
                        help="Whether to use data augmentation during pretraining phase")
    parser.add_argument('--pretrained-weights', default=None, type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=2, type=int,
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
    args.dp_loc = "experiment/train"
    args.wk_params = ["weight", "int"]
    graph_embedding(args)
    
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

