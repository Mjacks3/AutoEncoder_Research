from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from time import time
from FcDEC import FcDEC
from datasets import load_data, load_data_conv

def calculate_metrics(clusters, edge_list='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt' ):
    #cluster_pairs='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.clusters'

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

                if source_dest[0] not in node_partition_mapping:
                    #print("unmapped node: " , source_dest[0])
                    pass #A bug if hit;  each node should be mapped

                elif source_dest[1] not in node_partition_mapping:
                    #print("unmapped node: " , source_dest[1])
                    pass #A bug if hit;  each node should be mapped
                    pass


                elif node_partition_mapping[source_dest[0]] == str(partition) and \
                node_partition_mapping[source_dest[1]] == str(partition):
                    internal += 1
                    incident += 1

                elif node_partition_mapping[source_dest[0]] == str(partition) or \
                node_partition_mapping[source_dest[1]] == str(partition):

                    incident += 1

        modQ += ( (internal/(2.0*m_norm)) -   (incident/(2.0*m_norm))**2)

    #print("Modularity Q Value")
    #print(modQ)



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
                
    print(partion_node_counting)  

    plt.plot([10], [modQ], 'ro')
    plt.show()



def _get_data_and_model(args):
    # prepare dataset
    if args.method =='FcDEC':
        x, y = load_data(args.dataset)
    else:
        raise ValueError("Invalid value for method, which can only be in FcDEC")

    # prepare optimizer
    if args.optimizer in ['sgd', 'SGD']:
        optimizer = SGD(args.lr, 0.9)
    else:
        optimizer = Adam()
    # prepare the model
     
    n_clusters = args.num_clusters
    print("NUM CLUSTERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(args.num_clusters)

    if 'FcDEC' in args.method:
        model = FcDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters)
        model.compile(optimizer=optimizer, loss='kld')
    else:
        raise ValueError("Invalid value for method, which can only be in ['FcDEC']")

    # if -DA method, we'll force aug_pretrain and aug_cluster is True
    if '-DA' in args.method:
        args.aug_pretrain = True
        args.aug_cluster = True

    return (x, y), model


def train(args):
    # get data and model
    (x, y), model = _get_data_and_model(args)
    model.model.summary()

    # pretraining
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):  # load pretrained weights
        model.autoencoder.load_weights(args.pretrained_weights)
    else:  # train
        pass
        pretrain_optimizer = SGD(1.0, 0.9) if args.method in ['FcDEC', 'FcIDEC', 'FcDEC-DA', 'FcIDEC-DA'] else 'adam'
        model.pretrain(x, y = None, optimizer=pretrain_optimizer, epochs=args.pretrain_epochs, batch_size=args.batch_size,
                       save_dir=args.save_dir, verbose=args.verbose, aug_pretrain=args.aug_pretrain)
    t1 = time()
    print("Time for pretraining: %ds" % (t1 - t0))

    # clustering
    # do not fit
    y_pred = model.fit(x, y = None,  maxiter=args.maxiter, batch_size=args.batch_size, update_interval=args.update_interval,
                       save_dir=args.save_dir, aug_cluster=args.aug_cluster)
    t2 = time()
    print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    print('='*60)


def test(args):
    assert args.weights is not None
    (x, y), model = _get_data_and_model(args)  
    model.model.summary()

    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)

    y_pred = model.predict_labels(x)


    node_clusters = []
    for node_id, cluster in zip(y, y_pred):
        print( node_id + " " + cluster)
        node_clusters.append( str(node_id) + " " + str(cluster))

    return node_clusters


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--method', default='FcDEC',
                        choices=['FcDEC', 'FcIDEC', 'ConvDEC', 'ConvIDEC', 'FcDEC-DA', 'FcIDEC-DA', 'ConvDEC-DA', 'ConvIDEC-DA'],
                        help="Clustering algorithm")
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'mnist-test', 'usps', 'fmnist','demo'],
                        help="Dataset name to train on")
    parser.add_argument('-d', '--save-dir', default='results/temp',
                        help="Dir to save the results")
    parser.add_argument('--num_clusters', default=10, type=int,
                        help="Number Clusters")

           

    # Parameters for pretraining
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

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Testing the clustering performance with provided weights")
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
    args = parser.parse_args()
    print('+' * 30, ' Parameters ', '+' * 30)
    print(args)
    print('+' * 75)

    # testing
    if args.testing:
        node_clusters = test(args)
        if args.analysis:
            calculate_metrics(node_clusters)
    else:
        train(args)
