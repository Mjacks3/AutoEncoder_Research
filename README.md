# AutoEncoder_Research

Description: 
This project utilizes thee DEC-DA Autoencoder, found here <https://github.com/XifengGuo/DEC-DA> and the DeepWalk Graph Embedding Algorithm, found here: <https://github.com/shenweichen/GraphEmbedding/blob/master/examples/deepwalk_wiki.py>

Our research utilizes autoencoders to advice service decomposition into several  microservices. The Services are composed of Java projects, represented using Neo4j as a weighted edgelist. The edgelist connects methods and classes as two methods might make an call to one another, a class might inherit another and so forth. The generated edgelist  is converted into a embedding layer using deepwalk. In the layer, each node is represented by a feature vector of floats ,length 128. Each of the embedding layers are combined into one file and a autoencoder are trained on the data file. The model is also configured during training to cluster nodes in the embedding layer into a N, user submitted number of clusters. Following training, the autoencoder can be run on individual applications to cluster them into N microserverices and compute metrics including Modularity Q.

Usage:
Command flags are as follows

    parser.add_argument('--train_dataset', default='datasets/demo',help="Dataset Directory")
    parser.add_argument('-d', '--save-dir', default='results/demo',help="Dir to save the results")
    parser.add_argument('--num_clusters', default=2, type=int,help="Number Clusters") #may range this

        
    # Parameters for pretraining
    parser.add_argument('--training', action='store_true', help="Training")

    parser.add_argument('--aug-pretrain', action='store_true',help="Whether to use data augmentation during pretraining phase")
    parser.add_argument('--pretrained-weights', default=None, type=str,help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs', default=2, type=int,help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose', default=1, type=int,help="Verbose for pretraining")

    # Parameters for clustering
    parser.add_argument('-a', '--analysis', action='store_true', help="Display clustering metrics")
    parser.add_argument( '--edge_list',default="", help="File to Edge List")

    parser.add_argument('-t', '--testing', action='store_true',help="Testing the clustering performance with provided weights")
    parser.add_argument('--test_dataset', default='datasets/demo_1/acmeair',help="Test Dataset Directory")



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
    parser.add_argument('--dp_loc',  default='datasets/demo',help="Graph Embedding Data Location") #dataset to look for embedding
    parser.add_argument('--wk_params', action='store',nargs="*", help='Tuple param type for extra edgelist parameters.Types are int/float/str. ') #Use nargs = '*' for multiple arguments  # need sets of two


Example commands for various actions:
Coming Soon

