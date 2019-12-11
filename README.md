# AutoEncoder_Research
 TEST I-2
Description: 
This project utilizes thee DEC-DA Autoencoder, found here <https://github.com/XifengGuo/DEC-DA> and the DeepWalk Graph Embedding Algorithm, found here: <https://github.com/shenweichen/GraphEmbedding/blob/master/examples/deepwalk_wiki.py>

Our research utilizes autoencoders to advice service decomposition into several  microservices. The Services are composed of Java projects, represented using Neo4j as a weighted edgelist. The edgelist connects methods and classes as two methods might make an call to one another, a class might inherit another and so forth. The generated edgelist  is converted into a embedding layer using deepwalk. In the layer, each node is represented by a feature vector of floats ,length 128. Each of the embedding layers are combined into one file and a autoencoder are trained on the data file. The model is also configured during training to cluster nodes in the embedding layer into a N, user submitted number of clusters. Following training, the autoencoder can be run on individual applications to cluster them into N microserverices and compute metrics including Modularity Q.

Usage:
Command flags are as follows

    #Parameters for Graph Embedding 
    parser.add_argument('--deepwalk' help="Flag to Enable Graph Embedding Using Deepwalk") 
    parser.add_argument('--dp_loc', help="Graph Embedding Data Location") 
    parser.add_argument('--wk_params',help='2 pair: Parameter Name and Type for extra edgelist parameters.Types are int/float/str.') 


    parser.add_argument('--train_dataset', help="Dataset Directory")
    parser.add_argument('-d', '--save-dir',,help="Dir to save the results")
    parser.add_argument('--num_clusters',help="Number Clusters") 

        
    # Parameters for pretraining
    parser.add_argument('--training',  help="Set Training Flag ")
    parser.add_argument('--aug-pretrain', help="Whether to use data augmentation during pretraining phase")
    parser.add_argument('--pretrained-weights', help="Pretrained weights of the autoencoder")
    parser.add_argument('--pretrain-epochs',  type=int,help="Number of epochs for pretraining")
    parser.add_argument('-v', '--verbose',help="Verbose for pretraining")

    # Parameters for clustering
    parser.add_argument('-a', '--analysis',help="Display clustering metrics")
    parser.add_argument( '--edge_list',help="File to Edge List")

    parser.add_argument('-t', '--testing',help="Testing the clustering performance with provided weights")
    parser.add_argument('--test_dataset',help="Test Dataset Directory")


    parser.add_argument('-w', '--weights', help="Model weights, used for testing")
    parser.add_argument('--aug-cluster', help="Whether to use data augmentation during clustering phase")
    parser.add_argument('--optimizer', help="Optimizer for clustering phase")
    parser.add_argument('--lr', help="learning rate during clustering")
    parser.add_argument('--batch-size', help="Batch size")
    parser.add_argument('--maxiter',help="Maximum number of iterations")
    parser.add_argument('-i', '--update-interval',help="Number of iterations to update the target distribution")
    parser.add_argument('--tol', help="Threshold of stopping training")

    parser.add_argument('--experiment', help="Ignore all other arguments and run experiment")

Example commands for various actions:
Coming Soon

