import os
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


if __name__ == "__main__":
    loc = "/home/mjacks3/monize/tahdith/datasets/train"

    for r, d, f in os.walk("/home/mjacks3/monize/tahdith/datasets/train"):
        for file in f:
            if '.txt' in file and 'embedding' not in file:

                print(file)
                print(file[0:-4])
                
                G = nx.read_edgelist(loc +'/'+ file[0:-4]+'/'+ file , create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
                model = DeepWalk(G, walk_length=20, num_walks=200, workers=1)
                model.train(window_size=10, iter=500)
                embeddings = model.get_embeddings()

                f = open(loc +'/'+ file[0:-4]+'/'+ file+".embedding",'w+')

                # Close opend file
                f.write("10 "+ str(len(list(embeddings.items())[0][1]))+"\n")
                for array in embeddings.items():
                    f.write(array[0]+ " ")
                    for number in array[1]:
                        f.write(str(number) + " ")
                    f.write("\n")
                f.close()
                 
        
   # dataset = "spring_micro/spring_micro"

    #G = nx.read_edgelist('../../monize/tahdith/datasets/'+ dataset +'.txt',
                         #create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    #model = DeepWalk(G, walk_length=20, num_walks=200, workers=1)
    #model.train(window_size=10, iter=500)
    #embeddings = model.get_embeddings()
    """
 
    f = open("../../monize/tahdith/datasets/"+ dataset +"_embedding.txt",'w+')
    # Close opend file
    f.write("10 "+ str(len(list(embeddings.items())[0][1]))+"\n")
    for array in embeddings.items():
        f.write(" 0 ")
        for number in array[1]:
            f.write(str(number) + " ")
        f.write("\n")
    f.close()
    
    f = open("../../monize/tahdith/datasets/"+ dataset +"_unused_embedding.txt",'w+')
    # Close opend file
    f.write("10 "+ str(len(list(embeddings.items())[0][1]))+"\n")
    for array in embeddings.items():
        f.write(array[0]+ " 0 ")
        for number in array[1]:
            f.write(str(number) + " ")
        f.write("\n")
    f.close()
    """

    #evaluate_embeddings(embeddings)
    #plot_embeddings(embeddings)