# -*- coding: utf-8 -*-
# coding: utf-8
import argparse
import os
import networkx as nx
from ge import DeepWalk


def graph_embedding(data_location, em_params):
    print("Beginning Deepwalk Embedding \n")
    print("Displaying Additional Parameters \n")

    data = []

    if em_params and len(em_params) % 2 == 0:
        
        for param, p_type in zip(em_params[0::2],em_params[1::2]) :

            if p_type == 'int':
                data.append((param,int))
            elif p_type == 'float':
                data.append((param,float))
            else :
                data.append((param,str))         

    else:
        print("No Additional Parameters or Uneven Argument Pairs Given. Proceeding...")

    print(data)


    for r, d, f in os.walk(data_location):
        for file in f:
            if '.txt' in file and 'embedding' not in file:
                print("\n\n" + file + "\n\n")
                
                #data : bool or list of (label,type) tuples
                G = nx.read_edgelist(data_location +'/'+ file[0:-4]+'/'+ file , create_using=nx.DiGraph(), nodetype=None, data=data)
                model = DeepWalk(G, walk_length=20, num_walks=200, workers=1)
                model.train(window_size=10, iter=500)
                embeddings = model.get_embeddings()

                f = open(data_location +'/'+ file[0:-4]+'/'+ file+".embedding",'w+')

                # Close opend file
                f.write("Variable_N "+ str(len(list(embeddings.items())[0][1]))+"\n")

                for array in embeddings.items():
                    f.write(array[0]+ " ")
                    for number in array[1]:
                        f.write(str(number) + " ")
                    f.write("\n")
                f.close()  
    return 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--deepwalk', action='store_true',help="Graph Embedding Using Deepwalk") #dataset to look for embedding
    parser.add_argument('--dp_loc',  default='datasets/demo',help="Graph Embedding Data Location") #dataset to look for embedding
    parser.add_argument('--wk_params', action='store',nargs="*", help='Tuple param type for extra edgelist parameters.Types are int/float/str. ') #Use nargs = '*' for multiple arguments  # need sets of two


    args = parser.parse_args()
    if args.deepwalk:
        graph_embedding(args.dp_loc,args.wk_params)

