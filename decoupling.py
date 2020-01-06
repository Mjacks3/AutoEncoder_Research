import os
import  matplotlib.pyplot as plt

source_dir = "experiment/test/WeatherIconView"
#source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain = []
autoencoder = []


data = [louvain, autoencoder]


for r, d, f in os.walk(source_dir):
    edge_list = r+"/"+r.split("/")[-1] +".txt"
    for a_file in f:
        if "louvain" in f or "clustering" in f:
                with open(r+'/'+ a_file) as fi:
                    clusters = fi.read().splitlines()

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
                
                #print(node_partition_mapping)

