import os
import  matplotlib.pyplot as plt

#source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain = []
clus_2 = []
clus_3 = []
clus_4 = []
clus_5 = []
clus_6 = []
clus_7 = []
clus_8 = []
clus_9 = []
clus_10 = []
clus_11 = []
clus_12 = []
clus_13 = []
clus_14 = []
clus_15 = []
clus_16 = []
clus_17 = []
clus_18 = []
clus_19 = []
full_set = [louvain, clus_2,clus_3,clus_4,clus_5,clus_6,clus_7,clus_8,clus_9,clus_10,clus_11,clus_12,clus_13,clus_14,clus_15, clus_16,clus_17,clus_18,clus_19]


coupling = [] 

for r, d, f in os.walk(source_dir):
  if r.split("/")[-1]+".embedding" in f:
    edge_list_file = r+"/"+r.split("/")[-1] +".txt"
    edge_list = []
    with open(edge_list_file) as fi:
        edge_list = fi.read().splitlines()
    for a_file in f:
        if "louvain" in a_file or "clustering" in a_file:
            
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
                #print(partion_node_counting)
                score = 0
                for parition in partion_node_counting.keys():
                    cnt  = 0.00
                    for edge in edge_list:
                        if node_partition_mapping[edge.split()[0]] == parition and node_partition_mapping[edge.split()[1]] == parition:
                            cnt += 1
                    if int (partion_node_counting[parition] *(partion_node_counting[parition] - 1) ) != 0:
                        score = cnt/ float(partion_node_counting[parition] *(partion_node_counting[parition] - 1) )


                    
                    if "_2.clustering" in a_file:
                        clus_2.append(score)

                    elif "_3.clustering" in a_file:
                        clus_3.append(score)

                    elif "_4.clustering" in a_file:
                        clus_4.append(score)

                    elif "_5.clustering" in a_file:
                        clus_5.append(score)

                    elif "_6.clustering" in a_file:
                        clus_6.append(score)

                    elif "_7.clustering" in a_file:
                        clus_7.append(score)

                    elif "_8.clustering" in a_file:
                        clus_8.append(score)

                    elif "_9.clustering" in a_file:
                        clus_9.append(score)

                    elif "10.clustering" in a_file:
                        clus_10.append(score)

                    elif "11.clustering" in a_file:
                        clus_11.append(score)

                    elif "12.clustering" in a_file:
                        clus_12.append(score)

                    elif "13.clustering" in a_file:
                        clus_13.append(score)

                    elif "14.clustering" in a_file:
                        clus_14.append(score)

                    elif "15.clustering" in a_file:
                        clus_15.append(score)

                    elif "16.clustering" in a_file:
                        clus_16.append(score)

                    elif "17.clustering" in a_file:
                        clus_17.append(score)

                    elif "18.clustering" in a_file:
                        clus_18.append(score)

                    elif "19.clustering" in a_file:
                        clus_19.append(score)
                    
                    elif "isolated_louvain" in a_file:
                        louvain.append(score)         

full_set_minus_zeros = []
for item in full_set:
    full_set_minus_zeros.append([i for i in item if float(i) != 0.0])

        
print(full_set_minus_zeros[2])

fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(full_set_minus_zeros[0:])
plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['Louvain', '2-Cluster', '3-Cluster ', '4-Cluster','5-Cluster','6-Cluster','7-Cluster','8-Cluster ','9-Cluster','10-Cluster','11-Cluster','12-Cluster','13-Cluster','14-Cluster','15-Cluster','16-Cluster','17-Cluster','18-Cluster','19-Cluster'], rotation='vertical')

ax.set_title('AutoEncoder Coupling Analysis ~ 750 Repository Dataset')
ax.set_xlabel('N_Cluster Size')
ax.set_ylabel('Coupling Value')
plt.ylim(0, .6) 

plt.show()

