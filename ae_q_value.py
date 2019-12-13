import os
import  matplotlib.pyplot as plt

#source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
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

full_set = [clus_2,clus_3,clus_4,clus_5,clus_6,clus_7,clus_8,clus_9,clus_10,clus_11,clus_12,clus_13,clus_14,clus_15, clus_16,clus_17,clus_18,clus_19]

#Note: Take you louvain out. THis is the unoptomized, just pure Qvlues
#There needs to be another with optimized qvalues



for r, d, f in os.walk(source_dir):

    #print(f)

    #First get the edgelist to process louvain and clustering
    if r.split("/")[-1]+".embedding" in f:
        for a_file in f:
            if  ".clustering" in a_file:
                edge_list = r+'/'+r.split("/")[-1]+".txt"

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

                
                m_norm = 0

                with open(edge_list ) as f:
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

                #print("Modularity Q Value")
                #print(modQ)
                #output this qvalue to file
                

                if "2.clustering" in a_file:
                    clus_2.append(modQ)

                elif "3.clustering" in a_file:
                    clus_3.append(modQ)

                elif "4.clustering" in a_file:
                    clus_4.append(modQ)

                elif "5.clustering" in a_file:
                    clus_5.append(modQ)

                elif "6.clustering" in a_file:
                    clus_6.append(modQ)

                elif "7.clustering" in a_file:
                    clus_7.append(modQ)

                elif "8.clustering" in a_file:
                    clus_8.append(modQ)

                elif "9.clustering" in a_file:
                    clus_9.append(modQ)

                elif "10.clustering" in a_file:
                    clus_10.append(modQ)

                elif "11.clustering" in a_file:
                    clus_11.append(modQ)

                elif "12.clustering" in a_file:
                    clus_12.append(modQ)

                elif "13.clustering" in a_file:
                    clus_13.append(modQ)

                elif "14.clustering" in a_file:
                    clus_14.append(modQ)

                elif "15.clustering" in a_file:
                    clus_15.append(modQ)

                elif "16.clustering" in a_file:
                    clus_16.append(modQ)

                elif "17.clustering" in a_file:
                    clus_17.append(modQ)

                elif "18.clustering" in a_file:
                    clus_18.append(modQ)

                elif "19.clustering" in a_file:
                    clus_19.append(modQ)


fig = plt.figure()
fig.suptitle('Microservice A.E. Q-Value Analysis for  750 Repository DataSet', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(full_set)
plt.xticks([1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], ['2-Cluster AE', '3-Cluster AE', '4-Cluster AE','5-Cluster AE','6-Cluster AE','7-Cluster AE','8-Cluster AE','9-Cluster AE','10-Cluster AE','11-Cluster AE','12-Cluster AE','13-Cluster AE','14-Cluster AE','15-Cluster AE','16-Cluster AE','17-Cluster AE','18-Cluster AE','19-Cluster AE'], rotation='vertical')

#ax.set_title('Number Microservices Generated Vs. Metho')
ax.set_xlabel('Generation Method')
ax.set_ylabel('Q-Value')
#plt.ylim(0, 20) 

plt.show()



