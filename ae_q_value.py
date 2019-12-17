import os
import  matplotlib.pyplot as plt

#source_dir = "experiment/test/WeatherIconView"
#source_dir = "experiment/test"

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



for r, d, f in os.walk(source_dir):

    #print(f)

    #First get the edgelist to process louvain and clustering
    if r.split("/")[-1]+".embedding" in f:
        for a_file in f:
            if  ".clustering" in a_file or "louvain" in a_file :
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


      

                if "_2.clustering" in a_file:
                    clus_2.append(modQ)

                elif "_3.clustering" in a_file:
                    clus_3.append(modQ)

                elif "_4.clustering" in a_file:
                    clus_4.append(modQ)

                elif "_5.clustering" in a_file:
                    clus_5.append(modQ)

                elif "_6.clustering" in a_file:
                    clus_6.append(modQ)

                elif "_7.clustering" in a_file:
                    clus_7.append(modQ)

                elif "_8.clustering" in a_file:
                    clus_8.append(modQ)

                elif "_9.clustering" in a_file:
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
                
                elif "louvain" in a_file:
                    louvain.append(modQ)


        #print(r+"/"+r.split("/")[-1] +".qvalues")

        """
        with open(r+"/"+r.split("/")[-1] +".qvalues",'w+') as f:
            for clus, num_clus in zip(full_set, range(1,20)):
                if num_clus == 1 :
                    f.write("louvain " + str(clus[-1])+"\n")
                else:
                    f.write(str(num_clus) +" " + str(clus[-1])+"\n")
        """

        


fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(full_set[0:])
plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['Louvain', '2-Cluster', '3-Cluster ', '4-Cluster','5-Cluster','6-Cluster','7-Cluster','8-Cluster ','9-Cluster','10-Cluster','11-Cluster','12-Cluster','13-Cluster','14-Cluster','15-Cluster','16-Cluster','17-Cluster','18-Cluster','19-Cluster'], rotation='vertical')

ax.set_title('AutoEncoder Q-Value Analysis ~ 750 Repository Dataset')
ax.set_xlabel('N_Cluster Size')
ax.set_ylabel('Q-Value')
#plt.ylim(0, 20) 

plt.show()



