import os
import  matplotlib.pyplot as plt

#source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

folders = ["AE_I_2","AE_L","AE_4","AE_5","AE_6"]





full_set = []


for folder in folders:

    clus_2 = []
    clus_5 = []
    clus_10 = []
    clus_15 = []

    print(folder)
    for r, d, f in os.walk("../"+folder+"/"+"AutoEncoder_Research/experiment/test"):

        #First get the edgelist to process louvain and clustering
        if r.split("/")[-1]+".embedding" in f:
            for a_file in f:
                if  ".clustering" in a_file  :
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
                                if source_dest[0] in node_partition_mapping and  source_dest[1] in node_partition_mapping:
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

         
                    elif "_5.clustering" in a_file:
                        clus_5.append(modQ)

      

                    elif "10.clustering" in a_file:
                        clus_10.append(modQ)

        

                    elif "15.clustering" in a_file:
                        clus_15.append(modQ)
    

    full_set.append(clus_2)
    full_set.append(clus_5)
    full_set.append(clus_10)
    full_set.append(clus_15)
            

fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(full_set[0:])
plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], [ '2C-100TD', '5C-100TD', '10C-100TD ', '15C-100TD','2C-200TD','5C-200TD','10C-200TD','15C-200TD ','2C-300TD','5C-300TD','10C-300TD','15C-300TD','2C-400TD','5C-400TD','10C-400TD','15C-400TD','2C-500TD','5C-500TD','10C-500TD','15C-500TD'], rotation='vertical')

ax.set_title('AutoEncoder Training Set Size Vs. Q-Value Analysis [300 Epochs] ')
ax.set_xlabel('[X] Cluster Model -Trained on- [Y] Training Data Set Size')
ax.set_ylabel('Q-Value')
#plt.ylim(0, 20) 

plt.show()

