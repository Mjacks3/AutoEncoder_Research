import os
import  matplotlib.pyplot as plt
import seaborn as sns
import statistics

source_dir = "experiment/test/WeatherIconView"
#source_dir = "experiment/test"

folders = ["e_1","e_2","e_3","e_4","e_5"]

count = 0



full_set = []


for folder in folders:

    clus_2 = []
    clus_5 = []
    clus_10 = []
    clus_15 = []

    print(folder)
    for r, d, f in os.walk("../"+folder+"/"+"AutoEncoder_Research/experiment/test"):
        count+= 1
        print("Progress: "+ str(count/(753.0 *1)))

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
                            m_norm += int(edge.split()[-1])

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
                                        internal += int(edge.split()[-1])
                                        incident += int(edge.split()[-1])

                                    elif node_partition_mapping[source_dest[0]] == str(partition) or \
                                    node_partition_mapping[source_dest[1]] == str(partition):

                                        incident += int(edge.split()[-1])

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
    
    
    full_set.append([statistics.mean(clus_2),statistics.mean(clus_5),statistics.mean(clus_10),statistics.mean(clus_15)])



fig = plt.figure()

ax = fig.add_subplot(111)
#ax.boxplot(full_set[0:])


sns.heatmap(full_set, vmin=0,vmax=.3,annot=True, cmap ="PuBu",cbar_kws={'label': 'Mean Modularity Q Score of 753 Sized Test Dataset '} ,yticklabels = ['100 Epochs','200 Epochs','300 Epochs','400 Epochs','500 Epochs'],  xticklabels = ['2-Cluster','5-Cluster','10-Cluster','15-Cluster'])

#ax.set_title('AutoEncoder Epoch Size Vs.Model Type Vs. Q-Value Analysis')

ax.set_xlabel('Model Type')
ax.set_ylabel('Model Training Epoch Size')
#plt.ylim(0, 20) 

plt.show()

