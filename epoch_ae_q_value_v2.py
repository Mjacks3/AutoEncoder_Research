import os
import  matplotlib.pyplot as plt
import seaborn as sns
import statistics

source_dir = "experiment/test/SeeWeather"
#source_dir = "experiment/test"


models = [2,5,10,15]
num_epochs = [100,200,300,400,500]


clus_2 = []
clus_5 = []
clus_10 = []
clus_15 = []



count = 0 

for r, d, f in os.walk(source_dir):
        count +=1
        print("Progress: "+ str(count/(753.0 *1)))

        if r.split("/")[-1]+".embedding" in f:
            edge_list = r+'/'+r.split("/")[-1]+".txt"

            for model in models:
                e_100 = []
                e_200 = []
                e_300 = []
                e_400 = []

                e_500 = []
                for sub_model in num_epochs:
                        clust_file = r+"/"+r.split("/")[-1]+"_"+str(model)+"-"+str(sub_model)+".clustering"

                        with open(clust_file) as fi:
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

                        if sub_model == 100 :
                            e_100.append(modQ)
                        elif sub_model == 200:
                            e_200.append(modQ)
                        elif sub_model == 300 :
                            e_300.append(modQ)
                        elif sub_model == 400:
                            e_400.append(modQ)
                        else: 
                            e_500.append(modQ)
                #here
                if model == 2 :
                    clus_2.append(statistics.mean(e_100))
                    clus_2.append(statistics.mean(e_200))
                    clus_2.append(statistics.mean(e_300))
                    clus_2.append(statistics.mean(e_400))
                    clus_2.append(statistics.mean(e_500))

                elif model == 5:
                    clus_5.append(statistics.mean(e_100))
                    clus_5.append(statistics.mean(e_200))
                    clus_5.append(statistics.mean(e_300))
                    clus_5.append(statistics.mean(e_400))
                    clus_5.append(statistics.mean(e_500))

                elif model == 10 :
                    clus_10.append(statistics.mean(e_100))
                    clus_10.append(statistics.mean(e_200))
                    clus_10.append(statistics.mean(e_300))
                    clus_10.append(statistics.mean(e_400))
                    clus_10.append(statistics.mean(e_500))

                elif model == 15:
                    clus_15.append(statistics.mean(e_100))
                    clus_15.append(statistics.mean(e_200))
                    clus_15.append(statistics.mean(e_300))
                    clus_15.append(statistics.mean(e_400))
                    clus_15.append(statistics.mean(e_500))


                

    
    
for a,b,c,d in zip(clus_2,clus_5,clus_10,clus_15):
    full_set.append([0,b,c,d])



fig = plt.figure()

ax = fig.add_subplot(111)
#ax.boxplot(full_set[0:])


sns.heatmap(full_set, vmin=0,vmax=.3,annot=True, cmap ="PuBu",cbar_kws={'label': 'Mean Modularity Q Score of 753 Sized Test Dataset '} ,yticklabels = ['100 Epochs','200 Epochs','300 Epochs','400 Epochs','500 Epochs'],  xticklabels = ['2-Cluster','5-Cluster','10-Cluster','15-Cluster'])

ax.set_title('AutoEncoder Epoch Size Vs.Model Type Vs. Q-Value Analysis')

ax.set_xlabel('Model Type')
ax.set_ylabel('Model Training Epoch Size')
#plt.ylim(0, 20) 

plt.show()
