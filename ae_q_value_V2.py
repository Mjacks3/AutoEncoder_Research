import os
import statistics
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

source_dir = "experiment/test/Short_DayTrader"
source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"
modq_values  = []


for subset in range(1,20,1):
    modq_values.append([])


count = 0
for r, d, f in os.walk(source_dir):
    count+= 1
    print("Progress: "+ str(count/753.0))
    m_norm = 0

    basename = r.split("/")[-1]

    #Get Edgelist and Calculate M_Norm Value  
    if basename+".txt" in f:
        edge_list = r+'/'+basename+".txt"

        m_norm = 0
        with open(edge_list ) as f:
            for edge in f:
                m_norm += int(edge.split()[-1])

    for subset in range(1,20,1):
        if basename != 'test':

            if subset == 1:
                clusterfile = basename+".isolated_louvain"
            else:
                clusterfile = basename+"_"+str(subset)+".clustering"

            with open(r+"/"+clusterfile) as fi:
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
            #print(int(subset/10))

            modq_values[int(subset) -1].append(modQ)



#avg_mod_q_values = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 35}

matplotlib.rc('font', **font)

fig = plt.figure()

ax = fig.add_subplot(111)

bp = ax.boxplot(modq_values[0:])

for box in bp['boxes']:
    box.set(linewidth=10)
    
plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [ 'Louvain', '2-AE', '3-AE ', '4-AE','5-AE','6-AE','7-AE','8-AE ','9-AE','10-AE','11-AE','12-AE','13-AE','14-AE','15-AE','16-AE','17-AE','18-AE','19-AE'],rotation='vertical')

#x.set_title(' Q-Value Analysis on 753 Repository DataSet')
ax.set_xlabel('Model Type (Louvain or X-Class AutoEncoder)',fontsize = 50)
ax.set_ylabel('Mod Q-Score',fontsize = 50)


plt.show()