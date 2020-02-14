import os
import statistics
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import  matplotlib.pyplot as plt

source_dir = "experiment/test/short"
#source_dir = "experiment/test/WeatherIconView"
#source_dir = "experiment/test"
modq_values  = []
avg_mod_q_values = []
model = "10"

for subset in range(10,510,10):
    modq_values.append([])



count = 0
for r, d, f in os.walk(source_dir):
    count+= 1
    print("Progress: "+ str(count/751.0))
    m_norm = 0

    basename = r.split("/")[-1]

    #Get Edgelist and Calculate M_Norm Value  
    if basename+".txt" in f:
        edge_list = r+'/'+basename+".txt"

        m_norm = 0
        with open(edge_list ) as f:
            for edge in f:
                m_norm += int(edge.split()[-1])

    for subset in range(10,510,10):
        if basename != 'test':

            clusterfile = basename+"_"+model+"-"+str(subset)+".clustering"

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
            print("m_odQ!!!!!!!!!!!!!!!!!!!!!")
            print(modQ)
            print()

            #print("Modularity Q Value")
            #print(modQ)
            #output this qvalue to file
            #print(int(subset/10))

            modq_values[int(subset/10) -1].append(modQ)

for array_of_qvalues in modq_values:
    avg_mod_q_values.append(statistics.mean(array_of_qvalues))

        

#avg_mod_q_values = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
count  = 10
x = []

for a in avg_mod_q_values:
    x.append(count)
    count += 10


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(avg_mod_q_values))
#ax.set_xticks(y_pos)

ax.set_title('AutoEncoder (10 Grouping Model)  Epochs Vs Q-Value on 751 Test Dataset ')
ax.set_xlabel('# Epochs Trained Per Model')
ax.set_ylabel('Q-Value')

plt.bar(x,avg_mod_q_values,color='blue')
plt.show()


