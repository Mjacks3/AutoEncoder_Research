import os
import statistics
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
#source_dir = "experiment/short"
#source_dir = "experiment/test/CoolViewPager"
source_dir = "experiment/test"
modq_values  = []
avg_mod_q_values = []
#models = 5, 10


font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 35}

matplotlib.rc('font', **font)

five = []
ten = []
x = []

for subset in range(10,510,10):
    five.append([])
    ten.append([])
    x.append(subset)


count = 0 
for r, d, f in os.walk(source_dir):
    count+= 1
    print("Progress: "+ str(count/751.0))

    basename = r.split("/")[-1]

    if basename+".txt" in f:
        edge_list = r+'/'+basename+".txt"

        m_norm = 0
        with open(edge_list ) as f:
            for edge in f:
                m_norm += int(edge.split()[-1])
    
        for model in ["5","10"]:
            for subset in range(10,510,10):
            

                clusterfile = basename+"_"+model+"-"+str(subset)+".clustering"

                with open(r+"/"+clusterfile) as fi:
                    clusters = fi.read().splitlines()

                node_partition_mapping = {}
                partion_node_counting = {}

                for pair in clusters:
                    node_partion = pair.split()
                    #print(node_partion)
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
                #print("m_odQ!!!!!!!!!!!!!!!!!!!!!")
                #print(modQ)

                if model == "5":
                    five[int(subset/10) -1].append(modQ)
                else:
                    ten[int(subset/10) -1].append(modQ)

#print(five)
for arr,arr2, ind in zip( five,ten, range(len(five))):
    #print(ind)
    #print (arr)

    five[ind] = statistics.mean(arr)
    ten[ind] = statistics.mean(arr2)


#print(five)



#plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(ten))
print(y_pos)


#ax.set_title('AutoEncoder Training Data Vs Q-Value on 753 Test Dataset ')
ax.set_xlabel('Train Dataset Size',fontsize = 55)
ax.set_ylabel('Mean Q-Score',fontsize = 55)

plt.plot(x, five, linestyle='-', marker='v',color='b',linewidth=3,markersize=17)
plt.plot(x, ten, linestyle='-', marker='s',color='g',linewidth=3,markersize=17)
ax.set_xlim(left=100,right =450)
ax.legend(['5 Class Model', '10 Class Model'])

plt.show()


