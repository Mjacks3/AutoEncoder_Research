import os
import  matplotlib.pyplot as plt
import statistics
import matplotlib
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 35}

matplotlib.rc('font', **font)

source_dir = "experiment/test/WeatherIconView"
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
count = 0

for r, d, f in os.walk(source_dir):
  count+= 1
  print("Progress: "+ str(count/753.0))
  if r.split("/")[-1]+".embedding" in f:
    edge_list_file = r+"/"+r.split("/")[-1] +".txt"
    edge_list = []

    with open(edge_list_file) as fi:
        edge_list = fi.read().splitlines()
    for a_file in f:
        if "isolated_louvain" in a_file or "clustering" in a_file:
            
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

                for parition_a in partion_node_counting.keys():
                    for parition_b in partion_node_counting.keys():
                        cnt  = 0.00
                        for edge in edge_list:
                            if not(parition_a != parition_b) and ( (node_partition_mapping[edge.split()[0]] == parition_a  and node_partition_mapping[edge.split()[1]] == parition_b) or (node_partition_mapping[edge.split()[0]] == parition_b  and node_partition_mapping[edge.split()[1]] == parition_a)):
                                cnt += 1

                        if int (2 * partion_node_counting[parition_a] * partion_node_counting[parition_b] ) != 0:
                            score = cnt/ float( 2 * partion_node_counting[parition_a] * partion_node_counting[parition_b]   )



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

        
print(full_set)

fig = plt.figure()

full_set_minus_zeros = []
for item in full_set:
    full_set_minus_zeros.append([i for i in item if float(i) != 0.0])


ax = fig.add_subplot(111)
bp = ax.boxplot(full_set[0:])

for box in bp['boxes']:
    box.set(linewidth=10)

plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [ 'Louvain', '2-AE', '3-AE ', '4-AE','5-AE','6-AE','7-AE','8-AE ','9-AE','10-AE','11-AE','12-AE','13-AE','14-AE','15-AE','16-AE','17-AE','18-AE','19-AE'],rotation='vertical')

#ax.set_title('Inter-Cluster Coupling Analysis With Dropped Zeroes ~ 753 Repository Dataset')

ax.set_xlabel('Model Type (Louvain or X-Class AutoEncoder)',fontsize = 50)
ax.set_ylabel('Coupling Value',fontsize = 50)
plt.ylim(-.01, .3) 
for a in full_set:
    print(statistics.mean(a))
plt.show()

