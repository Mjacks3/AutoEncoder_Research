import os
import  matplotlib.pyplot as plt

source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain =  []
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

#clust_2 = {}
#Further Analysis might be number and size of microservice generated

#further analysis should be louvain vs highest Q_Value reccomendation

count = 0
for r, d, f in os.walk(source_dir):
    count+= 1
    print("Progress: "+ str(count/(753.0 *1)))
    for a_file in f:
        if ".isolated_louvain" in a_file or ".clustering" in a_file:
            file_cluster_count = set([])
            with open(r+'/'+ a_file) as fi:
                all_lines = fi.read().splitlines()
                for  line in all_lines:
                    file_cluster_count.add(line.split()[-1])

            if ".isolated_louvain" in a_file:
                louvain.append(len(file_cluster_count))

            elif "_2.clustering" in a_file:
                clus_2.append(len(file_cluster_count))

            elif "_3.clustering" in a_file:
                clus_3.append(len(file_cluster_count))

            elif "_4.clustering" in a_file:
                clus_4.append(len(file_cluster_count))

            elif "_5.clustering" in a_file:
                clus_5.append(len(file_cluster_count))

            elif "_6.clustering" in a_file:
                clus_6.append(len(file_cluster_count))

            elif "_7.clustering" in a_file:
                clus_7.append(len(file_cluster_count))

            elif "_8.clustering" in a_file:
                clus_8.append(len(file_cluster_count))

            elif "_9.clustering" in a_file:
                clus_9.append(len(file_cluster_count))

            elif "10.clustering" in a_file:
                clus_10.append(len(file_cluster_count))

            elif "11.clustering" in a_file:
                clus_11.append(len(file_cluster_count))

            elif "12.clustering" in a_file:
                clus_12.append(len(file_cluster_count))

            elif "13.clustering" in a_file:
                clus_13.append(len(file_cluster_count))

            elif "14.clustering" in a_file:
                clus_14.append(len(file_cluster_count))

            elif "15.clustering" in a_file:
                clus_15.append(len(file_cluster_count))

            elif "16.clustering" in a_file:
                clus_16.append(len(file_cluster_count))

            elif "17.clustering" in a_file:
                clus_17.append(len(file_cluster_count))

            elif "18.clustering" in a_file:
                clus_18.append(len(file_cluster_count))

            elif "19.clustering" in a_file:
                clus_19.append(len(file_cluster_count))


#print(clus_3)
#print(full_set)


fig = plt.figure()
#fig.suptitle('Microservice Cluster Analysis for  750 Repository DataSet', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(full_set)
plt.xticks([1,2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['Louvain','2-Cluster', '3-Cluster ', '4-Cluster','5-Cluster','6-Cluster','7-Cluster','8-Cluster ','9-Cluster','10-Cluster','11-Cluster','12-Cluster','13-Cluster','14-Cluster','15-Cluster','16-Cluster','17-Cluster','18-Cluster','19-Cluster'], rotation='vertical')
ax.set_title('Microservice Cluster Analysis  for  750 Repository DataSet')
ax.set_xlabel('Generation Method')
ax.set_ylabel('Number of Clusters Generated')
plt.ylim(0, 10) 

plt.show()

