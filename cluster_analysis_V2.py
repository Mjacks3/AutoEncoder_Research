import os
import  matplotlib.pyplot as plt
import matplotlib.ticker as plticker

#source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain = []
autoencoder = []


data = [louvain, autoencoder]



for r, d, f in os.walk(source_dir):
    qvalue_file = r+"/"+r.split("/")[-1] +".qvalues"

    if os.path.exists(r+"/"+r.split("/")[-1] +".isolated_louvain"):
        file_cluster_count = set([])
        with open(r+"/"+r.split("/")[-1] +".louvain") as fi:
            all_lines = fi.read().splitlines()
            for  line in all_lines:
                file_cluster_count.add(line.split()[-1])

        louvain.append(len(file_cluster_count))


        highest_ae_qvalue_line = "0 -9999"
        all_lines = []
        with open(qvalue_file) as fi:
                all_lines = fi.read().splitlines()
        for line in all_lines:
            splt_line = line.split()

            if splt_line[0] !="louvain":
                if float(splt_line[1]) >= float(highest_ae_qvalue_line.split()[1]):
                    highest_ae_qvalue_line = line


        #Take the Highest Q-Valued cluster and analyze its cluster
        highest_qvalue_file = r+"/"+r.split("/")[-1] +"_"+ highest_ae_qvalue_line.split()[0] +".clustering"

        file_cluster_count = set([])
        with open(highest_qvalue_file) as fi:
            all_lines = fi.read().splitlines()
            for  line in all_lines:
                file_cluster_count.add(line.split()[-1])
        autoencoder.append(len(file_cluster_count))




fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(data)
plt.xticks([1,2], ['Louvain Clusters', 'Autoencoder Clusters'])

ax.set_title(' Louvain vs Combined Autoencoder Cluster Analysis V2 ~ 750 Repository Dataset')
ax.set_xlabel('Method')
ax.set_ylabel('Number of Clusters Generated')
plt.ylim(0, 20)
plt.show()