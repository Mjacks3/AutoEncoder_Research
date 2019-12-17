import os
import  matplotlib.pyplot as plt

#source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain = []
autoencoder = []


data = [louvain, autoencoder]



for r, d, f in os.walk(source_dir):
    qvalue_file = r+"/"+r.split("/")[-1] +".qvalues"

    if  os.path.exists(qvalue_file):
        highest_ae_qvalue =  -9999
        all_lines = []
        with open(qvalue_file) as fi:
                all_lines = fi.read().splitlines()
        for line in all_lines:
            splt_line = line.split()

            if splt_line[0] =="louvain":
                louvain.append(float(splt_line[1]))
            else:
                if float(splt_line[1]) > highest_ae_qvalue:
                    highest_ae_qvalue =  float(splt_line[1]) 

        autoencoder.append(highest_ae_qvalue)
        
        



fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(data)
plt.xticks([1,2], ['Louvain Clusters', 'AutoEncoder Clusters'])

ax.set_title(' Louvain vs Combined AutoEncoder Q-Value Analysis ~ 750 Repository Dataset')
ax.set_xlabel('Method')
ax.set_ylabel('Q-Value')

plt.show()



