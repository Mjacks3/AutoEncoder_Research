import os
import  matplotlib.pyplot as plt

source_dir = "experiment/test/WeatherIconView"
source_dir = "experiment/test"

#Analyis meaning number of microservices generated
louvain = []
autoencoder = []


data = [louvain, autoencoder]



coupling = [] 

 
for r, d, f in os.walk(source_dir):
    qvalue_file = r+"/"+r.split("/")[-1] +".qvalues"

    if  os.path.exists(qvalue_file):
        highest_ae_qvalue_line = "0 -9999"

        all_lines = []
        with open(qvalue_file) as fi:
                all_lines = fi.read().splitlines()
        for line in all_lines:
            splt_line = line.split()

            if splt_line[0] !="louvain":
                if float(splt_line[1]) > float(highest_ae_qvalue_line.split()[1]):
                    highest_ae_qvalue_line =  line


        edge_list_file = r+"/"+r.split("/")[-1] +".txt"
        edge_list = []
        with open(edge_list_file) as fi:
            edge_list = fi.read().splitlines()


        #louvain coup
        with open(r+"/"+r.split("/")[-1] +".isolated_louvain") as fi:
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

        score = 0

        for parition_a in partion_node_counting.keys():
            for parition_b in partion_node_counting.keys():
                cnt  = 0.00
                for edge in edge_list:
                    if not(parition_a != parition_b) and ( (node_partition_mapping[edge.split()[0]] == parition_a  and node_partition_mapping[edge.split()[1]] == parition_b) or (node_partition_mapping[edge.split()[0]] == parition_b  and node_partition_mapping[edge.split()[1]] == parition_a)):
                        cnt += 1

                if int (2 * partion_node_counting[parition_a] * partion_node_counting[parition_b] ) != 0:
                    score = cnt/ float( 2 * partion_node_counting[parition_a] * partion_node_counting[parition_b]   )
                
                louvain.append(score)



        #High score ae coup
        highest_qvalue_file = r+"/"+r.split("/")[-1] +"_"+ highest_ae_qvalue_line.split()[0] +".clustering"

        with open(highest_qvalue_file) as fi:
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

        score = 0

        for parition_a in partion_node_counting.keys():
            for parition_b in partion_node_counting.keys():
                cnt  = 0.00
                for edge in edge_list:
                    if not(parition_a != parition_b) and ( (node_partition_mapping[edge.split()[0]] == parition_a  and node_partition_mapping[edge.split()[1]] == parition_b) or (node_partition_mapping[edge.split()[0]] == parition_b  and node_partition_mapping[edge.split()[1]] == parition_a)):
                        cnt += 1

                if int (2 * partion_node_counting[parition_a] * partion_node_counting[parition_b] ) != 0:
                    score = cnt/ float( 2 * partion_node_counting[parition_a] * partion_node_counting[parition_b]   )
                autoencoder.append(score)
        

fig = plt.figure()

ax = fig.add_subplot(111)
ax.boxplot(data[0:])
plt.xticks([1,2], ['Louvain Clusters', 'AutoEncoder Clusters'])

ax.set_title('AutoEncoder Deoupling Analysis ~ 750 Repository Dataset')
ax.set_xlabel('Generation Method')
ax.set_ylabel('Coupling Value')
#plt.ylim(0, 20) 

plt.show()

