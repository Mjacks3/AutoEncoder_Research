import matplotlib.pyplot as plt

#

#Calculate Metrics
cluster_pairs='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt.clusters'
edge_list='/home/mjacks3/monize/tahdith/datasets/train/Java.git/Java.git.txt'


m_norm = 0

node_partition_mapping = {}
#edgelist = {}
#partion_node_mapping = {} Unused for now.
partion_node_counting = {}

with open(cluster_pairs) as f:
    for pair in f:
        #print (pair)
        node_partion = pair.split()
        node_partition_mapping[node_partion[0]] = node_partion[1]

#print(node_partition_mapping)

with open(edge_list) as f:
    for edge in f:
        m_norm += 1
        """
        source_dest = edge.split()

        if  source_dest[0] in edgelist:
            edgelist[source_dest[0]].append(source_dest[1] )
        else:
            edgelist[source_dest[0]] =  [source_dest[1]]
        """


for a in set(node_partition_mapping.values()):
    partion_node_counting[a] = 0
    for b in node_partition_mapping.items():
        if str(b[1]) == str(a) :
            partion_node_counting[a] += 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


modQ = 0 

for partition in partion_node_counting.keys():
    internal = 0 
    incident = 0
    with open(edge_list) as f:
        for edge in f:
            source_dest = edge.split()

            if source_dest[0] not in node_partition_mapping:
                #print("unmapped node: " , source_dest[0])
                pass #A bug if hit;  each node should be mapped

            elif source_dest[1] not in node_partition_mapping:
                #print("unmapped node: " , source_dest[1])
                pass #A bug if hit;  each node should be mapped
                pass


            elif node_partition_mapping[source_dest[0]] == str(partition) and \
            node_partition_mapping[source_dest[1]] == str(partition):
                internal += 1
                incident += 1

            elif node_partition_mapping[source_dest[0]] == str(partition) or \
            node_partition_mapping[source_dest[1]] == str(partition):

                incident += 1

    modQ += ( (internal/(2.0*m_norm)) -   (incident/(2.0*m_norm))**2)

print("Modularity Q")
print(modQ)



for partition_a in partion_node_counting.keys():
    for partition_b in partion_node_counting.keys():
        if partition_a == partition_b:

            
            actual_edges =  0 
            possible_edges =   (partion_node_counting[partition_a] - 1) * partion_node_counting[partition_b]

            with open(edge_list) as f:
                    for edge in f:
                        source_dest = edge.split()

                        if source_dest[0] not in node_partition_mapping:
                            #print("unmapped node: " , source_dest[0])
                            pass #A bug if hit;  each node should be mapped

                        elif source_dest[1] not in node_partition_mapping:
                            #print("unmapped node: " , source_dest[1])
                            pass #A bug if hit;  each node should be mapped


                        elif (str(node_partition_mapping[source_dest[0]]) == partition_a and \
                            str(node_partition_mapping[source_dest[1]]) == partition_b):

                            actual_edges += 1
            print(partition_a + " to " + partition_b)
            print(str(actual_edges)+ "/" + str(possible_edges))
            #print()



            

        else:
            #print(partition_a+" "  partition_b " ", end="")
            actual_edges =  0 
            possible_edges =  2* partion_node_counting[partition_a] * partion_node_counting[partition_b]

            with open(edge_list) as f:
                    for edge in f:
                        source_dest = edge.split()

                        if source_dest[0] not in node_partition_mapping:
                            #print("unmapped node: " , source_dest[0])
                            pass #A bug if hit;  each node should be mapped

                        elif source_dest[1] not in node_partition_mapping:
                            #print("unmapped node: " , source_dest[1])
                            pass #A bug if hit;  each node should be mapped


                        elif (str(node_partition_mapping[source_dest[0]]) == partition_a and \
                            str(node_partition_mapping[source_dest[1]]) == partition_b)  or \
                            (str(node_partition_mapping[source_dest[0]]) == partition_b and \
                             str(node_partition_mapping[source_dest[1]]) == partition_a):

                            actual_edges += 1
            print(partition_a + " to " + partition_b)
            print(str(actual_edges)+ "/" + str(possible_edges))
            print()
            
print(partion_node_counting)  

plt.plot([10], [modQ], 'ro')
plt.show()