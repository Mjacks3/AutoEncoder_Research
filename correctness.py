import matplotlib.pyplot as plt
import os

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
#source_dir = "experiment/test/WeatherIconView"
#"../"+folder+"/"+"AutoEncoder_Research/experiment/test")

source_dir = "yeet"



folders = ["AE_I_2","AE_L","AE_4","AE_5","AE_6"]

clusters = []
for folder in folders:
    for r, d, f in os.walk("../"+folder+"/"+"AutoEncoder_Research/experiment/test/websphere"):
        print(r)
        if r.split("/")[-1]+".embedding" in f:
    
            for a_file in f:
                if  ".clustering" in a_file :
                    print(a_file)

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

                    #print(partion_node_counting.values())
                    #print(partion_node_counting.keys())
                    #print(partion_node_counting.items())
                    
                    labels = [] 
                    sizes = []
                    total = sum(partion_node_counting.values())


                    for partition in partion_node_counting.keys():
                        label = "[" 
                        cnt = 0 
                        for node in node_partition_mapping.keys():
                            if str(node_partition_mapping[node]) == str(partition):
                                label += node.split(".")[-1] + " "
                                cnt +=1

                            if cnt % 5 == 0 and cnt != 0:
                                cnt+= 1
                                label += "\n"
                            
                        label +="]"

                        labels.append(label)
                        percent = 100* (partion_node_counting[partition] /float(total))
                        sizes.append(percent)
                    

                    #print(sizes)


                    



                    #sizes = [15, 30, 45, 10]
                    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
                    
                    fig1, ax1 = plt.subplots()
                    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                            shadow=True, startangle=90)
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



                    folders = ["AE_I_2","AE_L","AE_4","AE_5","AE_6"]

                    name = a_file[0:-10]
                    if folder == "AE_I_2":
                        name+= "_100TD"

                    elif folder == "AE_L":
                        name+= "200TD"

                    elif folder == "AE_4":
                        name+= "300TD"

                    elif folder == "AE_5":
                        name+= "400TD"

                    elif folder == "AE_6":
                        name+= "500TD"

                    plt.savefig("../Downloads/data/correctness/websphere/"+name+".png")
                    #plt.show()
                    