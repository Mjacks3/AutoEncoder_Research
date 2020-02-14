import numpy as np
import os
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#source_dir = "experiment/test"
#source_dir = "../../coba_test/coba/experiment/final_test"
source_dir = "../../coba_test/coba/experiment/balanced_test"
data_size = []
small = 0
med = 0
large = 0
for r, d, f in os.walk(source_dir):

    basename = r.split("/")[-1]
    if basename+".txt" in f:

        with open(source_dir+"/"+basename+"/"+basename+".txt") as f:
            temp = f.readlines()
            data_size.append(len(temp) )
            if len(temp) < 30:
                small += 1
            elif 30< len(temp) < 75:
                med += 1
            else:
                large+= 1

print("files less than 30: " + str(small))
print("files  30-75: " + str(med))
print("files >75: " + str(large))


font = { 'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

fig, ax = plt.subplots()

#plt.hist(data_size, bins='auto',range=(0,750),label='V1 Distribution') 
plt.hist(data_size, bins='auto',range=(0,750),label='V2 Distribution') 

#plt.title("Size Distribution of 753 Repoitiories V2 ")
ax.set_xlabel('Size of Edge List (# Lines)',fontsize = 55)
ax.set_ylabel('Occurences in Test Set',fontsize = 55)
#ax.set_ylim(bottom=0,top=500)
plt.show()
