import numpy as np
import os
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#source_dir = "experiment/test"
source_dir = "../../coba_test/coba/experiment/final_test"
source_dir_2 = "../../coba_test/coba/experiment/balanced_test"
data_1 = []
data_2 = []

for r, d, f in os.walk(source_dir):

    basename = r.split("/")[-1]
    if basename+".txt" in f:

        with open(source_dir+"/"+basename+"/"+basename+".txt") as f:
            temp = f.readlines()
            data_1.append(len(temp) )

for ri, di, f2 in os.walk(source_dir_2):

    basename = ri.split("/")[-1]
    if basename+".txt" in f2:

        with open(source_dir_2+"/"+basename+"/"+basename+".txt") as fi:
            temp = fi.readlines()
            data_2.append(len(temp) )


font = { 'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

fig, ax = plt.subplots()

plt.hist(data_2, alpha=0.5,bins=100,range=(0,750),label='V2 Test Data Distribution') 
plt.hist(data_1, alpha=0.5,bins=100,range=(0,750),label='V1 Test Data Distribution') 
plt.legend(loc='upper right')

#plt.title("Size Distribution of 753 Repoitiories V2 ")
ax.set_xlabel('Size of Edge List (# Lines)',fontsize = 55)
ax.set_ylabel('Occurences in Test Set',fontsize = 55)
#ax.set_ylim(bottom=0,top=500)
ax.set_xlim(bottom=100,top=400)
plt.show()
