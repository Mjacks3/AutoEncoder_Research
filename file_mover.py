import numpy as np
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import shutil


source_dir = "../../coba_test/coba/experiment/test"
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
            if len(temp) < 30 and small <= 250:
                small += 1
                shutil.copytree("../../coba_test/coba/experiment/test/"+basename, "../../coba_test/coba/experiment/balanced_test/"+basename)  
            elif 30 < len(temp) and len(temp) < 75 and med <= 250:
                med += 1
                shutil.copytree("../../coba_test/coba/experiment/test/"+basename, "../../coba_test/coba/experiment/balanced_test/"+basename)  
            elif 75 < len(temp)  and   large <= 250:
                large+= 1
                shutil.copytree("../../coba_test/coba/experiment/test/"+basename, "../../coba_test/coba/experiment/balanced_test/"+basename)  

print("files less than 30: " + str(small))
print("files  30-75: " + str(med))
print("files >75: " + str(large))



  
