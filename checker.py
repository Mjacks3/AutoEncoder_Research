import os, sys, argparse,json, time, subprocess
"""
for num_clusters in range (2,20):
    in_progress = 0
    full = 0
    for r, d, f in os.walk("experiment/balanced_test"):

        for a_file in f:
            if  "_" + str(num_clusters) + ".clustering" in a_file:
                in_progress += 1

        full += 1
            
    print(str(num_clusters) + " in_progress: "+ str( in_progress))
    print("full "+ str( full))

"""
print("helo")
#for r, d, f in os.walk("../../coba_test/coba/experiment/balanced_test2/"): # for each file 
#for r, d, f in os.walk("experiment/balanced_test"): # for each file 
for r, d, f in os.walk("../../balanced_test/"): # for each file 
    #print (r)
    if len(f) < 4  : 
        print( r)
