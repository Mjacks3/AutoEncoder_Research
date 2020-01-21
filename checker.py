import os, sys, argparse,json, time, subprocess

models = [5]
for num_clusters in  models:
    for sub_epochs in range(0,510,10):

        in_progress = 0
        for r, d, f in os.walk("experiment/test"):

            for a_file in f:
                if  "_" + str(num_clusters) + "-"+ str(sub_epochs)+ ".clustering" in a_file:
                    in_progress += 1
                
        print(str(num_clusters) +"-"+ str(sub_epochs)+ " in_progress: "+ str( in_progress))
print("full: 751")
