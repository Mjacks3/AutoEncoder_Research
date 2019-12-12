import os, sys, argparse,json, time, subprocess


for num_clusters in range (2,20):
    in_progress = 0
    full = 0
    for r, d, f in os.walk("experiment/test"):

        for a_file in f:
            if  str(num_clusters) + ".clustering" in a_file:
                in_progress += 1

        full += 1
            
    print(str(num_clusters) + " in_progress: "+ str( in_progress))
    print("full "+ str( full))