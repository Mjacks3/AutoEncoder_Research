import os, sys, argparse,json, time, subprocess

edgelist_cnt = 0
embedding_cnt = 0

for r, d, f in os.walk("experiment/test"):

    if len(f) == 2:
         edgelist_cnt +=1
    elif len(f) == 3:
        print(f)
        edgelist_cnt +=1
        embedding_cnt += 1

print(edgelist_cnt)
print(embedding_cnt)

