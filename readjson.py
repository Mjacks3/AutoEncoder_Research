


clusters = []
with open("DayTrader.json") as fi:
    clusters = fi.read().splitlines()


for clus, ind in zip(clusters , range(len(clusters))):
    if "group\"" in clus:
        print(clusters[ind -1])
        print(clusters[ind])
