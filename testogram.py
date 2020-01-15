import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

y = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
count  = 10
x = []

for a in y:
    x.append(count)
    count += 10


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(x))
ax.set_xticks(y_pos)


plt.bar(x,y,color='blue')
plt.show()