import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import  matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)

uniform_data = np.random.rand(10, 12)
uniform_data[0] = [0,0,0,0,0,0,0,0,0,0,0,0]
ax = sns.heatmap(uniform_data,cbar_kws={'label': 'colorbar title'})
#print(uniform_data[0]))
plt.show()



#2c-100,5c100,10c100,2c


#x axis = model type 
#y axis = mode training dat size
#z = qvalue

#xticklabels, yticklabels : “auto”, bool, list-like, or int, optional

#If True, plot the column names of the dataframe. If False, don’t plot the column names. 
#If list-like, plot these alternate labels as the xticklabels. If an integer, use the column names but plot only every n label. If “auto”, try to densely plot non-overlapping labels.