# plot correlation matrix
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (15,8)
plt.rcParams['axes.titlesize'] = 'large'

data = pd.read_csv('data/abalone.csv', index_col="Unnamed: 0")
data.head(5)

plt.style.use('fivethirtyeight')
sns.set_style("white")

# Compute the correlation matrix 相关性矩阵
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
data, ax = plt.subplots(figsize=(8, 8))
plt.title('The Correlation Between Features of Abalone and Rings')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# Draw the heat map with the mask and correct aspect ratio
sns.heatmap(corr, vmax=1.2, square='square', cmap=cmap, mask=mask, ax=ax, annot=True, fmt='.2g', linewidths=2)
plt.show()





