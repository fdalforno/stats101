import matplotlib.pyplot as plt
import numpy as np

# create some randomly ddistributed data:
data = np.random.randn(10000)

data_sorted = np.sort(data)
unique, counts = np.unique(data_sorted, return_counts=True)
pmf = counts / len(data_sorted)
cdf = np.cumsum(pmf)
result = np.column_stack((unique, pmf,cdf)) 

print(result)