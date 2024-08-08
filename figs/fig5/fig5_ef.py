import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

left = 1
right = 1
c1_co_data = np.array(pd.read_csv("./condition_uc1_co.csv"))[:, left+1:-right]
c1_ce_data = np.array(pd.read_csv("./condition_uc1_ce.csv"))[:, left+1:-right]
c2_co_data = np.array(pd.read_csv("./condition_uc2_co.csv"))[:, left+1:-right]
c2_ce_data = np.array(pd.read_csv("./condition_uc2_ce.csv"))[:, left+1:-right]


c1_co_data2 = np.array(pd.read_csv("./condition_uc1_co_iv.csv"))[:, left+1:-right]
c1_ce_data2 = np.array(pd.read_csv("./condition_uc1_ce_iv.csv"))[:, left+1:-right]
c2_co_data2 = np.array(pd.read_csv("./condition_uc2_co_iv.csv"))[:, left+1:-right]
c2_ce_data2 = np.array(pd.read_csv("./condition_uc2_ce_iv.csv"))[:, left+1:-right]


fig, axes = plt.subplots(1, 2)
axes[0].plot(c1_co_data2[0], c1_co_data2[1], label="faulted corner")
axes[0].plot(c1_ce_data2[0], c1_ce_data2[1], label="faulted center")
axes[0].plot(c2_co_data2[0], c2_co_data2[1], label="unfaulted corner")
axes[0].plot(c2_ce_data2[0], c2_ce_data2[1], label="unfaulted center")



k = 3
s = 5
spl_sts1 = UnivariateSpline(c1_co_data[0], c1_co_data[1], k=k, s=s)
spl_sts2 = UnivariateSpline(c1_ce_data[0], c1_ce_data[1], k=k, s=s)
spl_sts3 = UnivariateSpline(c2_co_data[0], c2_co_data[1], k=k, s=s)
spl_sts4 = UnivariateSpline(c2_ce_data[0], c2_ce_data[1], k=k, s=s)

x = np.linspace(c1_co_data[0][50], c1_co_data[0][-50], 1000)




axes[1].plot(x, spl_sts1(x), label="faulted corner")
axes[1].plot(x, spl_sts2(x), label="faulted center")
axes[1].plot(x, spl_sts3(x), label="unfaulted corner")
axes[1].plot(x, spl_sts4(x), label="unfaulted center")




axes[0].legend()
axes[1].legend()
plt.show()



