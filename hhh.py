import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = [1, 2, 3, 4, 1, 2, 3, 4]
y = [0.1, 0.5, 0.3, 0.7, 0.324, 0.534, 0.123, 0.687]
label = [1, 1, 1, 1, 2, 2, 2, 2]

sns.set(style="ticks", rc={"lines.linewidth": 2, "legend.fontsize": 22})
data = pd.DataFrame(dict(x = x, y = y, label = label))
g = sns.relplot(x = 'x', y = 'y', hue = 'label', kind = 'line', data = data)
plt.show()