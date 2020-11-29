import numpy as np
import pandas as pd
import csv

df = pd.read_csv("路径",delimiter=",")

#第一种
#ar = np.array.(df)
#m = ar.tolist()

#第二种
m = df.values.tolist()

#第三种
#m = df.tolist()

new = []
for i in range(0,len(m),3):
    a = []
    b = []
    a = m(i).extend(m(i+1))
    b = a.extend(m(i+2))
    new.append(b)
pos = np.array(new)
f = open('新的csv路径','w',newline='')
writer=csv.writer(f)
for line in pos:
    writer.writerow(line)
f.close()
