
import numpy as np
import pandas as pd
import csv
'''
data_x = pd.read_table('F:/keyan/train.txt',encoding='gb2312')
#data_x = pd.read_table("F:/keyan/train.txt")#pd.dataframe
#train_data = np.array(data_x)#np.ndarray()
train_x_list = data_x.values
#train_x_list=train_data.tolist()
f = open('F:/keyan/train2.csv','w',newline='')
writer=csv.writer(f)
#print(train_x_list)
for line in train_x_list:
    for x in range(0,len(line),19):
        writer.writerow(line[x:x+19]+'\n')
        #for i in line[x:x+19]:
            #writer.writerow(line[x:x+19]) 
        #print(str1)
f.close()
'''
filename = 'array_reflection_2D_TM_vertical_normE_center.txt'
# txt文件和当前脚本在同一目录下，所以不用写具体路径
pos = []
#Efield = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.read().splitlines()
        # 整行读取数据
        if not lines:
            break
            pass
        for ai in lines:
            
            p_tmp = [float(i) for i in ai.split()]
            # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            pos.append(p_tmp) # 添加新读取的数据
            #Efield.append(E_tmp)
            pass
    pos = np.array(pos)# 将数据从list类型转换为array类型。
    #Efield = np.array(Efield)
    pass
f = open('F:/keyan/train2.csv','w',newline='')
writer=csv.writer(f)
for line in pos:
    for x in range(0,len(line),19):
        writer.writerow(line[x:x+19]+'\n')
        #for i in line[x:x+19]:
            #writer.writerow(line[x:x+19]) 
        #print(str1)
f.close()

