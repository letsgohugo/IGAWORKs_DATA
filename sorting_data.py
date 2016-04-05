# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:54:24 2016

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:15:32 2016

@author: user
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()
"""
y = np.loadtxt("/home/andychoi/datatest/data1.txt")#순위 데이터
y_data2 = np.loadtxt("/home/andychoi/datatest/data2.txt")#판매량 데이터
y_data2 = np.transpose(y_data2)

y =np.transpose(y)

y_solve_x = np.array([])
y_solve_y = np.array([])
y_solve_z = np.array([])

x_3d = np.array([])
y_3d = np.array([])
z_3d = np.array([])

k = 0

for i in y_data2[2] :
    if i !=0 :
        y_solve_z = np.append(y_solve_z,i)
        y_solve_y = np.append(y_solve_y,y_data2[1][k])
        y_solve_x = np.append(y_solve_x,y_data2[0][k])
    k = k+1
"""
month1 = y[0:16740]
month2 = y[16741:16741+15120]
month3 = y[16741+15120:16741+15120+16740]
month4 = y[16740*2+15120+1:16740*2+15120+1+16200]
month5 = y[16740*2+15120+1+16200:16740*2+15120+1+16200+16620]
month6 = y[16740*2+15120+1+16200+16620:]

month = np.transpose(month1)
"""
"""
j = np.unique(y_solve_x)
for i  in j :
    for plot in y :
        if i == plot[0] :
            x_3d = np.append(x_3d,plot[0])
            y_3d = np.append(y_3d,plot[1])
            z_3d = np.append(z_3d,plot[2])
"""
cnt = 0

day_test = np.array([])
for month in range(7) :
    if month == 1 :
        for day in range(31) :            
            day_test = np.append(day_test,20150000+100+day+1)
    if month == 2 :
        for day in range(28) :            
            day_test = np.append(day_test,20150000+200+day+1)
    if month == 3 :
        for day in range(31) :            
            day_test = np.append(day_test,20150000+300+day+1)
    if month == 4 :
        for day in range(30) :            
            day_test = np.append(day_test,20150000+400+day+1)
    if month == 5 :
        for day in range(31) :            
            day_test = np.append(day_test,20150000+500+day+1)
    if month == 6 :
        for day in range(31) :            
            day_test = np.append(day_test,20150000+600+day+1)
#날짜 만들기
"""
for l in day_test :
    cnt = 0
    x_sort = np.array([])
    y_sort = np.array([])
    z_sort = np.array([])
    
    for o in y_solve_y :
        
        cnt_m = 0
        for m in y_3d :
            if ((l == m) & (l==o) & (y_solve_x[cnt] == x_3d[cnt_m])) :
                x_sort = np.append(x_sort,int(z_3d[cnt_m]))
                y_sort = np.append(y_sort,int(y_solve_z[cnt]))
                z_sort = np.append(z_sort,int(y_solve_x[cnt]))
            cnt_m = cnt_m+1
        cnt = cnt+1
    file = np.array([x_sort,y_sort,z_sort])
    file = np.transpose(file)
    np.savetxt('/home/andychoi/datatest/'+str(l)+'fitted.txt', file, delimiter='\t', newline='\r\n')
    plt.plot(x_sort,y_sort)
    #날짜별 데이터 총 151개의 순위 별 판매량 데이터(판매하는 도서 이름과 해당 날짜가 같으면 순위 판매량을 이어주고, 같지 않거나 판매량 데이
    #없으면 버린다.) 만들기,
"""
index = np.array([])
index1 =np.array([])
index2 = np.array([])
for i in range(2861) :
    k=0
    for j in y[0]:
        if i ==j :
            index = np.append(index,j)
            index1 = np.append(index1,y[1][k])
            index2 = np.append(index2,y[2][k])
        k=k+1
y = np.array([])
count = np.array([])
flag = 0
outer =0
sigma =0
out_sigma=0
for i in range(97619) :
    if index2[i]<10 :
        flag = 1
        outer =1
        sigma = sigma+1
    if flag == 1 :
        if index2[i]>50 :
            if outer ==1 :
                out_sigma = out_sigma+1
                
            outer =0
            
        y = np.append(y,index2[i])
        count = count+1
    if index[i] !=index[i+1] :
        plt.plot(np.linspace(0,y.size,y.size),y)
        y=np.array([])
        flag =0
            

"""fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax.scatter(y_solve_x,y_solve_y,y_solve_z,c = 'b')
ax.scatter(x_3d,y_3d,z_3d,c='r')

#ax.scatter(month[0],month[1],month[2])
"""
k = np.array([index])
k = np.append(k,[index1],axis=0)
k = np.append(k,[index2],axis=0)

plt.show()
np.savetxt('/home/andychoi/datatest/sortting_byname&date.txt', k, delimiter='\t', newline='\r\n')
