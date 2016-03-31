# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:37:18 2016

@author: andychoi
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.mlab as mlab
from scipy import stats


def lin(x,p) :
    y = p[0]*x+p[1]
    return y

def liner(p,x,z) :
    return lin(x,p)-z

def poli(x,p) :
    y=0
    for i in range(10) :
        y = y + p[i]*pow(x,i)
    return y
    
def errorfunc(p,x,z) :
    return poli(x,p)-z

def lorentzian(x,p):
    numerator =  (p[0]**2 )
    denominator = ( x - (p[1]) )**2 + p[0]**2
    y = p[2]*(numerator/denominator)
    return y
 
def gaussian(x,p):
    c = p[0] / 2 / np.sqrt(2*np.log(2))
    numerator = (x-p[1])**2
    denominator = 2*c**2
    y = p[2]*np.exp(-numerator/denominator)
    return y
 
def residuals(p,y,x):
    err = y - lorentzian(x,p)
    return err


def local_fit(x, y):
    y_bg=y.min()
    p = [(x.max()-x.min())/2, (x.max()-x.min())/2+x.min() , x.max()-x.min()]
    # [fwhm, peak center, intensity] #
    pbest = leastsq(residuals, p, args=(y-y_bg,x), full_output=1)
    best_parameters = pbest[0]
    best_parameters[0] *= 2
    x_fit = np.linspace(1,100,100)
    fit = lorentzian(x_fit,best_parameters) + y_bg
    return best_parameters,  x_fit, fit
    
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
null_data = np.array([])

const = 2470000
truth = np.array([])

rsqs = np.array([])

ks_sum = np.array([])
data_fig = np.array([])



rsq1 =0

percent = 0
arr_first = 0
leastfit = np.array([])
leastlin = np.array([])

for i in day_test :
    
    k = np.loadtxt('/home/andychoi/datatest/'+str(i)+'fitted.txt')
    
    big_num = 0
    if k.size != 0 :
        k = np.transpose(k)
        N = k.size/3
        for n in k[0] :
            big_num = big_num+np.exp((541-n)/50)
        sum_truth = (9*big_num + 1*N*const/540)/(const)
        truth = np.append(truth,sum_truth)
        
        
        percent = percent + np.sort(k[1])[-(k[1].size/10):].sum()/k[1].sum()
        k[1] = np.log(k[1])
        
                
        if sum_truth >0.5 :
            z = np.polyfit(k[0],k[1],100)
            ks_sum = np.append(ks_sum,np.exp(k[1]).sum())
        data_fig = np.append(data_fig,[k[1].size])

        fit_result = local_fit(k[0], k[1])
        
        #plt.plot(fit_result[1],fit_result[2])
        
        p = np.linspace(0,0,20)
        if sum_truth >0.5 :
            solp, ier = leastsq(errorfunc, 
                   p, 
                   args=(k[0],k[1]),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)
                   
        if sum_truth >0.5 :
            slope, iere = leastsq(liner, 
                   np.array([0,0]), 
                   args=(k[0],k[1]),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)
            
                              
            y_fit = lin(k[0],slope)
            rtt = np.var(y_fit)
            ssr = np.var(k[1])
            
            rsq = rtt/ssr
            rsqs = np.append(rsqs,rsq)            
            rsq1 = rsq+rsq1
            #plt.plot(k[0],y_fit,'-r',lw=2)
            #plt.plot(k[0],k[1],'wo')
            
        if (arr_first == 0) : 
            leastlin = np.array([slope])
        else :
            if sum_truth >0.5 :
                leastlin = np.append(leastlin,[slope],axis =0)

        
        if (arr_first == 0) : 
            poly_tab =np.array([z])
            leastfit = np.array([solp])
        else :
            if sum_truth >0.5 :
                poly_tab = np.append(poly_tab,[z],axis=0)
                leastfit = np.append(leastfit,[solp],axis =0)
        arr_first = arr_first + 1
        #k[1] = (k[1]/k[1].sum())
        
        """"""

        #if (arr_first == 118) :
        #if ((arr_first == 118)|(arr_first==119)|(arr_first==120)|(arr_first==121)|(arr_first==122)) :
        plt.plot(k[0],np.exp(k[1]),'wo')
            #plt.plot(k[0],lin(k[0],slope),'r-')

"""    
Data = np.loadtxt('/home/andychoi/datatest/20150101.0fitted.txt')

Data = np.transpose(Data)
"""
truth0 = np.append(truth[0:55],truth[56:])
entropy = np.exp(truth0*truth0)/np.exp(truth0*truth0).sum()
#entropy = (truth0*truth0)/(truth0*truth0).sum()
t= np.array([0,0])
leaster = np.transpose(leastfit)[0]
leaster1 = np.transpose(leastlin)[0]
#leaster = np.transpose(leastlin)[1]
#leaster1 = np.transpose(leastlin)[0]
#leaster = np.exp(leaster)

ver_sum = ks_sum

ks_sum = (entropy*ks_sum)


solp3, ier3 = leastsq(liner, 
                   t, 
                   args=(np.log(ks_sum),leaster),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)
                   
solp4, ier4 = leastsq(liner, 
                   t, 
                   args=(np.log(ks_sum),leaster1),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)


x_fit = np.linspace(1,150,150)                  

y_fit = lin(np.log(ks_sum),solp3)
y_fit2 = lin(np.log(ks_sum),solp4)
y_fitted = lin(x_fit,solp3)

#plt.plot(x_fit,y_fitted)


#plt.plot(np.log(ks_sum),y_fit2,'r-')#for test
#plt.plot(np.log(ks_sum),leaster,'wo')
"""
plt.plot(np.log(ks_sum),leaster1,'wo')
plt.plot(np.log(ks_sum),y_fit2,'r',lw=2)
"""
#plt.plot(x_fit,ks_sum)
#plt.plot(x_fit,data_fig)

#plt.plot(np.log(ks_sum),leaster,'wo')

rtt = np.var(y_fit)*150
ssr = np.var(np.transpose(leastfit)[0])*150

rsq = rtt/ssr

rtt2 = np.var(y_fit2)*150
ssr2 = np.var(leaster)*150

rsq2 = rtt2/ssr2
#plt.plot(np.log(ks_sum),leaster,'wo')

for i in range(entropy.size) :
    for j in range(entropy.size-1) :
        if entropy[j] > entropy[j+1] :
            temp = entropy[j]
            entropy[j] = entropy[j+1]
            entropy[j+1] = temp
            
            temp2 = ks_sum[j]
            ks_sum[j]= ks_sum[j+1]
            ks_sum[j+1] = temp2
            
            temp3 = leaster[j]
            leaster[j]= leaster[j+1]
            leaster[j+1] = temp3
            
            temp4 = leaster1[j]
            leaster1[j]= leaster1[j+1]
            leaster1[j+1] = temp4
    
#plt.plot(np.log(ks_sum),leaster1,'wo')#for test

 

for i in range(15) :
    end = (i+1)*10
    start = i*10
    solp, ier = leastsq(liner, 
                   t, 
                   args=(np.log(ks_sum[start:end]),leaster[start:end]),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)
    
    
    solp1, ier1 = leastsq(liner, 
                   t, 
                   args=(np.log(ks_sum[start:end]),leaster1[start:end]),
                   Dfun=None,
                   full_output=False,
                   ftol=1e-9,
                   xtol=1e-9,
                   maxfev=100000,
                   epsfcn=1e-10,
                   factor=0.1)
    
    
    if i == 0 :
        fit_entropy = np.array([entropy[start:end].sum()])
        w = np.array([solp])
        w1 = np.array([solp1])
    else : 
        w = np.append(w,[solp],axis=0)
        w1 = np.append(w1,[solp1],axis=0)
        fit_entropy = np.append(fit_entropy,[entropy[start:end].sum()],axis=0)
    
deep_learning = np.append([entropy],[ks_sum,leaster,leaster1],axis=0)

sol = np.array([0,0])
sol_a = np.array([0,0])



for i in range(15) :
    sol = sol + np.array([fit_entropy[i]*(w[i][0]),fit_entropy[i]*(w[i][1])])
    
    sol_a = sol_a + np.array([fit_entropy[i]*(w1[i][0]),fit_entropy[i]*(w1[i][1])])


leastg = np.transpose(leastfit)[0]

leastg =np.sort(leastg)
leastg = leastg

counter=np.linspace(0,0,15)

u = 0.4
"""
for i in leastg :
    if i >(9+u*15) :
        counter[15] = counter[15]+1
    elif i >(9+u*14):
        counter[14] = counter[14]+1
    elif i >(9+u*13):
        counter[13] = counter[13]+1
    elif i >(9+u*12):
        counter[12] = counter[12]+1
    elif i >(9+u*11):
        counter[11] = counter[11]+1
    elif i >(9+u*10):
        counter[10] = counter[10]+1
    elif i >(9+u*9):
        counter[9] = counter[9]+1
    elif i >(9+u*8):
        counter[8] = counter[8]+1
    elif i >(9+u*7):
        counter[7] = counter[7]+1
    elif i >(9+u*6):
        counter[6] = counter[6]+1
    elif i >(9+u*5):
        counter[5] = counter[5]+1
    elif i >(9+u*4):
        counter[4] = counter[4]+1
    elif i >(9+u*3):
        counter[3] = counter[3]+1
    elif i >(9+u*2):
        counter[2] = counter[2]+1
    elif i >(9+u*1):
        counter[1] = counter[1]+1
    else :
        counter[0] = counter[0]+1    
count_gaussain = gaussian(np.linspace(9,15,15),np.array([np.std(leastg),np.mean(leastg),22])) 
"""
num_bins =50   

#n, bins, patches = plt.hist(leastg, num_bins, normed=1, facecolor='green', alpha=0.5)    
#y = mlab.normpdf(bins, np.mean(leastg), np.std(leastg))
#plt.plot(bins, y, 'r--')
#plt.title(r'Histogram of a: $\mu=11.93$, $\sigma=1.18$')
#plt.plot(bins[0:50],n,'wo')
#plt.plot(bins,y,'r-')
n = n*leastg.sum()/150
y = y*leastg.sum()/150
#plt.plot(bins[0:50],n,'wo')
#plt.plot(bins,y,'r-')

ch2, pval = stats.chisquare(n,y[0:50])
"""
for i in poly_tab :
    f =np.poly1d(i)
    x = np.linspace(1,300,300)
    y = f(x)
    #y = poli(x,i)
    #plt.plot(x,y)


for i in leastfit :
    x = np.linspace(1,400,400)
    y = poli(x,i)
    #plt.plot(x,y)"""

    #"""youdothis"""
    
#plt.plot(np.log(ks_sum),lin(np.log(ks_sum),sol_a))
plt.show()