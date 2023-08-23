from pandas import read_csv
from matplotlib import pyplot as plt
import pandas as pd
from numpy import arange
from scipy.optimize import curve_fit
import numpy as np

# #Define Objective function: Independent variable t, parameters A, s, k, off.
# def objective(t, A, s, k, off):
#     #return (y_i - y_e)/(1 + s*np.exp(-k*(t - t0))) + y_e
#     t0 = 15
#     return A / (1 + s*np.exp(-k*(t-t0))) + off

#Define Objective function: Independent variable t, parameters t0, s, k, A, xe.
def objective(t, t0, s, k, A, x_off):
    return A / (1 + s*np.exp(-k*(t-t0))) + x_off

#Open and read csv
file = "newGrowthCurvesJuly2023.csv"
dataframe = read_csv(file, sep = ",", header=0)
data = dataframe.values
array = dataframe.to_numpy()


#Convert timedeltas into int values with units of minutes
#Assign input and output values, plot 

dataframe['minutes'] = dataframe['Time']/pd.Timedelta(minutes=1)        #Timedelta conversion
x = dataframe['minutes'].values         # y is just data[:, i] for column i

plt.figure()
plt.xlabel('Time (mins)')
plt.ylabel('Optical Density')
plt.title('June Growth Curve Fits')
column_names = ['Mean Sept w/ PE', 'Mean Sept w/out PE', 'Mean June w/ PE', 'Mean June w/out PE', 'Sept w/ PE', 'Sept w/out PE', 'June w/ PE', 'June w/out PE']

#Do everything outside of a loop first
l1, l2 = plt.plot(x,data[:,7],'bo', x, data[:,8], 'ro')

PEparams, _ = curve_fit(objective, x, data[:,7], p0 = [400, 0.5, 0.5, 1.4, 0.1087])
NoPEparams, _ = curve_fit(objective, x, data[:,8], p0 = [400, 0.5, 0.5, 1.4, 0.1087]) 
Pt0, Ps, Pk, PA, Px_off = PEparams
t0, s, k, A, x_off = NoPEparams
    
#Define new array of inputs and plot associated output achieved from parameter-fit function
x_curve = arange(min(x),max(x),1)
y_curve_PE = objective(x_curve, Pt0, Ps, Pk, PA, Px_off) 
y_curve_NoPE = objective(x_curve, t0, s, k, A, x_off) 

l3 = plt.plot(x_curve, y_curve_PE, '--', color = 'black')
l4 = plt.plot(x_curve, y_curve_NoPE, '--', color = 'black')
             
txtpt1 = (f" {column_names[6]} \n Inflection point ={Pt0:.3f} mins \n Symmetry Parameter = {Ps:.3f} \n Growth Rate = {Pk:.3f} opt. dens. per min \n Asymptotic Difference = {PA:.3f} opt. dens. \n Vertical offset = {Px_off:.3f} opt. dens.")
txtpt2 = (f" {column_names[7]} \n Inflection point ={t0:.3f} mins \n Symmetry Parameter = {s:.3f} \n Growth Rate = {k:.3f} opt. dens. per min \n Asymptotic Difference = {A:.3f} opt. dens. \n Vertical offset = {x_off:.3f} opt. dens.")
plt.text(0,1.2,txtpt1, ha = 'left', wrap = True)
plt.text(400,0.2,txtpt2, ha = 'left', wrap = True)
plt.legend((l1, l2),('June w/ PE', 'June w/out PE'), loc = 'center right')
image = f"JuneCurveGraph.jpg"
plt.savefig(image)



# i = 0
# j = 1

# while i<=(len(dataframe.T)-1)/2 - 1:
#     plt.subplot(4,1,i+1)
#     plt.plot(x,data[:,j],'bo', x, data[:,j+1], 'ro')
#     i+=1
#     j+=2

# k = 1

# while k <= len(dataframe.T):

# # 5 parameter model
#     params, _ = curve_fit(objective, x, data[:,k], p0 = [400, 0.5, 0.5, 1.4, 0.1087])
#     a, b, c, d, e = params
    
#     #Define new array of inputs and plot associated output achieved from parameter-fit function
#     x_curve = arange(min(x),max(x),1)
#     y_curve = objective(x_curve, a, b, c, d, e) 
#     plt.subplot(4,1,k)
#     plt.plot(x_curve, y_curve, '--', color = 'green')
#     k+=1
#     if                
#     #Save plot to a file 

#     t = (f"Inflection point ={t0} " f"Symmetry Parameter = {s}" f"Growth Rate = {k}" f"Xi - Xe = {A}" f"Horizontal offset = {x_off}")
#     plt.text(15,1,t, ha = 'left', wrap = True)
#     image = f".jpg"
#     plt.savefig(image)


