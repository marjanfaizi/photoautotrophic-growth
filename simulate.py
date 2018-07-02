############################### PACKAGES ###############################
import csv
import numpy as np
import scipy.optimize
from photo_mini import PhotoMini
from numpy import genfromtxt
import pandas as pd
from scipy.integrate import odeint
########################################################################

################################ INPUT #################################
pref = '/Users/marjanfaizi/Documents/Seafile/PhD/Data/'
# variable parameter set     
p = genfromtxt(pref+'variable_parameters.csv',delimiter=',',skip_header=1)
Mcell=1.4e10; Acell=3.85e-9; micro=1.0e-6; N=6.022e23; Vcell=2.24e-14
# initial conditions
S_init = np.zeros(7)
S_init[1] = 6.642e9
S_init[2] = 1.0e6
S_init[6] = 1.0e6
# time interval
t = np.arange(0.0, 8.0e9, 10000.0)
Irr = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 27.5, 55.0, 110.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0,
	 			880.0, 990.0, 1100.0])
Irr = Irr[::-1]
# carbon availability
#Cix = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0,
#				600.0, 800.0, 1000.0, 5000.0])
Cix = np.array([1000.0])
Cix = Cix[::-1]
# create some constraints
# sum(beta) <= 1 ------> 1-sum(beta) > 0
# for all beta_i: beta_i > 0
cons = ({'type':'ineq','fun': lambda x: 1.0-x.sum()},
         {'type':'ineq','fun': lambda x: int(all([i>=0.0 for i in x]))})
bnds = [(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]

# column names for dataframe
#header = ['cix','gr','beta_r','beta_t','beta_m','beta_p','ci','aa', 'R', 'T', 'M', 'P', 'e']
header = ['light','gr','beta_r','beta_t','beta_m','beta_p','ci', 'aa', 'R', 'T', 'M', 'P', 'e']
########################################################################

########################## FIND MAX FUNCTION ###########################
def Findmax(beta,I,cix,p,S0,t):
	S = odeint(PhotoMini,S0,t,args=(np.abs(beta),I,cix,p),mxstep=10000000)
	vd = p[0]*60.0*Acell*((micro*N*cix)-(S[-1,0]/Vcell))
	vt = S[-1,3]*p[1]*60.0*(cix/(p[2]+cix))*(S[-1,6]/(p[6]+S[-1,6]))
	return -((vd+vt)/(Mcell*5.0))
########################################################################

############################## OPTIMIZATION ############################
x_init = np.array([0.0078,0.0026,0.04489,0.94466])

for c in range(np.size(Cix)):
#for i in range(np.size(Irr)):
	# initial conditions
	'''
	S0 = np.zeros(7)
	S0[1] = 6.642e9
	S0[2] = 1.0e6
	S0[6] = 1.0e6
	'''
	S0 = S_init
	x_init = np.array([0.01,0.0,0.01,0.98])
	output = np.zeros([np.size(Irr),13])
#	output = np.zeros([np.size(Cix),13])
	x0 = x_init
	for i in range(np.size(Irr)):
#	for c in range(np.size(Cix)):
		print '##### RESULT FOR: ', 'I=', Irr[i], 'C=', Cix[c], ' #####'
		res = scipy.optimize.minimize(Findmax,x0,args=(Irr[i],Cix[c],p, S0,t),method='SLSQP',constraints=cons,bounds=bnds, options={'maxiter':100,'ftol':1.0e-9})
		if (-res.fun<1.0e-6):
			print "SECOND LOOP"
			x0 = [0.005,0.0,0.005,0.99]
			res = scipy.optimize.minimize(Findmax,x0,args=(Irr[i],
					Cix[c],p,S0,t),method='SLSQP',constraints=cons,
					bounds=bnds,options={'maxiter':100,'ftol':1.0e-9})
		x0 = np.abs(res.x)
		S = odeint(PhotoMini,S0,t,args=(np.abs(res.x), Irr[i], Cix[c], p), mxstep=1000000)
		output[i] = np.concatenate(([Irr[i]],[-res.fun],np.abs(res.x),S[-1,:]))
#		output[c] = np.concatenate(([Cix[c]],[-res.fun],np.abs(res.x),S[-1,:]))
		if (all(i>=0.0 for i in S[-1,:])): S0 = S[-1,:]
		if (i==0): S_init = S[-1,:]
		print 'BETAS: ', x0
		print 'LAMBDA: ', -res.fun
	file_name = 'data_extended_atp1e10_cix'+str(Cix[c])+'.csv'
#	file_name = 'data_basic_irr'+str(Irr[i])+'.csv'
	output = output[::-1]
	df = pd.DataFrame(output,columns=header)
	df.to_csv(file_name,index=0)
########################################################################


