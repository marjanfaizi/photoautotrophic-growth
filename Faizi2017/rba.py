##################################### REQUIRED PACKAGES ############################################
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
####################################################################################################

##################################### MODEL PARAMETERS #############################################
### Parameter
sigma = 0.166
kd = 1.6e-6
Kt = 15.0
####################################################################################################

########################################## MODEL ###################################################
#### Create model for a given growth rate and external light
def run_model(mu, light, cix):
	
	### Create the minimal model in gurbobi
	model = Model('photo_model')

	### Create variables: 
	# flux vector and species concentration
	x_name = ['vt', 'vm', 'vER', 'vET', 'vEM', 'vEP0', 'v1', 'v2', 'vi', 'ER', 'ET', 'EM', 'EP0', 'EP1']
	x_len = len(x_name)
	idx = np.arange(0,x_len)
	obj = np.zeros([x_len])
	# lower bounds, all reactions are irreversible
	lb = np.zeros([x_len])
	# upper bounds
	ub = np.repeat(GRB.INFINITY, x_len)
	# add all variables to the model
	x = model.addVars(idx,lb=lb, ub=ub, obj=obj,vtype=GRB.CONTINUOUS,name='x')  

	### add constraints
	# mass conservation for all species  ci, atp, aa, R, T, M, P0, P1
	A_eq = np.array([[1.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [-1.0, -45.0, -3.0*7358.0, -3.0*1681.0, -3.0*28630.0, -3.0*95451.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [0.0, 1.0, -7358.0, -1681.0, -28630.0, -95451.0, 0.0, 0.0, 95451.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -mu, 0.0, 0.0, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -mu, 0.0, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -mu, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -mu, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -mu]
				  	 ])
	b_eq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	b_len = len(b_eq)
	model.addConstrs( (quicksum( A_eq[i,j]*x[j] for j in range(x_len) ) == b_eq[i] for i in range(b_len)), 'mass_conservation' )  
	# constant density
	A_eq2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7358.0, 1681.0, 28630.0, 95451.0, 95451.0])
	model.addConstr( quicksum(A_eq2[i]*x[i] for i in range(x_len)) == 1.4e10, 'fixed_density')
	# enzyme capacity constraints for R, T, M, P1
	A_ub = np.array([[0.0, 0.0, 7358.0/(1320.0*60.0), 1681.0/(1320.0*60.0), 28630.0/(1320.0*60.0), 95451.0/(1320.0*60.0), 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
				 	 [1.0/(726.0*60.0*(cix/(Kt+cix))), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
				 	 [0.0, 1.0/(545.0*60.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/(1900.0*60.0), 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
				 	 ])
	b_ub = np.array([0.0, 0.0, 0.0, 0.0])
	b2_len = len(b_ub)
	model.addConstrs( (quicksum( A_ub[i,j]*x[j] for j in range(x_len) ) <= b_ub[i] for i in range(b2_len)), 'enzyme_capacity')
	# light input
	A_eq3 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/(kd*light*sigma*60.0*60.0), 0.0, 0.0, 0.0, 0.0, -1.0],
					  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/(light*sigma*60.0*60.0), 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
				  	 ])
	b_eq3 = np.array([0.0, 0.0])
	b3_len = len(b_eq3)
	model.addConstrs( (quicksum( A_eq3[i,j]*x[j] for j in range(x_len) ) == b_eq3[i] for i in range(b3_len)), 'light_input' )  

	### optimize
	model.optimize()
	
	return model
####################################################################################################


####################################### BINARY SEARCH ##############################################
### binary search for mu (given ci_ex and light_ex)
def binary_search(mu, light_ex, cix, mu_lb=0.0, mu_ub=0.0, mu_prev=0.0,):
	# if prev_model is set to 1, calculate mu for the previuos model for which a solution extist
	prev_model = 0.0
	m = run_model(mu, light_ex, cix)
	# if optimization terminated successfully
	if m.status ==  GRB.status.OPTIMAL:
		mu_lb = mu
		if mu_ub == 0.0: mu_new = 2.0*mu_lb
		else: mu_new = (mu_lb+mu_ub)/2.0
		mu_prev = mu
	# and if not 
	else:
		prev_model = 1.0
		mu_ub = mu
		if mu_lb == 0.0: mu_new = mu_ub/2.0
		else: mu_new = (mu_ub+mu_lb)/2.0
	if 100.0*(np.abs(mu_new-mu)/mu) > 0.01:
		return binary_search(mu_new, light_ex, cix, mu_lb, mu_ub, mu_prev)
	else:
		if prev_model == 0.0: 
			return [mu, m.getAttr('x')]	
		else: 	 
			m = run_model(mu_prev, light_ex, cix)
			return [mu_prev, m.getAttr('x')]	
####################################################################################################	
		 

######################################### SIMULATION ###############################################
### Simulate for different light intensities
light = np.arange(10.0, 1500.0, 50.0)
#light = [1.0e1, 5.0e1, 1.0e2, 5.0e2, 1.0e3, 5.0e3]
#light = [2.0e7]
gr = np.zeros([len(light)])
flux = np.zeros([len(light),14])
for i in range(len(light)):
	print '##### LIGHT:'+str(i)+' #####'
	gr[i], flux[i,:] = binary_search(0.01,light[i], 1.0e11)

### Simulate for different carbon concentrations
cix = [1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1, 1.0e2, 1.0e3]
cix = np.arange(1.0e-3, 1.0e1, 1.0e-1)
gr = np.zeros([len(cix)])
flux = np.zeros([len(cix),14])
for i in range(len(cix)):
	print '##### CIX:'+str(i)+' #####'
	gr[i], flux[i,:] = binary_search(0.01, 1000.0, cix[i])
####################################################################################################


######################################### PLOT RESULTS #############################################
# Plot style
sns.set(color_codes=True)
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.14)



# plot growth rate for different light intensities
fig = plt.figure()
plt.plot(cix, gr,'-o')
#plt.xlabel('light intensity [$\mu$E m$^{-2}$s$^{-1}$]')
plt.xlabel('external inorganic carbon [$\mu$M]')
plt.ylabel('growth rate [h$^{-1}$]')
sns.despine()
plt.title('With Light = 1000 [$\mu$E m$^{-2}$s$^{-1}$]')
#plt.legend()
plt.tight_layout()
plt.savefig('growth_rate_complex_model_increasing_cix.pdf')
plt.show()


# plot growth rate for different light intensities
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(221)
plt.plot(gr, 7358.0*flux[:,9]/(flux[:,9]*7358.0+flux[:,10]*1681.0+flux[:,11]*28630.0+(flux[:,12]+flux[:,13])*95451.0),'ro')
plt.ylabel('Ribosomes [molecules]')
plt.xlabel('growth rate [h$^{-1}$]')
ax = fig.add_subplot(222)
plt.plot(gr, 1681.0*flux[:,10]/(flux[:,9]*7358.0+flux[:,10]*1681.0+flux[:,11]*28630.0+(flux[:,12]+flux[:,13])*95451.0),'go')
plt.ylabel('Transporter [molecules]')
plt.xlabel('growth rate [h$^{-1}$]')
ax = fig.add_subplot(223)
plt.plot(gr, 28630.0*flux[:,11]/(flux[:,9]*7358.0+flux[:,10]*1681.0+flux[:,11]*28630.0+(flux[:,12]+flux[:,13])*95451.0),'bo')
plt.ylabel('Metabolic Enzymes [molecules]')
plt.xlabel('growth rate [h$^{-1}$]')
ax = fig.add_subplot(224)
plt.plot(gr, 95451.0*(flux[:,12]+flux[:,13])/(flux[:,9]*7358.0+flux[:,10]*1681.0+flux[:,11]*28630.0+(flux[:,12]+flux[:,13])*95451.0),'yo')
plt.ylabel('Photosystem [molecules]')
plt.xlabel('growth rate [h$^{-1}$]')
sns.despine()
plt.tight_layout()
plt.savefig('growth_laws_complex_model_increasing_cix.pdf')
plt.show()


####################################################################################################

	
