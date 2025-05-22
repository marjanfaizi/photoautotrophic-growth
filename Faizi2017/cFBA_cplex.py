##################################### REQUIRED PACKAGES ############################################
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
####################################################################################################


##################################### MODEL PARAMETERS #############################################
### Parameter
sigma = 0.166*60.0*60.0
kd = 1.6e-6
Kt = 15.0
####################################################################################################


###################################### DISCRETIZE TIME #############################################
## time intervall is given
time_start = 0
time_end = 24
time_points = 100
time = np.linspace(time_start, time_end, num = time_points, dtype = float)
####################################################################################################

############################ LIGHT FUNCTION FOR DIURNAL GROWTH #####################################
## half-wave rectified sine function for day-night cycle
# Input: lmax is the maximum light intensity and t is an array of time points for which the light intensity should be calculated
# Output: an array of light intensites 
def light_cycle( lmax, t ):
	if ( np.sin(2.0*np.pi*t/24.0) >= 0.0 ):
		l = lmax * np.sin(2.0*np.pi*t/24.0)
	else: l = 0.0
	return l
####################################################################################################

#################################### GENERATE FLUXES OUTPUT ########################################
# generate matrix with rows for the fluxes and species and columns for the time points
def fluxes_output(var_results):
	# generate keys names
	fluxes = ['vt', 'vm', 'vER', 'vET', 'vEM', 'vEP0', 'v1', 'v2', 'vi']
	species = ['ER', 'ET', 'EM', 'EP0', 'EP1']
	time_fluxes = [(time[j]+time[j-1])/2 for j in range(1,len(time))]
	key_fluxes = [ ('V'+str(time_fluxes[t])+'_'+str(i),i,t) for i in range(len(fluxes)) for t in range(len(time_fluxes)) ]
	key_species = [ ('C'+str(time[t])+'_'+str(i),i,t) for i in range(len(species)) for t in range(len(time)) ]
	# create matrices with all keys and values of zero
	flux_matrix = np.zeros([len(fluxes),len(time)-1])
	species_matrix = np.zeros([len(species),len(time)])
	# fill matrices
	for i in key_fluxes:
		if var_results.has_key(i[0]):
			flux_matrix[i[1:]] += var_results[i[0]]
	for i in key_species:
		if var_results.has_key(i[0]):
			species_matrix[i[1:]] += var_results[i[0]]
	return flux_matrix, species_matrix


####################################################################################################


########################################## MODEL ###################################################
#### Create model for a given growth rate and external light
def run_model(mu, cix, lmax):

	### Create the minimal model in gurbobi
	model = Model('photo_model')


	### Create variables: 
	fluxes = ['vt', 'vm', 'vER', 'vET', 'vEM', 'vEP0', 'v1', 'v2', 'vi']
	species = ['ER', 'ET', 'EM', 'EP0', 'EP1']
	var_idx_fluxes = np.arange(0, len(fluxes))
	var_idx_species = np.arange(0, len(species))
	# for every species one time point tj and fluxes (tj+tj-1)/2
	time_fluxes = [(time[j]+time[j-1])/2 for j in range(1,len(time))]
	#lower bounds 0 if all reactions are irreversilbe and default upper bounds are infinity
	# add fluxes
	v = []
	for i in time_fluxes:
		flux_name = 'V'+str(i)
		v.append(model.continuous_var_list(var_idx_fluxes, lb=0, name = flux_name))
	# add species
	c = []
	for i in time:
		species_name = 'C'+str(i)
		c.append(model.continuous_var_list(var_idx_species, lb=0, name = species_name))

	### add constraints
	# mass conservation for all internal species ci, atp, aa,
	met = np.array([ [1.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [-1.0, -45.0, -3.0*7358.0, -3.0*1681.0, -3.0*28630.0, -3.0*95451.0, 0.0, 8.0, 0.0],
					 [0.0, 1.0, -7358.0, -1681.0, -28630.0, -95451.0, 0.0, 0.0, 95451.0]
				  	 ])
	model.add_constraints( i == 0.0 for t in range(1,len(time)) for i in np.dot(met,v[t-1]) )
	# Biosynthesis of proteins/macromolecules R, T, M, P0, P1
	macro = np.array([ [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
					   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0],
					   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0]
				   	   ])
	#model.add_constraints( c[t][i] == c[t-1][i]+np.dot(macro[i],v[t-1])-mu*((c[t][i]+c[t-1][i])/2.0) for i in range(len(macro)) for t in range(1,len(time)) )
	model.add_constraints( c[t][i] == c[t-1][i]+(time[t]-time[t-1])*np.dot(macro[i],v[t-1]) for i in range(len(macro)) for t in range(1,len(time)) )
	# growht rate
	model.add_constraints( mu*c[0][i] == c[-1][i] for i in range(len(species)) ) 
	# constant density
	alpha = np.array([7358.0, 1681.0, 28630.0, 95451.0, 95451.0])
	model.add_constraint( np.dot(alpha,c[0]) == 1.4e10 ) 
	# enzyme capacity constraints for R, T, M, P1
	venz = np.array([[0.0, 0.0, 7358.0/(1320.0*60.0), 1681.0/(1320.0*60.0), 28630.0/(1320.0*60.0), 95451.0/(1320.0*60.0), 0.0, 0.0, 0.0],
				 	 [1.0/(726.0*60.0*(cix/(Kt+cix))), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				 	 [0.0, 1.0/(545.0*60.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/(1900.0*60.0), 0.0]
				 	 ])
	cenz = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
				 	 [0.0, 1.0, 0.0, 0.0, 0.0],
				 	 [0.0, 0.0, 1.0, 0.0, 0.0],
					 [0.0, 0.0, 0.0, 0.0, 1.0]
				 	 ])
	model.add_constraints( np.dot(venz[i],v[t-1]) <= np.dot(cenz[i],np.add(c[t],c[t-1])/2.0) for i in range(len(venz)) for t in range(1,len(time)) )
	# light input
	vlight = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/(kd*sigma)],
					 	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0/sigma, 0.0, 0.0]
				  	 	])					
	clight = np.array([ [0.0, 0.0, 0.0, 0.0, 1.0],
						[0.0, 0.0, 0.0, 1.0, 0.0]
						]) 	
	model.add_constraints( np.dot(vlight[i],v[t-1]) == np.dot(clight[i],np.add(c[t],c[t-1])/2.0)*light_cycle(lmax,time[t]) for i in range(len(vlight)) for t in range(1,len(time)) )
	### optimize
	model.solve()
	
	return model
####################################################################################################


####################################### BINARY SEARCH ##############################################
### binary search for mu (given ci_ex and light_ex)
def binary_search(mu, cix, lmax, mu_lb=0.0, mu_ub=0.0, mu_prev=0.0,):
	# if prev_model is set to 1, calculate mu for the previuos model for which a solution extist
	prev_model = 0.0
	m = run_model(mu, cix, lmax)
	# if optimization terminated successfully
	if m.solve_details.status_code ==  1:
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
		return binary_search(mu_new, cix, lmax, mu_lb, mu_ub, mu_prev)
	else:
		if prev_model == 0.0: 
			return [mu, m.solution.as_dict()]	
		else: 	 
			m = run_model(mu_prev, cix, lmax)
			return [mu_prev, m.solution.as_dict()]	
####################################################################################################	
		 

######################################### SIMULATION ###############################################
### run binary search
%timeit gr, flux = binary_search(4.0, 1000.0, 2000.0)

v,c = fluxes_output(flux)
####################################################################################################


######################################### PLOT RESULTS #############################################
# Plot style
sns.set(color_codes=True)
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.14)



# plot growth rate for different light intensities
fig = plt.figure()
plt.plot(time, c[0,:],'-or', label='Er')
plt.plot(time, c[1,:],'-og', label='Et')
plt.plot(time, c[2,:],'-ob', label='Em')
plt.plot(time, c[3,:]+c[4,:],'-oy', label='Ep')
plt.xlabel('time [h$^{-1}$]')
plt.ylabel('molecules per cell')
sns.despine()
plt.legend()
plt.tight_layout()
plt.savefig('minimal_model_cFBA.pdf')
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

	
