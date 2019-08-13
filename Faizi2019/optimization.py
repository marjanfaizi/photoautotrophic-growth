#####################################################################################################################
############################################# REQUIRED PACKAGES #####################################################
#####################################################################################################################
import numpy as np
from apm import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
############################################## INPUT PARAMETER ######################################################
#####################################################################################################################
## path and suffix name to save simulated data
path = '../Simulated_Data/'
filename_suffix = '_010819'
## incident light intensity
I0 = np.array([ 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0,
                190.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1000.0, 1100.0 ])
rho = np.concatenate([ np.arange(0.0,5.0,0.1), np.arange(5.0,31.0,1.0), np.arange(32.0,50.1,2.0) ])*1.0e8
outflux = np.arange(0.0,0.102,0.002)
zm = np.array([ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.4, 5.0, 7.5, 10.0, 20.0, 30.0, 40.0, 50.0 ]) 
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
##################################################  FUNCTIONS #######################################################
#####################################################################################################################
## CONNECT TO SERVER 
def connect( model_name ):
	## connect to server
	server = 'http://byu.apmonitor.com'
	app = model_name
	apm(server, app,'clear all')
	## load model file
	apm_load(server, app, model_name+'.apm')
	## select steady-state optimization
	apm_option(server,app,'nlc.imode',3)
	## pption to select solver (1=APOPT, 2=BPOPT, 3=IPOPT)
	apm_option(server,app,'nlc.solver',3)
	## change iteration and tolerance to get a more precise maximum
	apm_option(server,app,'nlc.max_iter',1000000)
	apm_option(server,app,'nlc.max_time',500)
	return server, app

## OPTIMIZATION FUNCTION
def apm_optimize(model_name, par_input, input_name, var_list, S_list, flux_list, output_array, output_header, output_filename):
	## connect to server and create model
	server, app = connect( model_name )
	## designate parameters to change
	apm_info(server,app,'FV', input_name)
	## measure time required for optimization
	start_time = time.time()
	#### start optimization
	for i in range(len(par_input)):
		## change external condition
		apm_meas(server,app,input_name,par_input[i])
		## solve on APM server
		solver_output = apm(server,app,'solve')
		res = apm_sol(server,app)
		if (not apm_tag(server,app,'nlc.appstatus')):
			solver_output = apm(server,app,'solve')
			res = apm_sol(server,app)
		## save current optimization results
		print (res['growth_rate'], par_input[i], apm_tag(server,app,'nlc.appstatus'))
		var = np.array([ res[v] for v in var_list ])
		S = np.array([ res[s] for s in S_list ])
		flux = np.array([ res[f] for f in flux_list ])
		output_array[i] = np.concatenate(( [par_input[i]], [res['growth_rate']], var, S, flux ))
	## stop the time
	print("--- %s seconds ---" % (time.time() - start_time))
	## write results into csv file
	output_array = output_array[::-1]
	df = pd.DataFrame(output_array, columns=output_header)
	df.to_csv(path+output_filename+filename_suffix+'.csv',index=0)
	return df
	
	
## SENSITIVITY ANALYSIS
def change_par_opt( model_name, par, par_name ):
	## connect to server and create model
	server, app = connect( model_name )
	## designate parameters to change
	apm_info( server, app, 'FV', par_name )	
	apm_meas( server, app, par_name, par )
	## solve on APM server
	solver_output = apm( server, app, 'solve' )
	res = apm_sol( server, app )
	counter = 0
	while ( (not apm_tag(server, app, 'nlc.appstatus')) and (counter<10) ):
		solver_output = apm( server, app, 'solve' )
		res = apm_sol( server, app )
		counter += 1
		print (counter)
			
	return res['growth_rate']*1.0e-8*res['rho']

def sens_P( model_name, dist_par, dist=0.0001 ):
	# increase/decrease one paramter for 0.1%
	p_fw = np.fromiter( dist_par.values(), dtype=float ) * ( 1.0 + dist )
	p_bw = np.fromiter( dist_par.values(), dtype=float ) - np.fromiter( dist_par.values(), dtype=float ) * dist
	dist_name = list( par.keys() )
	delta = np.zeros( np.size(dist_name) )
	# iterate over each parameter
	for i in range( np.size(dist_name) ):
		# calculate the new controlled value: productivity
		c_fw = change_par_opt( model_name, p_fw[i], dist_name[i] )
		c_bw = change_par_opt( model_name, p_bw[i], dist_name[i] )
		print ( c_fw, c_bw )
		# calculate increase/decrease compared to normal values
		p_dist = p_fw[i] - p_bw[i]
		c_dist = c_fw - c_bw
		delta[i] = ( (p_bw[i]+p_dist/2.0) * c_dist ) / ( (c_bw+c_dist/2.0) * p_dist )
		print ( delta[i] )

	return delta

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
############################################# SINGLE CELL ###########################################################
#####################################################################################################################
I0 = np.array([ 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0,
                190.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1000.0, 1100.0, 1200.0 ])
input_name = 'i0'
input_value = I0
input_value = input_value[::-1]
## header (column names) for csv file
single_var = [ 'beta_et', 'beta_ec', 'beta_eq', 'beta_em', 'beta_r', 'beta_psu', 'beta_pq' ]
single_S = [ 'ci', 'c3', 'aa', 'et', 'ec', 'eq', 'em', 'r', 'psu0', 'psu1', 'e' ]
single_flux = [ 'vt', 'vc', 'vq', 'vm', 'vgamma', 'v1', 'v2', 'vi', 'sigma' ]
single_cell_header = [ input_name, 'mu' ] + single_var + single_S + single_flux
## save result
output_single = np.zeros([np.size(input_value),len(single_cell_header)])
## run optimization
single_cell_results = apm_optimize( 'single_cell_model', input_value, input_name, single_var, single_S, single_flux,
                                     output_single, single_cell_header, 'single_cell' )
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
############################################## POPULATION ###########################################################
#Ieff = np.array([ 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 
#				  150.0, 170.0, 190.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1000.0, 1100.0, 1200.0 ])
Ieff = np.array([ 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0 ])
#outflux = np.arange(0.0,0.106,0.002)
input_name = 'ieff'
input_value = Ieff
input_value = input_value[::-1]
## header (column names) for csv file
population_var = [ 'i0', 'beta_et', 'beta_ec', 'beta_eq', 'beta_em', 'beta_r', 'beta_psu', 'beta_pq' ]
population_S = [ 'ci', 'c3', 'aa', 'et', 'ec', 'eq', 'em', 'r', 'psu0', 'psu1', 'e', 'rho' ]
population_flux = [ 'vt', 'vc', 'vq', 'vm', 'vgamma', 'v1', 'v2', 'vi', 'alpha', 'ieff' ]
population_header = [ input_name, 'mu' ] + population_var + population_S + population_flux
## save result
output_population = np.zeros([np.size(input_value),len(population_header)])
## run optimization
population_results = apm_optimize( 'population_model', input_value, input_name, population_var, population_S, 
									population_flux, output_population, population_header, 'chemostat' )
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
########################################### PRODUCT FORMATION #######################################################
#####################################################################################################################
#EP = np.concatenate([ np.arange(0.0,0.1,0.05), np.arange(0.1,1.0,0.1),np.arange(1.0,10.0,0.5),np.arange(10.0,95.0,5.0), [90.3] ])*1.0e4
outflux = np.arange(0.0,0.1,0.002)
#outflux = np.array([ 0.0005, 0.028, 0.0995 ])
input_name = 'growth_rate'
input_value = outflux
input_value = input_value[::-1]
## header (column names) for csv file
product_var = [ 'beta_et', 'beta_ec', 'beta_eq', 'beta_ex', 'beta_em', 'beta_r', 'beta_psu', 'beta_pq' ]
product_S = [ 'ci', 'c3', 'aa', 'et', 'ec', 'eq', 'ex', 'em', 'r', 'psu0', 'psu1', 'e', 'rho', 'mx' ]
product_flux = [ 'i0', 'vt', 'vc', 'vq', 'vm', 'vx', 'vgamma', 'v1', 'v2', 'vi', 'alpha', 'ieff' ]
product_formation_header = [ input_name, 'mu' ] + product_var + product_S + product_flux 
## save result
output_product = np.zeros([np.size(input_value),len(product_formation_header)])
## run optimization
product_results = apm_optimize( 'heterogeneous_production', input_value, input_name, product_var, product_S, product_flux,
								 output_product, product_formation_header, 'heterogeneous_production' )
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
########################################## SENSITIVITY ANALYSIS #####################################################
#####################################################################################################################
par = { 'kcat_t':43560.0, '$K_t':15.0, 'kcat_c':32700.0, 'Kc':456702.458, 'kcat_q':72000.0, 'K_q':1.0e4, 
		'kcat_m':72000.0, 'K_m':1.0e4, 'gamma_max':79200.0, 'K_a':1.0e4, 'Ke':1.0e4, 'dp':0.04347826, 
		'vme':7.0e9, 'tau':1800000.0, 'kd':2.7e-7, 'sigma':15.0, 'alpha0':1.0e-10 }	
# call model withthe following input parameters: i0=440, D=0.026, cix=1.0e5	
model_name = 'population_model'

res = sens_P( model_name, par )

np.save( path+'sensitivity_analysis_fix_D0026.npz', res )

labels = [ '$\\rm k^t_{cat}$', '$\\rm K_t$', '$\\rm k^c_{cat}$', '$\\rm K_c$', '$\\rm k^q_{cat}$', 
			'$\\rm K_q$', '$\\rm k^m_{cat}$', '$\\rm K_m$', '$\\rm \gamma_{max}$', '$\\rm K_a$', '$\\rm K_e$', 
			'$\\rm d_p$', '$\\rm v_{me}$', '$\\rm \\tau$', '$\\rm k_d$', '$\\rm \hat \sigma$', '$\\rm \\alpha_0$' ]


sens = np.load( '../Simulated_Data/sensitivity_analysis.npz.npy' )
x = np.array( range(np.size(labels)) )

fig = plt.figure( figsize=(6.2,2) )
ax = plt.subplot(111)
plt.bar( x-0.2, res, width=0.5,align='center', color='0.4' )
plt.bar( x+0.2, sens, width=0.5,align='center', color='0.1' )
plt.axhline(y=0.0, linestyle='--',linewidth=0.8,color='0.3')
ax.set_xticks( x )
ax.set_yticks( [-0.15, 0.0, 0.15] )
ax.set_ylim( [-0.16, 0.16] )
ax.set_ylabel( 'sensitivity' )
ax.set_xticklabels( labels )
plt.tight_layout()
sns.despine()
#plt.savefig( 'figure_7.pdf' )
plt.show()
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



'''
server = 'http://byu.apmonitor.com'
app = 'heterogeneous_production'
apm(server, app,'clear all')
apm_load(server, app, 'heterogeneous_production.apm')
apm_option(server,app,'nlc.imode',3)
apm_option(server,app,'nlc.solver',3)
apm_option(server,app,'nlc.max_iter',1000000)
apm_option(server,app,'nlc.max_time',500)
solver_output = apm(server,app,'solve')
res = apm_sol(server,app)
'''



'''
Ieff = np.array([ 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 110.0, 
				  130.0, 150.0, 170.0, 190.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1000.0, 
				  1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1620.0, 1630.0, 1639.0 ])
Ieff = np.array([ 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 27.5, 30.0, 35.0, 40.0, 
				  45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0, 190.0, 220.0, 275.0, 
				  330.0, 385.0, 440.0, 495.0, 550.0, 605.0, 660.0, 715.0, 770.0, 800.0 ])
Ieff = np.array([ 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 27.5, 30.0, 35.0, 40.0, 
				  45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0, 190.0, 220.0, 275.0, 
				  330.0, 385.0, 400.0, 405.0, 406.0 ])
'''

