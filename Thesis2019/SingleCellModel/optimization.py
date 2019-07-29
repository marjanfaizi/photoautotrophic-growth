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
	df.to_csv(output_filename+'.csv',index=0)
	return df
	
	
## SENSITIVITY ANALYSIS
def change_par_opt( model_name, par, par_name, light_input ):
	## connect to server and create model
	server, app = connect( model_name )
	## change light intensity
	apm_info( server, app, 'FV', 'i' )	
	apm_meas( server, app, 'i', light_input )
	## designate parameters to change
	apm_info( server, app, 'FV', par_name )	
	apm_meas( server, app, par_name, par )
	## solve on APM server
	solver_output = apm( server, app, 'solve' )
	res = apm_sol( server, app )
	print (res['i'])
	counter = 0
	while ( (not apm_tag(server, app, 'nlc.appstatus')) and (counter<10) ):
		solver_output = apm( server, app, 'solve' )
		res = apm_sol( server, app )
		counter += 1
		print (counter)
			
	return res['growth_rate']

def sens_P( model_name, dist_par, light_input, dist=0.0001 ):
	# increase/decrease one paramter for 0.1%
	p_fw = np.fromiter( dist_par.values(), dtype=float ) * ( 1.0 + dist )
	p_bw = np.fromiter( dist_par.values(), dtype=float ) - np.fromiter( dist_par.values(), dtype=float ) * dist
	dist_name = list( par.keys() )
	delta = np.zeros( np.size(dist_name) )
	# iterate over each parameter
	for i in range( np.size(dist_name) ):
		# calculate the new controlled value: productivity
		c_fw = change_par_opt( model_name, p_fw[i], dist_name[i], light_input  )
		c_bw = change_par_opt( model_name, p_bw[i], dist_name[i], light_input )
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

light = np.array([ 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 40.0, 55.0, 70.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0,
                   190.0, 220.0, 330.0, 440.0, 550.0, 660.0, 770.0, 880.0, 990.0, 1000.0, 1100.0, 1200.0 ])
'''
cix = np.array([ 0.001, 0.005, 0.01,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 
				30.0, 50.0, 70.0, 100.0, 500.0, 1000.0, 10000.0, 100000.0 ])	
'''		
input_name = 'i'
input_value = light
input_value = input_value[::-1]
## header (column names) for csv file
single_var = [ 'beta_et', 'beta_em', 'beta_er', 'beta_psu', 'beta_pq' ]
single_S = [ 'ci', 'aa', 'et', 'em', 'er', 'psu0', 'psu1', 'e' ]
single_flux = [ 'vet', 'vem', 'ver', 'v1', 'v2', 'vi', 'sigma' ]
single_cell_header = [ input_name, 'mu' ] + single_var + single_S + single_flux
## save result
output_single = np.zeros([np.size(input_value),len(single_cell_header)])
## run optimization
single_cell_results = apm_optimize( 'single_cell_model', input_value, input_name, single_var, single_S, single_flux,
                                     output_single, single_cell_header, 'simulation100719' )
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
########################################## SENSITIVITY ANALYSIS #####################################################
#####################################################################################################################
par = { 'kcat_t':43560.0, 'K_t':15.0, 'kcat_m':32700.0, 'Km':456702.458, 'gamma_max':79200.0, 'K_a':1.0e4, 
		'Ke':1.0e4, 'dp':0.04347826, 'vme':7.0e9, 'tau':360000.0, 'kd':1.2e-6, 'sigma':6.0, 'i':10.0, 'cix':1.0e5 }	
model_name = 'single_cell_model'

	
res = np.zeros([ len(light), len(par) ])
for i in range( len(light) ):
	print ('##########'+str(i)+'##########')
	par['i'] = light[i]
	res[i,:] = sens_P( model_name, par, light[i] )

np.save( 'sensitivity_analysis_high_cix.npz', res )


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
