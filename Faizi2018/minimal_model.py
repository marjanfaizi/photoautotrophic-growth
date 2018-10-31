#from numba import jit

#@jit
# ODE model of phototrophic growth
def PhotoMini(S,t,beta,I,cix,p):
    ## Define species (initial conditions)
	ci = S[0]
	aa = S[1]
	R = S[2]
	T = S[3]
	M = S[4]
	P = S[5]
	e = S[6]

    ## Define parameters
    # fixed parameters
	Acell = 3.85e-9 # cell surface area
	Vcell = 2.24e-14 # cell volume
	N = 6.022e23 # avogadro constant
	Ki = 1.0e8 # e inhibition constant
	Mcell = 1.4e10 # total cell mass
	nr = 7358.0 # ribosome length
	nt = 1681.0 # length of bicarbonate transporter
	nm = 28630.0 # length of metabolic enzyme complexnt
	np = 95451.0 # length of photosystem unit
	factor = 60.0 # conversion factor between seconds and min
	micro = 1.0e-6 # metric prefix

	# stoichiometric coefficients
	mmu = 45.0 # molecules of e consumed to create one aa
	mgamma = 3.0 # molecules of e for one translational elongation step
	mphi = 8.0 # molecules of e produced via photosynthesis
	mc = 5.0 # average carbon chain length

    # variable parameters
	Pm = p[0]*factor # permeability of the cell membrane
	kcat_t = p[1]*factor # maximal import rate
	Kt = p[2] # inorganic carbon import threshold
	kcat_m = p[3]*factor # maximal carbon fixation rate
	Km = p[4] # carbon fixation threshold
	gamma_max = p[5]*factor # maximal transl. elongation rate
	Ka = p[6] # e and aa threshold
	sigma = p[7] # effective cross-section of photosystem
	k2 = (p[8]*factor)/(1.0+(e/Ki)**4) # turnover rate of photosystem
	kd = p[9] # photodamage rate

    # gene regulatory parameters
	beta_r = beta[0]
	beta_t = beta[1]
	beta_m = beta[2]
	beta_p = beta[3]

    ## Define rate equations
	vd = Pm*Acell*((micro*N*cix)-(ci/Vcell))
	vt = T*kcat_t*(cix/(Kt+cix))*(e/(Ka+e))
	vm = M*kcat_m*(ci/(Km+ci))*(e/(Ka+e))
	gamma = R*gamma_max*(e/(Ka+e))*(aa/(Ka+aa))
	gr = (vd+vt)/(Mcell*mc)
	alpha = (sigma*I*factor*factor)/(sigma*I*factor*factor+k2+kd*sigma*I*factor*factor+gr)
	v2 = k2*alpha*P
	vi = kd*sigma*I*factor*factor*alpha*P

    ## Define differential equations
	cid = vd + vt - mc*vm - gr*ci
	aad = vm - gamma*sum([beta_r,beta_t,beta_m,beta_p]) + np*vi - gr*aa
	Rd = beta_r*(gamma/nr) - gr*R
	Td = beta_t*(gamma/nt) - gr*T
	Md = beta_m*(gamma/nm) - gr*M
	Pd = beta_p*(gamma/np) - vi - gr*P
	ed = mphi*v2 - gamma*mgamma*sum([beta_r,beta_t,beta_m,beta_p]) - mmu*vm - vt - gr*e

	return [cid,aad,Rd,Td,Md,Pd,ed]



