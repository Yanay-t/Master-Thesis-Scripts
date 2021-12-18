
#Purpose: Compile all the solutions to eq (44) found for all lambda and sigma, calculate Q, the time scale and plot graphs
#Previous script results required: 
    #1. Solving C_dZ.npy for all lambda and sigma that is needed
    #2. sigma_crit.npy
#Input:
    #1.  mu, lambda, sigma_crit, d_sigma
    #2. omega vector parameters dw1, dw2,..dw5, wmid,...wmid4, wmax
#Output:
    #graphs:
    #1. Q vs sigma
    #2. Q vs sigma-sigma_crit
    #3. Time scale C(omega=0)/Q vs sigma-sigma_crit
    #4. Time scale C(omega=0)/Q*lambda^0.25 vs sigma-sigma_crit
    #5. Q vs sigma-sigma_crit with error estimation
    #6. C_n(t)/C_n(t=0) for each lambda
    #7. C_n(t)/C_n(t=0) vs t/[fit of time scale]
    #8. C_n(t)/C_n(t=0) vs t/[time scale]
    #9. C_n(t)/C_n(t=0) vs t/[time scale] log scale


import numpy as np
import matplotlib.pyplot as plt

def gauss(x, a):
    return np.exp(-0.5*(x/a)**2)

##################################################
# Setting/loading parameters 
##################################################

#lambda
pwr = np.arange(6,0,-1)
lam_all = np.concatenate(([0], np.power(0.1, pwr)))
lam_ind_slct = [2, 3, 4]
llam_slct = len(lam_ind_slct)

#critical sigma
sig_crit = np.load(r'C:\...\sigma critical\mu=10\sigma_crit, accuracy=1e-06.npy') #loading sigma_crit results
sig_crit = sig_crit[lam_ind_slct] #leave only sig_crit of lam=10^-5, 10^-4, 10^-3

#dsig (sigma = dsig + sig_crit)
dsig_list = np.array([0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
lsig = len(dsig_list)

mu = 10

a=2
dw1 = a*10**-4
dw2 = a*10**-3
dw3 = a*10**-2
dw4 = a*10**-1
dw5 = a*10**-0

wmid1 = 0.015
wmid2 = 0.065
wmid3 = 0.25
wmid4 = 0.75
wmax = 5.75

#for high sigma values (dsig>1)
# dw1 = a*10**-4
# dw2 = a*10**-3
# dw3 = a*10**-3
# dw4 = a*10**-2
# dw5 = a*10**-1

# wmid1 = 0.015
# wmid2 = 0.065
# wmid3 = 0.25
# wmid4 = 3.75
# wmax = 5.75

#loading a interpolated w vector, just to get its length (lw_intrp)
w_intrp = np.load(r'C:\...\Solving for C_dZ\lamda=1.0000000000000003e-05\mu=10\dsigma=0.0001\wmid1=0.015, wmid2=0.065, wmid3=0.25, wmid4=0.75, wmax=5.75, dw1=0.0002, dw2=0.002, dw3=0.02, dw4=0.2, dw5=2\w_intrp.npy')
lw_intrp = len(w_intrp)

##################################################
# Loading all C_dZ solutions onto one matrix
##################################################

C_arr = np.zeros((llam_slct, lsig, lw_intrp))

for j in range(llam_slct):
    for i in range(lsig):
        
        HOME = r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}'.format(lam_all[lam_ind_slct[j]], mu, dsig_list[i], wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5)
        C_arr[j, i, :] = np.load(HOME + '\\C_conv_sol.npy') 

##################################################    
# Calculating Q and Time Scale C(omega=0)/Q
##################################################

sig = dsig_list + sig_crit.reshape(-1,1) 

Q = sig**(-2)*np.trapz(C_arr, x=w_intrp, axis=2)
Cw0_div_Q = sig**(-2)*C_arr[:, :, int(lw_intrp/2)+1]/Q

##################
#graphs
##################

###Q vs sigma###
plt.figure()
for i in range(3):
    plt.plot(sig[i, :], Q[i, :], '-o', label='$\lambda={}$'.format(lam_all[lam_ind_slct[i]]))
plt.legend()
plt.title('Q vs $\sigma$')
plt.yscale('log')
plt.xlabel('$\sigma$')

###Q vs sigma-sigma_crit###
plt.figure()
for i in range(3):
    plt.plot(dsig_list, Q[i, :], 'o', label='$\lambda={}$'.format(lam_all[lam_ind_slct[i]]))
plt.title('Q vs $\sigma-\sigma_C$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\sigma - \sigma_C$')
plt.legend()

###C(omega=0)/Q vs sigma-sigma_crit###
plt.figure()
for j in range(llam_slct):
    plt.plot(dsig_list, Cw0_div_Q[j, :], 'o', label='$\lambda={}$'.format(np.round(lam_all[lam_ind_slct[j]],5 )))
    plt.plot(dsig_list, 1.4*dsig_list**-0.5 * lam_all[lam_ind_slct[j]]**(-1/4), label='$1.4(\sigma - \sigma_C)^{-0.5} \cdot \lambda^{-0.25}, \lambda=$ %f' % lam_all[lam_ind_slct[j]])
    # plt.plot(dsig_new[j, :], dsig_new[0, :]**-0.5 * lam_all[lam_ind_slct[j]]**(-0.3), label='$(\sigma - \sigma_C)^{-0.5} \cdot \lambda^{-0.3}$')
plt.title(r'$\tilde{C} _n (\omega=0) / Q$ vs $\sigma - \sigma_C$')
plt.xlabel('$\sigma - \sigma_C$')
plt.xscale('log')
plt.yscale('log')
plt.legend()

###C(omega=0)/Q*lambda^0.25 vs sigma-sigma_crit###
plt.figure()
for j in range(llam_slct):
    plt.plot(dsig_list, Cw0_div_Q[j, :]/lam_all[lam_ind_slct[j]]**(-0.25), 'o', label='$\lambda={}$'.format(lam_all[lam_ind_slct[j]]))
plt.title('$C(\omega=0)/Q \cdot \lambda^{0.25}$')
plt.xlabel('$\sigma - \sigma_C$')
plt.xscale('log')
plt.yscale('log')
plt.legend()


###Q vs sigma-sigma_crit with error estimation###
Q_stdev = np.zeros((llam_slct, lsig))

plt.figure()
for j in range(llam_slct):
    for i in range(lsig):
        
        HOME = r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}'.format(lam_all[lam_ind_slct[j]], mu, dsig_list[i], wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5)
        QvsCycle = np.load(HOME + r'\Q vs cycle.npy')
        Q_stdev[j, i] = np.std(QvsCycle[-10:])
        
    plt.errorbar(dsig_list, Q[j, :], yerr=Q_stdev[j, :], fmt='o', label='$\lambda={}$'.format(lam_all[lam_ind_slct[j]]))

plt.xlabel('$\sigma-\sigma_c$')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.title('Q')

np.savetxt(HOME+'\\Q', Q)
np.savetxt(HOME+'\\dsig_list', dsig_list)
np.savetxt(HOME+'\\lam_list', lam_all[lam_ind_slct])

###################################################
#Inverse FFT of solution to get C(t) 
###################################################

# #naive IFFT
t1 = np.arange(-5000, 5000, 1)
lt1 = len(t1)
C_n_t = np.zeros((llam_slct, lsig, lt1))
print('lt={}'.format(lt1))

for lam_ind in range(llam_slct):
    plt.figure()
    for k in range(lsig):
        for i in range(lt1):
            
            C_n_t[lam_ind, k, i] = 1/sig[lam_ind, k]**(-2)*np.trapz(np.exp(1j*w_intrp*t1[i])*C_arr[lam_ind ,k, :], x=w_intrp)
            if i%1000 == 0:
                print('$\lambda$: {}, $\sigma$: {}/{}, time: {}/{}'.format(lam_all[lam_ind_slct[lam_ind]], k, lsig-1, i, lt1))

        plt.plot(t1, C_n_t[lam_ind, k, :]/np.max(C_n_t[lam_ind, k,:]), label='$\sigma-\sigma_c={}$'.format(dsig_list[k]))

    plt.xlabel('t')
    plt.title('C_n(t)/max(C_n)')
    plt.legend()
# plt.xscale('log')
# plt.yscale('log')

##################
#graphs
##################

x = np.arange(-10, 10, 0.1)

#$C_n(t*)/C_n(0) vs t/[fit of time scale]
plt.figure()
for lam_ind in range(llam_slct):
    for k in range(lsig):
        
        plt.plot(t1*(dsig_list[k]**0.5*lam_all[lam_ind_slct[lam_ind]]**0.25)/1.4, C_n_t[lam_ind, k, :]/np.max(C_n_t[lam_ind, k,:]), label='$\sigma-\sigma_c={}$'.format(np.round(dsig_list[k],5)))

    plt.xlim(-10,10)
    plt.xlabel(r'$t*= \frac{t}{1.4(\sigma - \sigma_C)^{-0.5} \cdot \lambda^{-0.25}}$', fontsize=14)
plt.title('$C_n(t*)/C_n(0)$, all $\lambda$')

#$C_n(t*)/C_n(0) vs t/[time scale]
plt.figure()
for lam_ind in range(llam_slct):
    for k in range(lsig):
 
         plt.plot(t1/Cw0_div_Q[lam_ind, k], C_n_t[lam_ind, k, :]/np.max(C_n_t[lam_ind, k,:]))
         
plt.plot(x, gauss(x, 2.4), '--', label=r'exp($ - \frac{1}{2} \cdot \frac{(t*)^2}{2.4^2})$')
plt.xlabel(r'$t*= t / [C_n(\omega = 0)/Q]$')
plt.xlim(-10,10)
plt.title('$C_n(t*)/C_n(0)$, all $\lambda$')
plt.legend()  
    
#$C_n(t*)/C_n(0) vs t/[time scale] log scale
plt.figure()
for lam_ind in range(llam_slct):
    for k in range(lsig):
         plt.plot(t1/Cw0_div_Q[lam_ind, k], C_n_t[lam_ind, k, :]/np.max(C_n_t[lam_ind, k,:]))
plt.plot(x, gauss(x, 2.4), '--', label=r'exp($ - \frac{1}{2} \cdot \frac{(t*)^2}{2.4^2})$')
plt.plot(x, 3*np.exp(-0.6*x), '--', label=r'3*exp(-0.6*x)', color='black', linewidth=2)
plt.xlabel(r'$t*= t / [C_n(\omega = 0)/Q]$')
plt.xlim(0.1,12)
plt.ylim(10**-4,1)
plt.title('$C_n(t*)/C_n(0)$, log scale, all $\lambda$')
# plt.xscale('log')
plt.yscale('log')
plt.legend()  