
#Purpose: find sigma_crit for given mu and lambda
#Previous script results required: none
#input: mu, lambda(vector), accuracy 
#output: sigma_crit(vector)

#method:

#1 find rough sigma_crit:
    #a. for a given lambda, calculate f1(w=0) for a range of sigmas
    #b. Interpolate f1(w=0) sample to find sigma where f1(w=0)=1/sigma^2

#2 Refine  sigma_crit:
    #a. For each sigma_crit(lambda) value, define neighboring points (sigma_crit - ac, sigma_crit + ac)
    #b. Calculate f1(w=0, sigma) for neighboring points and find sigma_crit_new f1(w=0, sigma)=1/sigma^2 through interpolation
    #c. If |f1(w=0, sigma) - 1/sigma^2| < accuracy, finish. Otherwise, 

import numpy as np
from scipy.special import erf
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import winsound

###################################################
#Functions needed for finding m_z, s_z at lambda=0
###################################################

def omega0(x):
    return 0.5*(1+erf(x/2**0.5))

def omega1(x):
    return (2*np.pi)**(-0.5)*np.exp(-x**2/2) + x*omega0(x)

def omega2(x):
    return omega0(x) + x*omega1(x)

def mz(x, mu):
    return (1 + mu*omega1(x)/x)**-1

def sz(x, mu):
    # return mz(x,mu)/x
    return (1 - mz(x, mu))/(mu*omega1(x)) 

###################################################
#Functions need for finding m_z, s_z at lambda>0
###################################################

def NZ_fun(Z, lam):
    return (Z + np.sqrt(Z**2 + 4*lam))/2

def PZ_fun(Z, sig_z, m_z):
    return 1/(np.sqrt(2*np.pi)*sig_z)*np.exp(-(Z - m_z)**2/(2*sig_z**2))

def m_z_of_Z_fun(mu, sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*NZ_fun(Z, lam)
    return 1 - mu*integrate.quad(f, -np.inf, np.inf)[0]
    # return 1 - mu*integrate.quad(PZ_fun(Z)*NZ_fun(Z), -np.inf, np.inf, args=(sig_z, m_z))

def sig_z2_of_Z_fun(sig, sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*(NZ_fun(Z, lam))**2
    return sig**2*integrate.quad(f, -np.inf, np.inf)[0]
    # return sig**2*integrate.quad(PZ_fun(Z)*(NZ_fun(Z))**2, -np.inf, np.inf, args=(sig_z, m_z))

def fun_vec(p, lam, mu, sig):
    m_z, sig_z = p
    x = m_z_of_Z_fun(mu, sig_z, m_z, lam) - m_z
    y = sig_z2_of_Z_fun(sig, sig_z, m_z, lam) - sig_z**2
    return (x,y)

###################################################
#Fucntions needed for calculating f1
###################################################

def A(z, lam):
    return np.sqrt(z**2 + 4*lam)

def B(z, lam):
    return 0.5*(z + np.sqrt(z**2 + 4*lam))

def f1(z, lam, w):
    return B(z, lam)**2/(A(z, lam)**2 + w**2)

###################################################
#calculating sig_z and m_z at lambda=0
###################################################

#finds mean and variance of Zbar for lambda=0 at given sigma amd mu
#input: sigma, mu 
#output: Zbar mean and standard dev for lambda=0

def Zbar_stats_lam0(sig, mu):
    
    delta = np.append(np.arange(-5, 10, 0.02), 1000)
    delta_intrp = interp1d(omega2(delta), delta, kind='linear')
    delta_now = delta_intrp(1/sig**2) 
    
    mz_lam0 = mz(delta_now, mu)
    sz_lam0 = sz(delta_now, mu)
    
    return mz_lam0, sz_lam0 

###################################################
#calculating sig_z and m_z at lambda>0
###################################################

#finds mean and variance of Zbar for lambda>0 for give sigma, mu, lambda>0 values. 
#input: sigma, mu, lambda(vector)
#lambda vector must start with 0, be ordered from small to large and increase gradually for funciton to work.

#output: vectors of mean and standard dev. of Zbar for each lambda. m_z[i] and sig_z[i] correspond with given lam_all[i] value. 

def Zbar_stats(sig, mu, lam_all):
    
    m_z = np.zeros(len(lam_all))
    sig_z = np.zeros(len(lam_all))
    
    mz_lam0, sz_lam0 = Zbar_stats_lam0(sig, mu)
    
    m_z[0] = mz_lam0
    sig_z[0] = sz_lam0
    
    for i in range(1, len(lam_all)):
        lam = lam_all[i]
        init_guess = m_z[i-1], sig_z[i-1]
        sol = fsolve(fun_vec, init_guess, args=(lam, mu, sig))
        m_z[i] = sol[0]
        sig_z[i] = sol[1]
    
    return m_z, sig_z

###################################################
#Generate Zbar vector to integrate f1,2,3
###################################################

#Creates a Zbar vector to calculate the Zbar average of f1,2,3
#The range and density of Zbar sample points is modulated by the mean and standard dev. of Zbar probability density.

def Zbar_vector(m_z, sig_z):
    
    inner_bndry_amp = 0.99 
    middle_bndry_amp = 0.1
    outer_bndry_amp = 10**-6
    
    z_in_1 = -np.sqrt(2*sig_z**2*np.log(1/inner_bndry_amp)) + m_z
    z_in_2 = np.sqrt(2*sig_z**2*np.log(1/inner_bndry_amp)) + m_z
    z_mid_1 = -np.sqrt(2*sig_z**2*np.log(1/middle_bndry_amp)) + m_z
    z_mid_2 = np.sqrt(2*sig_z**2*np.log(1/middle_bndry_amp)) + m_z
    z_out_1 = -np.sqrt(2*sig_z**2*np.log(1/outer_bndry_amp)) + m_z
    z_out_2 = np.sqrt(2*sig_z**2*np.log(1/outer_bndry_amp)) + m_z
    
    dz_in = 10**-4
    dz_mid = 10**-3
    dz_out = 10**-1
    
    z1 = np.arange(z_out_1, z_mid_1, dz_out)
    z2 = np.arange(z_mid_1, z_in_1, dz_mid)
    z3 = np.arange(z_in_1, z_in_2, dz_in)
    z4 = np.arange(z_in_2, z_mid_2, dz_mid)
    z5 = np.arange(z_mid_2, z_out_2, dz_out)
    #to fix a bug that sometimes causes np.arange to include the end point when using steps of type float
    if z1[-1] == z_mid_1:
        z1 = z1[:-1]
    if z2[-1] == z_in_1:
        z2 = z2[:-1]
    if z3[-1] == z_in_2:
        z3 = z3[:-1]
    if z4[-1] == z_mid_2:
        z4 = z4[:-1]
    if z5[-1] == z_out_2:
        z5 = z1[:-1]
    
    z = np.concatenate((z1, z2, z3, z4, z5))
    
    return z

###################################################
#Setting system parameters
###################################################

mu = 10

pwr = np.arange(6,0,-1)
lam_all = np.concatenate(([0], np.power(0.1, pwr)))
llam = len(lam_all)

accuracy = 10**-6

###################################################
#Find rough sigma_crit
###################################################

dsig = 0.05
sig_arr = np.arange(1.3, 3, dsig) #suggested sigma values to scan  
lsig = len(sig_arr)

sigma_crit = np.zeros(llam)

print('step 1: find rough sigma_crit\nlambdas calculated')
for lam_ind in range(llam):
        
    print('{}/{}'.format(lam_ind+1, llam))
    
    f1_zavg_w0 = np.zeros(lsig)
    
    for j in range(lsig):
    
        sig = sig_arr[j]
        
        m_z, sig_z = Zbar_stats(sig, mu, lam_all[:lam_ind+1])
        
        lam = lam_all[lam_ind]
        m_z = m_z[lam_ind]
        sig_z = sig_z[lam_ind]
          
        #generate Z_bar vector to average f1(w=0) over
        z = Zbar_vector(m_z, sig_z)
        
        #calculate f1(w=0)
        temp1 = PZ_fun(z, sig_z, m_z)*f1(z, lam, 0)
        f1_zavg_w0[j] = np.trapz(temp1, z)
    
    #for debugging - make sure f1_zavg_w0 - 1/sig_arr**2 is present in each graph
    # plt.figure()
    # plt.plot(sig_arr, f1_zavg_w0 - 1/sig_arr**2, 'o')
    # plt.title('$\lambda={}$'.format(lam))
    
    sigma_intrp = interp1d(f1_zavg_w0 - 1/sig_arr**2, sig_arr)
    sigma_crit[lam_ind] = sigma_intrp(0)

plt.figure()
plt.plot(lam_all, sigma_crit, 'o')
plt.title('$\sigma_{crit}$')
plt.xlabel('$\lambda$')
plt.xscale('log')

###################################################
#Refine sigma_crit to desired accuracy
###################################################

print('step 2: Refine sigma_crit values')

sig_crit = np.zeros(llam)

for lam_ind in range(llam):
    
    accuracy_flag = 1
    lam = lam_all[lam_ind]
    ac = dsig*2
    
    s_crit_new = sigma_crit[lam_ind]
    s_crit_old = 0
    
    print('lambda={}'.format(lam_all[lam_ind]))
    
    #for debugging
    # s_crit_arr = sigma_crit[lam_ind]
    
    # while abs(s_crit_new - s_crit_old) > accuracy:
    while accuracy_flag:
        
        ac = ac*0.1
        sig = np.array([s_crit_new - ac/2, s_crit_new + ac/2])
    
        f1_zavg_w0 = np.zeros(2) 
    
        for sig_ind in range(2):
    
            m_z, sig_z = Zbar_stats(sig[sig_ind], mu, lam_all[:lam_ind+1])


            m_z = m_z[lam_ind]
            sig_z = sig_z[lam_ind]
                
            #generate Z_bar vector to average f1(w=0) over
            z = Zbar_vector(m_z, sig_z)
    
            temp1 = PZ_fun(z, sig_z, m_z)*f1(z, lam, 0)
            f1_zavg_w0[sig_ind] = np.trapz(temp1, z)
        
        s_crit_old = s_crit_new
        s_crit_new_intrp = interp1d(f1_zavg_w0 - sig**(-2), sig)
        s_crit_new = s_crit_new_intrp(0)
        
        # s_crit_arr = np.append(s_crit_arr, s_crit_new)
        
        print('|sig_new - sig_old|={}'.format(np.abs(s_crit_new - s_crit_old)))
        
        #test accuracy of s_crit_new
        m_z1, sig_z1 = Zbar_stats(s_crit_new, mu, lam_all[:lam_ind+1])
        m_z1 = m_z1[lam_ind]
        sig_z1 = sig_z1[lam_ind]
        z1 = Zbar_vector(m_z1, sig_z1)     
        temp2 = PZ_fun(z1, sig_z1, m_z1)*f1(z1, lam, 0)
        f1_zavg_w0_1 = np.trapz(temp2, z1)
        print('f1(w=0) - s^-2 = {}'.format(np.abs(f1_zavg_w0_1 - 1/s_crit_new**2)))
        
        if np.abs(f1_zavg_w0_1 - 1/s_crit_new**2) < accuracy:
            accuracy_flag = 0
        
    sig_crit[lam_ind] = s_crit_new

x = np.arange(10**-6, 0.1, 10**-6)

plt.figure()
plt.plot(lam_all[0:], sig_crit[0:] - sig_crit[0], 'o')
plt.plot(x, 7*x**0.5, label='$7 \sqrt{\lambda}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\lambda$')
plt.title('$\sigma_c - \sigma_c(\lambda=0)$')
plt.legend()

###############################################
#Save results 
###############################################

#Fill in directory name#

# np.save(r'C:\...\sigma critical\mu={}\sigma_crit.npy'.format(mu), sigma_crit)
# np.save(r'C:\...\sigma critical\mu={}\lam_all.npy'.format(mu), lam_all)

###############################################
#Finish tone
###############################################
duration = 1000 #millisec
freq = 440 #Hz
winsound.Beep(freq, duration)

    
    
    