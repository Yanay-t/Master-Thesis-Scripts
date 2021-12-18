
#Purpose: Calculate f1,2,3(omega, Omega, Zbar)'s average over Zbar
#Previous script results required: sigma_crit.npy 
#input: mu, lambda, sigma_crit for each lambda, d_sigma
#output: omega, f1(omega), f2(omega, Omega), f3(omega, Omega) files, saved in the designated HOME directory  

import numpy as np
from scipy.special import erf
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import winsound
import os

###################################################
#Functions needed for finding Zbar moments m_z, s_z at lambda=0
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
#Functions need for finding Zbar moments m_z, s_z at lambda>0
###################################################

def NZ_fun(Z, lam):
    return (Z + np.sqrt(Z**2 + 4*lam))/2

def PZ_fun(Z, sig_z, m_z):
    return 1/(np.sqrt(2*np.pi)*sig_z)*np.exp(-(Z - m_z)**2/(2*sig_z**2))

def m_z_of_Z_fun(sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*NZ_fun(Z, lam)
    return 1 - mu*integrate.quad(f, -np.inf, np.inf)[0]
    # return 1 - mu*integrate.quad(PZ_fun(Z)*NZ_fun(Z), -np.inf, np.inf, args=(sig_z, m_z))

def sig_z2_of_Z_fun(sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*(NZ_fun(Z, lam))**2
    return sig**2*integrate.quad(f, -np.inf, np.inf)[0]
    # return sig**2*integrate.quad(PZ_fun(Z)*(NZ_fun(Z))**2, -np.inf, np.inf, args=(sig_z, m_z))

def fun_vec(p, lam):
    m_z, sig_z = p
    x = m_z_of_Z_fun(sig_z, m_z, lam) - m_z
    y = sig_z2_of_Z_fun(sig_z, m_z, lam) - sig_z**2
    return (x,y)

###################################################
#Fucntions needed for calculating f1 f2 f3
###################################################

def A(z, lam):
    return np.sqrt(z**2 + 4*lam)

def B(z, lam):
    return 0.5*(z + np.sqrt(z**2 + 4*lam))

def f1(z, lam, w):
    return B(z, lam)**2/(A(z, lam)**2 + w**2)

def f2(z, lam, w, W):
    return B(z, lam)**2*(2*(A(z, lam)-B(z, lam))**2 + 0.5*w**2)/((A(z, lam)**2+ w**2)*(A(z, lam)**2 + W**2)*(A(z, lam)**2 + (w+W)**2))

def f3(z, lam, w, W):
    # return B(z, lam)**2*(6*A(z, lam)**5 - 16*A(z, lam)**4*B(z, lam) + 10*A(z, lam)**3*B(z, lam)**2 + 2*A(z, lam)**3*(2*w**2 + w*W + W**2) - 2*A(z, lam)**2*B(z, lam)*(w**2 + 2*W**2) + 2*A(z, lam)*B(z, lam)**2*(-3*w**2 - 2*w*W + W**2) + 2*(A(z, lam) - B(z, lam))*w**2*(w+W)**2)/(A(z, lam)*(A(z, lam)**2 + w**2)**2*(A(z, lam)**2 + W**2)*(A(z, lam)**2 + (w+W)**2))
    AminB = A(z,lam)-B(z,lam)
    f3_1 = B(z, lam)**2*(4*A(z,lam)**2*(AminB**2 - B(z,lam)*AminB) - w*W*(2*AminB**2-4*B(z,lam)*AminB + 2*B(z,lam)**2) + 4*w**2*(B(z,lam)*AminB - B(z,lam)**2)) / ((A(z, lam)**2 + w**2)**2*(A(z, lam)**2 + W**2)*(A(z, lam)**2 + (w+W)**2)) 
    f3_2 = 2*B(z, lam)**2*(AminB**3 - B(z,lam)**2*AminB + w**2*AminB) / (A(z,lam)*(A(z,lam)**2 + w**2)**2*(A(z,lam)**2 + W**2))
    return f3_1 + f3_2

###################################################
#Setting system parameters
###################################################

sig_crit = np.load(r'C:\...\sigma critical\mu=10\sigma_crit, accuracy=1e-06.npy')

lam_ind = 2 #lambda vector index

mu = 10
dsig = 10**-2 #distance from sigma_crit

sig = sig_crit[lam_ind] + dsig

###################################################
#calculating sig_z and m_z at lambda=0
###################################################

delta = np.append(np.arange(-5, 10, 0.02), 1000)
delta_intrp = interp1d(omega2(delta), delta, kind='linear')
delta_now = delta_intrp(1/sig**2) 

mz_lam0 = mz(delta_now, mu)
sz_lam0 = sz(delta_now, mu)

###################################################
#calculating sig_z and m_z at lambda>0
###################################################

pwr = np.arange(6,0,-1)
lam_all = np.concatenate(([0], np.power(0.1, pwr)))
m_z = np.zeros(len(lam_all))
sig_z = np.zeros(len(lam_all))

m_z[0] = mz_lam0
sig_z[0] = sz_lam0

for i in range(1, len(lam_all)):
    lam = lam_all[i]
    init_guess = m_z[i-1], sig_z[i-1]
    sol = fsolve(fun_vec, init_guess, args=(lam))
    m_z[i] = sol[0]
    sig_z[i] = sol[1]

###################################################
#calculating f1,2,3
###################################################

lam = lam_all[lam_ind]
m_z = m_z[lam_ind]
sig_z = sig_z[lam_ind]

print('$\lambda={}, \mu={}, \sigma - \sigma_c={}$'.format(lam, mu, dsig))

###Define Zbar sample points###

inner_bndry_amp = 0.99 #decide based on previous tuning
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

###Define omega sample points###

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

#for high sigma values
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

w1 = np.round(np.arange(-wmax, -wmid4, dw5), 2)
w2 = np.round(np.arange(-wmid4, -wmid3, dw4), 3)
w3 = np.round(np.arange(-wmid3, -wmid2, dw3), 3)
w4 = np.round(np.arange(-wmid2, -wmid1, dw2), 3)
w5 = np.round(np.arange(-wmid1, wmid1, dw1), 5)
w6 = np.round(np.arange(wmid1, wmid2, dw2), 3)
w7 = np.round(np.arange(wmid2, wmid3, dw3), 3)
w8 = np.round(np.arange(wmid3, wmid4, dw4), 3)
w9 = np.round(np.arange(wmid4, wmax+dw5, dw5), 2)
if w1[-1] >= -wmid4:
    w1 = w1[:-1]
if w2[-1] >= -wmid3:
    w2 = w2[:-1]    
if w3[-1] >= -wmid2:
    w3 = w3[:-1]
if w4[-1] >= -wmid1:
    w4 = w4[:-1]
if w5[-1] >= wmid1:
    w5 = w5[:-1]
if w6[-1] >= wmid2:
    w6 = w6[:-1]
if w7[-1] >= wmid3:
    w7 = w7[:-1]
if w8[-1] >= wmid4:
    w8 = w8[:-1]
if w9[-1] >= wmax+dw5:
    w9 = w9[:-1]
w = np.concatenate((w1, w2, w3, w4, w5, w6, w7, w8, w9))

W = w

lw = len(w)
lW = len(W)

intrp_x = 2 #interpolation grid multiplier
HOME = r'C:\...\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}'.format(lam, mu, dsig, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5)

f1_zavg = np.zeros(lw)
f2_zavg = np.zeros((lW, lw))
f3_zavg = np.zeros((lW, lw))

for i in range(lw):
    temp1 = PZ_fun(z, sig_z, m_z)*f1(z, lam, w[i])
    f1_zavg[i] = np.trapz(temp1, z)
    for j in range(lW):
        temp2 = PZ_fun(z, sig_z, m_z)*f2(z, lam, w[i], W[j])
        f2_zavg[j, i] = np.trapz(temp2, z)
        temp3 = PZ_fun(z, sig_z, m_z)*f3(z, lam, w[i], W[j])
        f3_zavg[j, i] = np.trapz(temp3, z)
    print('{}/{}'.format(i+1, lw))
        
if not os.path.exists(HOME):
    os.makedirs(HOME)
np.save(HOME +'\\f1', f1_zavg)
np.save(HOME +'\\f2', f2_zavg)
np.save(HOME +'\\f3', f3_zavg)
np.save(HOME +'\\w', w)
    
#Plot f1,2,3
# w_mesh, W_mesh = np.meshgrid(w, w)        

# fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
# plt.plot(w, f1_zavg)
# plt.title('f1')

# fig, ax = plt.subplots()
# pc1 = ax.pcolormesh(w_mesh,W_mesh,f2_zavg, shading='nearest')
# fig.colorbar(pc1)
# plt.xlabel('$\omega$')
# plt.ylabel('$\Omega$')
# plt.title('f2')
# plt.show()

# fig, ax = plt.subplots()
# pc2 = ax.pcolormesh(w_mesh,W_mesh,f3_zavg, shading='nearest')
# fig.colorbar(pc2)
# plt.xlabel('$\omega$')
# plt.ylabel('$\Omega$')
# plt.show()

print('Calculation of f1, f2, f3 is complete')
    
###############################################
#Finish tone
###############################################
duration = 1000 #millisec
freq = 440 #Hz
winsound.Beep(freq, duration)