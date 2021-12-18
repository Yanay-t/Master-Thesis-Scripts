#Purpose: 
    #1.Calculating f1,2,3(Optional)
    #2.Finding numerical solution to Eq (44) in thesis
#Previous script results required: 
    #1. sigma_crit.npy
    #2. f123.npy(Optional)
    #3. Solving C_dZ(Optional - to plug a C_dZ solution with neighboring parameters into solution finder) 
#input: mu, lambda, sigma_crit for each lambda, d_sigma, f1,2,3(Optional)
#Output: 
    #1. (optional) omega, f1, f2, f3 files saved to designated HOME folder
    #2. C_sol, C_dZ(omega) solution found for given parameters
    #3. Q vs cycle, C_match vs cycle files and graphs - used to check solution convergence and quantify error
    #4. Solution vs intial guess graph
    #5. RHS vs LHS graph of eq (44) with solution inserted 

import numpy as np
from scipy.special import erf
import scipy.integrate as integrate
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import winsound
from pylab import figure
from math import log10, pow, isnan
import os
import plotly.express as px

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
#Fucntions needed for calculating C(omega) and its intial guess parameters
###################################################

def FT_lorntz(Amp, gamma, w):
    return Amp*np.exp(-np.abs(w)/gamma)

def float_scale(x):
#find the logarithmic scale of a number. Used in finding gcd of floats
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(log10(int_part)) + 1
    if magnitude >= max_digits:
        return 0
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    return int(log10(frac_digits))

def float_gcd(X):
    #find gcd of elements in a vector
    scale = np.zeros(len(X))
    for i in range(len(X)):
        scale[i] = float_scale(X[i])
    factor = pow(10, np.max(scale))
    factored_X = np.multiply(X,factor)
    return np.gcd.reduce(factored_X.astype(int))/factor

def func_shifter_1d(C, x, shift):
    #assumes function does not changes at the far edges
    
    xmax = np.max(x)
    xmin = np.min(x)
    
    x_interpolate = np.concatenate(([2*xmin], x, [2*xmax]))
    C_interpolate = np.concatenate(([C[0]], C, [C[-1]]))
    C_intrp_fun = interp1d(x_interpolate, C_interpolate)
    
    return C_intrp_fun(x+shift)

def C_shifter(C, w, W):
    #Creating C(w+W)=C(-w-W) i.e. shifting C(W)'s argument by w
    #It will be a 2D matrix with indices corresponding with the w,W meshgrid. 
    #Each column will be shifted by a different w

    lw = len(w)
    lW = len(W)
    C_shifted = np.zeros((lW, lw))
    for w_ind in range(lw):
        C_shifted[:, w_ind] = func_shifter_1d(C, W, w[w_ind])
    
    return C_shifted

###################################################
#Setting system parameters
###################################################

sig_crit = np.load(r'C:\...\sigma critical\mu=10\sigma_crit, accuracy=1e-06.npy') #loading sigma_crit results

lam_ind = 2 #index in lambda vector

mu = 10
dsig = 10**-2
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

# dz_in = 5*10**-4
# dz_mid = 5*10**-3
# dz_out = 5*10**-1

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

#HOME directory - save/load files from here
HOME = r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}'.format(lam, mu, dsig, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5)

###Calculate/load f1,2,3###

yesno = 'a'
while yesno != 'y' and yesno != 'n':
    yesno = input('Calculate f1, f2, f3 from scratch? y/n:\n')
    
print('$\sigma={}, \mu={}, \lambda={}$'.format(sig, mu, lam))
if yesno == 'y':

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
    
    
    ###Plot f1,2,3 for debugging###
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
    
else:
    f1_zavg = np.load(HOME +'\\f1.npy')
    f2_zavg = np.load(HOME +'\\f2.npy')
    f3_zavg = np.load(HOME +'\\f3.npy')
    w = np.load(HOME + '\\w.npy')
    print('f1, f2, f3 loaded')

###################################################
#Interpolating term1,2,3
###################################################

w1_intrp = np.round(np.arange(-wmax, -wmid4, dw5/intrp_x), 5)
w2_intrp = np.round(np.arange(-wmid4, -wmid3, dw4/intrp_x), 5)
w3_intrp = np.round(np.arange(-wmid3, -wmid2, dw3/intrp_x), 5)
w4_intrp = np.round(np.arange(-wmid2, -wmid1, dw2/intrp_x), 5)
w5_intrp = np.round(np.arange(-wmid1, wmid1, dw1/intrp_x), 5)
w6_intrp = np.round(np.arange(wmid1, wmid2, dw2/intrp_x), 5)
w7_intrp = np.round(np.arange(wmid2, wmid3, dw3/intrp_x), 5)
w8_intrp = np.round(np.arange(wmid3, wmid4, dw4/intrp_x), 5)
w9_intrp = np.round(np.arange(wmid4, wmax+dw5/intrp_x, dw5/intrp_x), 5)
if w1_intrp[-1] >= -wmid4:
    w1_intrp = w1_intrp[:-1]
if w2_intrp[-1] >= -wmid3:
    w2_intrp = w2_intrp[:-1]    
if w3_intrp[-1] >= -wmid2:
    w3_intrp = w3_intrp[:-1]
if w4_intrp[-1] >= -wmid1:
    w4_intrp = w4_intrp[:-1]
if w5_intrp[-1] >= wmid1:
    w5_intrp = w5_intrp[:-1]
if w6_intrp[-1] >= wmid2:
    w6_intrp = w6_intrp[:-1]
if w7_intrp[-1] >= wmid3:
    w7_intrp = w7_intrp[:-1]
if w8_intrp[-1] >= wmid4:
    w8_intrp = w8_intrp[:-1]
if w9_intrp[-1] > wmax:
    w9_intrp = w9_intrp[:-1]
w_intrp = np.concatenate((w1_intrp, w2_intrp, w3_intrp, w4_intrp, w5_intrp, w6_intrp, w7_intrp, w8_intrp, w9_intrp))
if yesno == 'y':
    np.save(HOME + '\\w_intrp.npy', w_intrp)
lw_intrp = len(w_intrp)

f1_intrp_fun = interp1d(w, f1_zavg)
f2_intrp_fun = interp2d(w, W, f2_zavg)
f3_intrp_fun = interp2d(w, W, f3_zavg)

f1_intrp = f1_intrp_fun(w_intrp)
f2_intrp = f2_intrp_fun(w_intrp, w_intrp)
f3_intrp = f3_intrp_fun(w_intrp, w_intrp)

###################################################
#Find numerical solution
###################################################

####Construct initial guess####

#general initial guess - if no solution with nearby parameters is present
Amp1 = 0.01
gamma1 = 1
C_guess = FT_lorntz(Amp1, gamma1, w_intrp)

# when changing sigma
# dsig1 = 10**-4
# C_guess = np.load(r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}\C_conv_sol.npy'.format(lam, mu, dsig1, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5))

#when changing lambda
# lam1 = 0.00010000000000000002
# C_guess = np.load(r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}\C_conv_sol.npy'.format(lam1, mu, dsig, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5))

#when omega sample point density is different between loaded guess and currect omega vector
# a1 = 2
# dw1_1 = a1*10**-4
# dw2_1 = a1*10**-3
# dw3_1 = a1*10**-2
# dw4_1 = a1*10**-1
# dw5_1 = a1*10**0

# C_guess_origin = np.load(r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}\C_conv_sol.npy'.format(lam, mu, dsig, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5))
# w_guess = np.load(r'C:\...\Solving for C_dZ\lamda={}\mu={}\dsigma={}\wmid1={}, wmid2={}, wmid3={}, wmid4={}, wmax={}, dw1={}, dw2={}, dw3={}, dw4={}, dw5={}\w_intrp.npy'.format(lam, mu, dsig, wmid1, wmid2, wmid3, wmid4, wmax, dw1, dw2, dw3, dw4, dw5))

# C_guess_intrp = interp1d(w_guess, C_guess_origin)
# C_guess = C_guess_intrp(w_intrp)

###Soft injection loop###

frac = 0.7
C = C_guess

accuracy = 10**-5
d_arr =[]
Q_arr = []
Q_err = []
# plt.figure(1)
# plt.clf()
plt.figure(2)
plt.clf()

for i in range(9*10**3):
    
    term1 = f1_intrp*C
    term2 = np.trapz(f2_intrp*C.reshape(-1,1)*C_shifter(C, w_intrp, w_intrp), w_intrp, axis=0)
    term3 = C * np.trapz(f3_intrp*C.reshape(-1,1), w_intrp, axis=0)
    
    C_new = sig**2 * (term1 + term2 + term3)
    
    Cdiff = C_new - C
    
    d = np.max(np.abs(Cdiff))/np.max((np.abs(C), np.abs(C_new)))
    d_arr = np.append(d_arr, d)
    
    if (i+1)%5 == 0:
        print('max(|C_diff|)/max(|C|)={}'.format(d))
        # print('max(|C_diff|)={}'.format(max(abs(Cdiff))))
        # plt.figure(1)
        # plt.cla()
        # plt.plot(d_arr, label='C match factor')
        # plt.legend()
        # plt.xlabel('cycle #')
        # plt.yscale('log')
        # plt.pause(0.01)
        
    if i>10:
        Q_err = np.append(Q_err, np.std(Q_arr[-10:]))    
        
    Q_arr = np.append(Q_arr, np.trapz(C, w_intrp))
    plt.figure(2)
    plt.cla()
    plt.plot(Q_arr, label='Q chaos strength')
    plt.plot(d_arr, label='C match factor')
    # plt.plot(Q_err, label='st.dev of 10 points on Q')
    plt.xlabel('cycle #')
    plt.yscale('log')
    plt.legend()
    plt.pause(0.01)
    

    if d < accuracy:
        break 
    
    if isnan(sum(C_new)):
        print('Loop stopped due to NaN')
        break
    
    C = (1-frac)*C + frac*C_new
    
print('Ended after {} cycles'.format(i))

###generating plots and saving data###

# plt.figure(1)
# plt.savefig(HOME + '\\C_match vs cycle#')

plt.figure(2)
plt.savefig(HOME + '\\Q vs cycle + C match vs cycle#')

#interactive plot 
fig = px.scatter(x=range(len(d_arr)), y=[d_arr, Q_arr], log_y=True, title='Q and C_diff vs cycle#')
fig.write_html(HOME + '\Q and C_diff vs Cycle.html')

np.save(HOME + r'\C_diff vs cycle', d_arr)
np.save(HOME + r'\Q vs cycle', Q_arr)

#Original guess vs solution
C_simple_conv = C
fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(w_intrp, C_guess, '--o', label='original guess')
plt.plot(w_intrp, C, '--o', label='simple conv result')
# plt.xlim(-1,1)
plt.title('$C_dZ(\omega)$, $\sigma={}, \mu={}, \lambda={}$'.format(sig, mu, lam))
plt.xlabel('$\omega$')
# plt.xlim(-10**-2, 10**-2)
plt.show()
plt.legend()
plt.savefig(HOME + '\\Solution vs initial guess')

#Checking LHS and RHS of eq (44) when inputting solution
term1_conv = f1_intrp*C_simple_conv
term2_conv = np.trapz(f2_intrp*C_simple_conv.reshape(-1,1)*C_shifter(C_simple_conv, w_intrp, w_intrp), w_intrp, axis=0)
term3_conv = C_simple_conv * np.trapz(f3_intrp*C_simple_conv.reshape(-1,1), w_intrp, axis=0)

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

plt.plot(w_intrp, C_simple_conv, 'o', label='LHS simple convergence')
plt.plot(w_intrp, sig**2*(term1_conv + term2_conv + term3_conv), 'o', label='RHS simple convergence')

plt.title('C RHS vs LHS confirmation, $\sigma={}, \mu={}, \lambda={}$'.format(sig, mu, lam))
plt.legend()
# plt.xlim(-10**-2, 10**-2)
plt.savefig(HOME + '\\C RHS vs LHS confirmation')

#Q error vs convergence cycle
# iters = i+1
# sample = 10
# Q_err = np.zeros(iters-sample)
# for i in range(iters-sample):
#     Q_err[i] = np.std(Q_arr[i:i+sample])
    
# plt.figure()
# plt.plot(Q_err)
# plt.title('Q error vs convergence iterations')
# plt.xlabel('iteration #')

np.save(HOME +r'\C_conv_sol', C_simple_conv)
fig1 = px.scatter(x=w_intrp, y=[C_simple_conv, sig**2*(term1_conv + term2_conv + term3_conv)], title='LHS vs RHS')
fig1.write_html(HOME + '\LHS vs RHS confirmation.html')

duration = 1000 #millisec
freq = 440 #Hz
winsound.Beep(freq, duration)