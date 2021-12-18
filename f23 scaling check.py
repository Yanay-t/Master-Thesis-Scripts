# -*- coding: utf-8 -*-
"""
############################################
#Purpose: Checking the scaling of f2, f3 at lambda=0 for Omega > omega
############################################
#It was found by Guy that:
#f_i(w, W) = w*g_i(W/w)
#x = W/w
#g_2(|x|>>1) ~ c_2/|x|^3, g_2(|x|>>1) ~ c_3/x 
#f_2(|W|>>|w|) ~ c_2/w^4/W^3, f_3(|W|>>|w|) ~ c_3*w^2/W

#We reaffirm this now using our own data

Previous script results required: non
Input: mu, lambda, sigma
#Output:
    #graphs:
    #1. f2 positive & negative tails
    #2. f3 positive & negative 
    #3. f2 collapse
    #4. f3 collapse
"""

import numpy as np
from scipy.special import erf
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from math import log10

from pylab import figure
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

def m_z_of_Z_fun(sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*NZ_fun(Z, lam)
    return 1 - mu*integrate.quad(f, -np.inf, np.inf)[0]

def sig_z2_of_Z_fun(sig_z, m_z, lam):
    f = lambda Z: PZ_fun(Z, sig_z, m_z)*(NZ_fun(Z, lam))**2
    return sig**2*integrate.quad(f, -np.inf, np.inf)[0]

def fun_vec(p, lam):
    m_z, sig_z = p
    x = m_z_of_Z_fun(sig_z, m_z, lam) - m_z
    y = sig_z2_of_Z_fun(sig_z, m_z, lam) - sig_z**2
    return (x,y)

###################################################
#Fucntions needed for calculating f1 f2 f3
###################################################

def A(z, lam):
    return np.sqrt(z**2+4*lam)

def B(z, lam):
    return 0.5*(z + np.sqrt(z**2 + 4*lam))

def f1(z, lam, w):
    return B(z, lam)**2/(A(z, lam)**2 + w**2)

def f2(z, lam, w, W):
    return B(z, lam)**2*(2*(A(z, lam)-B(z, lam))**2 + 0.5*w**2)/((A(z, lam)**2+ w**2)*(A(z, lam)**2 + W**2)*(A(z, lam)**2 + (w+W)**2))

def f3(z, lam, w, W):
    AminB = A(z,lam)-B(z,lam)
    f3_1 = B(z, lam)**2*(4*A(z,lam)**2*(AminB**2 - B(z,lam)*AminB) - w*W*(2*AminB**2-4*B(z,lam)*AminB + 2*B(z,lam)**2) + 4*w**2*(B(z,lam)*AminB - B(z,lam)**2)) / ((A(z, lam)**2 + w**2)**2*(A(z, lam)**2 + W**2)*(A(z, lam)**2 + (w+W)**2)) 
    f3_2 = 2*B(z, lam)**2*(AminB**3 - B(z,lam)**2*AminB + w**2*AminB) / (A(z,lam)*(A(z,lam)**2 + w**2)**2*(A(z,lam)**2 + W**2))
    return f3_1 + f3_2

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

###################################################
#f1 f2 f3 in thier lambda=0 form. Used for comparing with general form to see if matches
###################################################
def f1_lam0(z, lam, w):
    return B(z, lam)**2/(B(z, lam)**2 + w**2)

def f2_lam0(z, lam, w, W):
    return 0.5*B(z, lam)**2*w**2/((B(z, lam)**2+ w**2)*(B(z, lam)**2 + W**2)*(B(z, lam)**2 + (w+W)**2))

def f3_lam0(z, lam, w, W):
    return -2*B(z, lam)**4*w*(2*w+W)/((B(z, lam)**2+ w**2)**2*(B(z, lam)**2 + W**2)*(B(z, lam)**2 + (w+W)**2))

############################
#Setting Parameters
############################

sig = 1.5
mu = 4

pwr = np.arange(6,0,-1)
lam_all = np.concatenate(([0], np.power(0.1, pwr)))
lam_ind = 0 

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
#Calculating 1D slices of f2, f3 
###################################################

lam = lam_all[lam_ind]
m_z = m_z[lam_ind]
sig_z = sig_z[lam_ind]

#new way to divide z
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

#1 dw section
wmax = 0.1
dw = 10**-4
w = np.arange(-wmax, wmax, dw)
w[1000] = 0
    
W = w
lw = len(w)
w0_ind = int(len(w)/2)

e1_ind = np.where(np.abs(w - 10**-1) < 10**-4)[0][0]
e2_ind = np.where(np.abs(w - 10**-2) < 10**-5)[0][0]
e3_ind = np.where(np.abs(w - 10**-3) < 10**-5)[0][0]
e4_ind = np.where(np.abs(w - 10**-4) < 10**-5)[0][0]

w_ind = [e1_ind, e2_ind, e3_ind]
itr = len(w_ind)

f2_zavg = np.zeros((itr,lw))
f3_zavg = np.zeros((itr,lw))

f2_zavg_lam0 = np.zeros((itr,lw))
f3_zavg_lam0 = np.zeros((itr,lw))

for i in range(itr):
    for j in range(lw):
        temp2 = PZ_fun(z, sig_z, m_z)*f2(z, lam, w[w_ind[i]], W[j])
        f2_zavg[i, j] = np.trapz(temp2, z)
        temp3 = PZ_fun(z, sig_z, m_z)*f3(z, lam, w[w_ind[i]], W[j])
        f3_zavg[i, j] = np.trapz(temp3, z)
        
x = np.arange(0.01, 1000, 0.01)

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f2 tail - positive $\Omega$')
for i in range(itr):
    plt.plot(w[w0_ind:]/w[w_ind[i]], w[w_ind[i]]*f2_zavg[i, w0_ind:], 'o',label='General calc, $\omega={}$'.format(w[w_ind[i]]))
plt.plot(x, 1/x**3, '--', label='$1/x^3$ trendline')
plt.plot(x, 0.1/x**3, '--', label='$1/x^3$ trendline')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x=\Omega/\omega$')
plt.ylabel('$\omega*f_2$')
plt.legend()

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f2 tail - negative $\Omega$')
for i in range(itr):
    plt.plot(-w[:w0_ind+1]/w[w_ind[i]], w[w_ind[i]]*f2_zavg[i, :w0_ind+1], 'o',label='General calc, $\omega={}$'.format(w[w_ind[i]]))
plt.plot(x, 1/x**3, '--', label='$1/x^3$ trendline')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x=-\Omega/\omega$')
plt.ylabel('$\omega*f_2$')
plt.legend()

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f3 tail - positive $\Omega$')
for i in range(itr):
    plt.plot(w[w0_ind:]/w[w_ind[i]], np.abs(w[w_ind[i]]*f3_zavg[i, w0_ind:]), 'o',label='General calc, $\omega={}$'.format(w[w_ind[i]]))
plt.plot(x, 1/x**2, '--', label='$1/x^2$ trendline', linewidth=5)
plt.plot(x, 1/x, '--', label='$1/x$ trendline', linewidth=5)
plt.plot(x, 1/x**3, '--', label='$1/x$ trendline', linewidth=5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x=\Omega/\omega$')
plt.ylabel('$\omega*|f_3|$')
plt.legend()

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f3 tail - negative $\Omega$')
for i in range(itr):
    plt.plot(-w[:w0_ind]/w[w_ind[i]], np.abs(w[w_ind[i]]*f3_zavg[i, :w0_ind]), 'o',label='General calc, $\omega={}$'.format(w[w_ind[i]]))
plt.plot(x, 1/x**2, '--', label='$1/x^2$ trendline', linewidth=5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$x=-\Omega/\omega$')
plt.ylabel('$\omega*|f_3|$')
plt.legend()

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f2')
for i in range(itr):
    plt.plot(w/w[w_ind[i]], w[w_ind[i]]*f2_zavg[i, :], 'o', label='General calc, $\omega={}$'.format(w[w_ind[i]]), linewidth='4')

plt.xlabel('$x=\Omega/\omega$')
plt.ylabel('$\omega*f_2$')
plt.xlim(-5, 5)
plt.legend()

fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.title('f3')
for i in range(itr):
    plt.plot(w/w[w_ind[i]], w[w_ind[i]]*f3_zavg[i, :], 'o', label='General calc, $\omega={}$'.format(w[w_ind[i]]))

plt.xlabel('$x=\Omega/\omega$')
plt.ylabel('$\omega*f_3$')
plt.xlim(-5, 5)
plt.legend()


duration = 1000 #millisec
freq = 440 #Hz
winsound.Beep(freq, duration)