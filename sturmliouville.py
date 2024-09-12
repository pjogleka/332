# Solver for a Sturm-Liouville problem (finite string Eigenmodes)
# This can be generalized to any BVP, as long as it's solvable
# by a shooting method.
import numpy as np
import matplotlib.pyplot as plt

#========================================
# Density function for string
# Returns the mass density at given position x.
# Here, we make use of optional arguments:
# The syntax would be (for a flat density):
# a = rho(x,imode=0)
def rho(x,**kwargs):
    for key in kwargs:
        if (key=='imode'):
            imode = kwargs[key]
    if (imode == 0):   # flat
        rho = 1
    elif (imode == 1): # exponential
        rho = 1 + np.exp(10 * x)
    return rho

#========================================
# fRHS for the (non-)uniform string
# Should return an array dydx containing three
# elements corresponding to y'(x), y''(x), and lambda'(x).
def dydx_string(x,y,**kwargs):
    dydx    = np.zeros(3)
    dydx[0] = y[1] 
    dydx[1] = - (y[2] * np.pi) ** 2 * rho(x, **kwargs) * y[0]
    dydx[2] = 0
    return dydx

#========================================
# fLOA for the string problem
# This is the (boundary) loading function.
# Should return the integration boundaries x0,x1, and
# the initial values y(x=0). Therefore, y must be
# an array with three elements. 
# The function takes an argument v, which in our case
# is just the eigenvalue lambda.
# Note that some of the initial values in y can
# (and should!) depend on lambda.
def load_string(v):
    x0 = 0
    x1 = 1
    y = np.array([0, v[0] * np.pi, v[0]])
    return x0,x1,y

#========================================
# fSCO for the string problem
# This is the scoring function. Should return
# the function that needs to be zeroed by the
# root finder. In our case, that's just y(1). 
# The function takes the arguments x and y, where
# y is an array with three elements.
def score_string(x,y):
    score = y[0][-1]
    return score # displacement should be zero.

#========================================
# Single rk4 step.
# Already provided. Good to go; 2.5 free points :-)
def rk4(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,**kwargs)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,**kwargs)
    k3 = dx*fRHS(x0+0.5*dx,y0+0.5*k2,**kwargs)
    k4 = dx*fRHS(x0+    dx,y0+    k3,**kwargs)
    y  = y0+(k1+2.0*(k2+k3)+k4)/6.0
    return y    

#========================================
# ODE IVP driver.
# Already provided. Good to go; 2.5 free points :-)
def ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs):
    nvar    = y0.size                      # number of ODEs
    x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
    y       = np.zeros((nvar,nstep+1))     # result array 
    y[:,0]  = y0                           # set initial condition
    dx      = x[1]-x[0]                    # step size
    for k in range(1,nstep+1):
        y[:,k] = fORD(fRHS,x[k-1],y[:,k-1],dx,**kwargs)
    return x,y

#=======================================
# A single trial shot.
# Sets the initial values (guesses) via fLOA, calculates 
# the corresponding solution via ode_ivp, and returns 
# a "score" via fSCO, i.e. a value for the rootfinder to zero out.
def bvp_shoot(fRHS,fORD,fLOA,fSCO,v,nstep,**kwargs):
    x0, x1, y0 = fLOA(v)
    x, y = ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs)
    score = score_string(x, y)
    return score # this should be zero, and thus can be directly used.

#=======================================
# The rootfinder.
# The function pointers are problem-specific (see main()). 
# v0 is the initial guess for the eigenvalue (in our case).
# Should return x,y, so that the solution can be plotted.
def bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,**kwargs):
    f0 = bvp_shoot(fRHS, fORD, fLOA, fSCO, v0, nstep, **kwargs)
    i = 1
    vi = v0 * 1.1 ** i
    fi = bvp_shoot(fRHS, fORD, fLOA, fSCO, vi, nstep, **kwargs)
    while f0 * fi >= 0:
        vi = v0 * 1.1 ** i
        fi = bvp_shoot(fRHS, fORD, fLOA, fSCO, vi, nstep, **kwargs)
        i += 1
    tol = 1e-6
    vmid = (v0 + vi)/2
    while abs(v0 - vi) >= tol:
        vmid = (v0 + vi)/2
        fmid = bvp_shoot(fRHS, fORD, fLOA, fSCO, vmid, nstep, **kwargs)
        if f0 * fmid < 0:
            vi = vmid
        elif fi * fmid < 0:
            v0 = vmid
    x0, x1, y0 = fLOA(vmid)
    x, y = ode_ivp(fRHS,fORD,x0,x1,y0,nstep,**kwargs)
    return x,y

#=======================================
def main():

    nstep = 500
    imode = 1
    v0    = np.array([0.5])
    fRHS  = dydx_string
    fLOA  = load_string
    fSCO  = score_string
    fORD  = rk4
    x,y   = bvp_root(fRHS,fORD,fLOA,fSCO,v0,nstep,imode=imode)

    u = y[0,:] # amplitude
    l = y[2,:] # eigenvalue

    plt.figure(num=1,facecolor='white')
    
    plt.subplot(221)
    plt.plot(x,u,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('x')
    plt.ylabel('u')
    
    plt.subplot(223)
    plt.plot(x,l,linestyle='-',color='black',linewidth=1.0)
    plt.xlabel('x')
    plt.ylabel('$l$')
    
    N = 20
    vn = np.zeros(20)
    v = v0
    for i in range(N):
        vn[i] = v
        x, y = bvp_root(fRHS,fORD,fLOA,fSCO,v,nstep,imode=imode)
        print("eig" + str(i+1) + " = " + str(y[2, 0]))
        plt.subplot(222)
        plt.plot(x,y[0, :],linestyle='-',linewidth=1.0)
        v = np.array([v0[0]*1.1**i])
        plt.xlabel('x')
        plt.ylabel('$u$')

    plt.subplot(224)
    plt.scatter(np.linspace(1, 20, 20), vn)
    plt.xlabel('n')
    plt.ylabel('$v$')
    
    plt.tight_layout()

    plt.show()



#=======================================
main()

