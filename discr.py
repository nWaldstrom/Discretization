import os
import sys
sys.path.append("..")
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
from scipy.optimize import minimize
# import scipy.stats as ss

def entropyObjective(Lambda, Tx, TBar, q, nMoments):   
    _, N = np.shape(Tx)
    L = nMoments
    # print(Tx.shape)
    # Compute objective function
    Tdiff = Tx - TBar[:,np.newaxis]
    
    if np.isclose(L,1): 
        temp = q * np.exp(Lambda * Tdiff)
    else:
        temp = q * np.exp(Lambda @ Tdiff)

    obj = np.sum(temp)
    
    # Compute gradient of objective function
    if np.isclose(L,1): 
        temp2 = temp * Tdiff
    else:
        temp2 = temp[np.newaxis,:] * Tdiff  
        
    gradObj = np.sum(temp2, axis=1)
    
    # print(gradObj.shape)
    # Compute hessian of objective function
    # print(Tdiff.shape, temp2.shape)
    hessianObj = temp2 @ Tdiff.T
    return obj, gradObj, hessianObj

def obj(Lambda, *args):
   Tx, TBar, q, nMoments = args
   obj, _, _ =  entropyObjective(Lambda, Tx, TBar, q, nMoments)
   # print(obj)
   return obj

def jac(Lambda, *args):
    Tx, TBar, q, nMoments = args
    _, gradObj, _ = entropyObjective(Lambda, Tx, TBar, q, nMoments)
    # print(gradObj)
    return gradObj

def hess(Lambda, *args):
    Tx, TBar, q, nMoments = args
    _, _, hessianObj = entropyObjective(Lambda, Tx, TBar, q, nMoments)
    # print(hessianObj.shape)
    return hessianObj


def discreteApproximation(D, T, TBar, q, nMoments):
    Lambda0 =  np.zeros(nMoments); 
    if q is None:
        q      =  np.ones(Nm)/Nm;
    Tx = T(D)

    # obj_func = lambda x : obj(x, Tx, TBar, q0)
    res = minimize(obj, Lambda0, method='Newton-CG', jac=jac, hess=hess, args=(Tx, TBar, q, nMoments))
    
    if not res.success:
        print(res.message )
        # q      =  np.ones(Nm)/Nm;
        # res = minimize(obj, Lambda0, method='Newton-CG', jac=jac, hess=hess, args=(Tx, TBar, q, nMoments), options={'maxiter' : 3000})
        res = minimize(obj, Lambda0, method='Nelder-Mead', args=(Tx, TBar, q, nMoments), options={'maxiter' : 3000})
         
    LambdaBar = res.x 
    objval, gradObjval, _ = entropyObjective(LambdaBar, Tx, TBar, q, nMoments);
    Tdiff = Tx- TBar[:,np.newaxis]
  
    if np.isclose(nMoments, 1):
        p = (q * np.exp(Tdiff * LambdaBar)) / objval
    else:
        p = (q * np.exp(LambdaBar @ Tdiff)) / objval
    momentError = gradObjval / objval
    return p, LambdaBar, momentError


def solve_dual_problem(mu, scale, Nm, X, TBar, q_func, condmean_f, nMoments):
    scalingFactor = max(abs(X)) / scale 

    scalingFactor_ = np.array([scalingFactor**(1+x) for x in range(nMoments)])
    mom = lambda y : np.array([(y-condMean)**(1+x) / scalingFactor_[x] for x in range(nMoments)])      
        
    # p, LambdaBar, momentError = discreteApproximation(X, mom, TBar[:nMoments]/ scalingFactor_)
    
    P = np.zeros([Nm, Nm]) * np.nan# matrix to store transition probability
    P1 = np.zeros([Nm, Nm]) * np.nan# matrix to store transition probability    
    
    for ii in range(Nm):
        condMean = condmean_f(X[ii])
        xPDF = X - condMean 
        # q = np.sum([p * ss.norm.pdf(xPDF, mu, sd) for mu, sd, p in zip(muC, sigmaC, pC)], axis=0)
        q = q_func(xPDF)
        q[q < 1e-08] = 1e-08
        P[ii,:], LambdaBar, momentError = discreteApproximation(X, mom, TBar[:nMoments]/ scalingFactor_, q, nMoments)
        
    return P, LambdaBar, momentError, q

 
def discr_mixnorm(mu, Nm, nMoments, X, TBar, q_func, condmean_f, tol=1e-05, scale=1):

    P, LambdaBar, momentError, q = solve_dual_problem(mu, scale, Nm, X, TBar, q_func, condmean_f, nMoments)
    if (max(np.abs(momentError)) > tol) and (nMoments > 1):
        for x in range(nMoments-1):
            print(f'Warning: Failed to match first {nMoments-(x)} moments.  Just matching {nMoments-(x+1)}')
            P, LambdaBar, momentError, q = solve_dual_problem(mu, scale, Nm, X, TBar, q_func, condmean_f, nMoments-(1+x))
            if  max(np.abs(momentError)) < tol:
                print(f'Convergence with {nMoments-(1+x)} moments!')
                markov_check(P)
                return P, LambdaBar, momentError, q 
        else:
            raise ValueError(f'No convergence even with only one moment!')
    else:
        print(f'Convergence with {nMoments} moments!')    
        markov_check(P)
        return P, LambdaBar, momentError, q 

def markov_check(Pi):
    (N,_) = Pi.shape
    for k in range(N):
        assert np.isclose(sum(Pi[k,:]),1)
    # assert np.isclose(sum(utils.stationary(Pi)),1)

def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Find invariant distribution of a Markov chain by iteration."""
    if pi_seed is None:
        pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    else:
        pi = pi_seed
         
     
    for it in range(maxit):
        pi_new = pi @ Pi
        
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    pi = pi_new

    return pi    
