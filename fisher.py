import numpy as np
from scipy.optimize import root
from scipy.special import zeta
import matplotlib.pyplot as plt
from scipy.integrate import quad

def est_eq(c, t, kb, kon):
    """
    The equation whose root in c gives the MLE for c given observation t.
    
    Equation:
      c*kb*(c*kb - kon)*t = kon*(exp((c*kb - kon)*t) - 1)
    
    Parameters:
      c   : concentration (scalar, unknown)
      t   : observed time
      kb  : rate constant k_b
      kon : rate constant k_on
      
    Returns:
      The value of the left-hand side minus the right-hand side.
    """
    return c * kb * (c * kb - kon) * t - kon * (np.exp((c * kb - kon) * t) - 1)

def MLEstimator(t, kb, kon):
    """
    Given an observation t and constants kb and kon, find the MLE for c by solving the
    likelihood equation numerically.
    
    Uses an initial guess of c0 = kon/kb.
    """
    # c0 = kon / kb  # initial guess
    c0 = 1
    sol = root(lambda c: est_eq(c, t, kb, kon), c0)
    if sol.success:
        return sol.x[0]
    else:
        raise RuntimeError(f"Root finding failed for t={t}, kb={kb}, kon={kon}")
    
def Ioff(c, kb, kon):
    """
    Computes the Fisher information from observing OFF times
    """

    if c*kb > kon:
        a = c*kb/(c*kb-kon)
        return (1 - a**2 * (1 - 2*kon * a**2 * zeta(3,a) / (c*kb))) / c**2
    
    elif c*kb < kon:
        a = c*kb/(c*kb-kon)
        b = kon/(kon-c*kb)
        return (1 - a**2 * (1 - 2*kon * a**2 * zeta(3,b) / (c*kb))) / c**2
    
    elif c*kb == kon:
        return 2 / c**2

def MLEstimatorOffVar(cTrue, n, kb, kon):
    """
    Computes the variance of the MLE
    """
    return 1/(n*cTrue**2*Ioff(cTrue, kb, kon))
    
def MoMEstimatorOff(t_mean, kb, kon):
    """
    Given an empirical mean of the off time, find the Method of Moments estimate of c.
    """
    return 1/(kb*(t_mean-1/kon))

def MoMEstimatorOffVar(cTrue, n, kb, kon):
    """
    Variance of the Method of Moments estimator for off times
    """
    return (1+(cTrue*kb)**2/kon**2)/n


def MoMEstimatorOnVar(c, n, kbon, koffu, kuon):
    """
    Variance of the Method of Moments estimator for ON times
    """
    return ((c*kbon + koffu + kuon)**2 - 2*kuon*koffu)/(n*c**2*kbon**2)


def sample_t(cTrue, kb, kon):
    """
    Sample a single observation t from the hypoexponential distribution.
    
    T1 ~ Exp(cTrue*kb) and T2 ~ Exp(kon), then t = T1 + T2.
    """
    T1 = np.random.exponential(scale=1.0/(cTrue * kb))
    T2 = np.random.exponential(scale=1.0/kon)
    return T1 + T2

def MLEstimatorList(cTrue, n, kb, kon):
    """
    Generate n observations from the hypoexponential distribution and return the list of 
    corresponding MLE and MoM estimates for c.
    """
    t_samples = [sample_t(cTrue, kb, kon) for _ in range(n)]
    c_mle_estimates = [MLEstimator(t, kb, kon) for t in t_samples]
    c_mom_estimate = MoMEstimatorOff(np.mean(t_samples), kb, kon)
    return np.array(c_mle_estimates), c_mom_estimate


# Define helper functions for coth and csch that handle small arguments.
def coth(x):
    # For very small x, use the series expansion: coth(x) ~ 1/x + x/3
    if np.abs(x) < 1e-8:
        return 1.0/x + x/3.0
    elif x > 100:
        return 1.0
    else:
        return np.cosh(x) / np.sinh(x)

def csch(x):
    # For very small x, use the series expansion: csch(x) ~ 1/x - x/6
    if np.abs(x) < 1e-8:
        return 1.0/x - x/6.0
    elif x > 100:
        return 0
    else:
        return 1.0/np.sinh(x)

# Define the integrand function.
def IonIntegrand(t, c, koffu, kuon, kbon):
    """
    Computes the integrand:
    
    I(t) = -{ A*exp(-0.5*(B+sqrt(D))*t) * (exp(sqrt(D)*t)-1)*kbon^2 *
           [ -4*(8*A + D) + 8*A*sqrt(D)*t*coth( (sqrt(D)*t)/2 )
             + D*(4*A + D)*t^2*csch( (sqrt(D)*t)/2 )^2 ]
          } / ( 4 * D^(5/2) ).
    """
    A = koffu * kuon    # Example value for A
    B = c *kbon + koffu + kuon   # Example value for B
    D = -4 *koffu* kuon + (c* kbon + koffu + kuon)**2    # Example value for D (should be positive)

    sqrtD = np.sqrt(D)
    exp_diff = np.exp(-0.5*(B - sqrtD)*t) - np.exp(-0.5*(B + sqrtD)*t)

    # Argument for coth and csch:
    arg = sqrtD * t / 2.0
    coth_val = coth(arg)
    csch_val = csch(arg)
    
    # The bracketed term
    bracket = -4*(8*A + D) + 8*A*sqrtD*t*coth_val + D*(4*A + D)*t**2 * (csch_val**2)
    
    numerator = A * exp_diff * (kbon**2) * bracket
    denominator = 4 * (D**(5/2))
    
    return numerator/denominator

def Ion(c, koffu, kuon, kbon):
    """
    Computes the Fisher information observing ON times
    """
    tauon = 1/koffu + 1/kuon + c*kbon/(koffu*kuon)

    # Define integration limits. For example, integrate from t=0 to t=10.
    t_min = 1e-4*tauon
    t_max = 15*tauon
    result, error = quad(IonIntegrand, t_min, t_max, args=(c, koffu, kuon, kbon))

    if error/(result * c**2) < 1e-2:
        return result
    else:
        raise ValueError("Relative error too large")
    
def VMLE(logrates, c, dc):
    """
    Computes the SPRT drift for MLE estimation
    """
    kboff, kbon, konb, kuon, koffu = 10**np.array(logrates)

    tauon = 1/koffu + 1/kuon + c*kbon/(koffu*kuon)
    tauoff = 1/(c*kboff) + 1/konb
    tau = tauon + tauoff

    if c < 40:
        ion = 0
        ioff = Ioff(c, kboff, konb)
    elif c > 1e4:
        ion = Ion(c, koffu, kuon, kbon)
        ioff = 0
    else:
        ion = Ion(c, koffu, kuon, kbon)
        ioff = Ioff(c, kboff, konb)

    return - 0.5*dc**2*(ion + ioff) / tau

def FisherRate(logrates, c, dc):

    return - VMLE(logrates, c, dc) * 2 * (c/dc)**2


def VonMLE(logrates, c, dc):
    """
    Computes the SPRT drift for MLE estimation
    """
    kboff, kbon, konb, kuon, koffu = 10**np.array(logrates)

    tauon = 1/koffu + 1/kuon + c*kbon/(koffu*kuon)
    tauoff = 1/(c*kboff) + 1/konb
    tau = tauon + tauoff

    if c < 40:
        ion = 0
    else:
        ion = Ion(c, koffu, kuon, kbon)

    return - 0.5*dc**2*(ion) / tau

def VoffMLE(logrates, c, dc):
    """
    Computes the SPRT drift for MLE estimation
    """
    kboff, kbon, konb, kuon, koffu = 10**np.array(logrates)

    tauon = 1/koffu + 1/kuon + c*kbon/(koffu*kuon)
    tauoff = 1/(c*kboff) + 1/konb
    tau = tauon + tauoff

    if c > 1e4:
        ioff = 0
    else:
        ioff = Ioff(c, kboff, konb)

    return - 0.5*dc**2*(ioff) / tau


def FisherRateOn(logrates, c, dc):
    return - VonMLE(logrates, c, dc) * 2 * (c/dc)**2


def FisherRateOff(logrates, c, dc):
    return - VoffMLE(logrates, c, dc) * 2 * (c/dc)**2