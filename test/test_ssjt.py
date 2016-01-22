import numpy as np
import statsmodels.api as sm
import sys
from scipy import stats

def JT( b1, b2, p1, p2, r):
    chi1 = stats.chi2.ppf( 1-p1, 1 )
    chi2 = stats.chi2.ppf( 1-p2, 1 )
    Z1 = np.sign(b1)*np.sqrt(chi1)
    Z2 = np.sign(b2)*np.sqrt(chi2)
    chiJ = (Z1**2 + Z2**2 - 2*r*Z1*Z2)/(1-r**2)
    pJ = 1-stats.chi2.cdf(chiJ,2)
    return chiJ,pJ


N = 5000
n=N
b1 = 0.1
b2 = 0.1
p1=0.4
p2=0.1
r=-0.3

g1a = np.random.binomial(1,p1,N)
t2a = np.random.binomial(1,p1,N)
u2a = np.random.binomial(1,abs(r),N)
g1b = np.random.binomial(1,p2,N)
t2b = np.random.binomial(1,p2,N)
u2b = np.random.binomial(1,abs(r),N)
if r >= 0:
    g2a = ((1-u2a)*t2a+u2a*g1a)
    g2b = ((1-u2b)*t2b+u2b*g1b)
else:
    g2a = (((1-u2a)*t2a+u2a*g1a)+1)%2
    g2b = (((1-u2b)*t2b+u2b*g1b)+1)%2
g1 = g1a+g1b
g2 = g2a+g2b
g1 = g1-np.mean(g1)
g2 = g2-np.mean(g2)

Y = b1*g1 + b2*g2 + np.random.normal(size=N)
Y = Y - np.mean(Y)

M1 = sm.OLS(Y,g1).fit()
M2 = sm.OLS(Y,g2).fit()
J = sm.OLS(Y,np.vstack([g1,g2]).T).fit()

beta1 = M1.params[0]
beta2 = M2.params[0]
pV1 = M1.pvalues[0]
pV2 = M2.pvalues[0]

varY = np.var(Y)
den = 1.0/(1-r**2)
hv1 = 2*N*p1*(1.0-p1)
hv2 = 2*N*p2*(1.0-p2)
hcv = np.sqrt(hv1*hv2)*r
XXTIhat = np.array([[hv2,hcv],[hcv,hv1]])/(hv1*hv2-hcv**2)

B1J = den*(beta1 - r*beta2)
B2J = den*(beta2 - r*beta1)

resid = varY*(n/(n-2)) - (B1J*hv1*beta1 + B2J*hv2*beta2)/(n-2)
varBJ = resid*XXTIhat
B1JSE = np.sqrt(varBJ[0,0])
B2JSE = np.sqrt(varBJ[1,1])
chiJ, pJ = JT(beta1, beta2, pV1, pV2, r)

print B1J, B2J, B1JSE, B2JSE
print pJ
print J.summary()

sys.exit()
OR1J = np.exp(B1JSE)
OR2J = np.exp(B2JSE)
T1J = B1J/B1JSE
T2J = B2J/B2JSE
pV1J = 1-stats.t.cdf(T1J, n-2)
pV2J = 1-stats.t.cdf(T2J, n-2)
