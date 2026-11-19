#!/usr/bin/env python3
"""Symbolic verification of the N^{-4} secular cancellation in the main paper."""
import sympy as sp
x,J,Sigma,tau=sp.symbols('x J Sigma tau')
s=1-x**2
A4=(tau*(3*J-1)+J*(3*J-1)*x)/60 \
    +(tau*(2*J-Sigma+1)/20+x*(4*J**2-J*Sigma+J-2*tau**2)/60)/s \
    +(-tau*(2*J-7*Sigma+11)/5+x*(3*J*Sigma-7*J-3*Sigma**2+10*Sigma-6*tau**2-7)/15)/s**2 \
    +(-8*tau*(3*Sigma-5)/15+2*x*(3*Sigma**2-10*Sigma+12*tau**2+7)/15)/s**3
B4=J*(20*J**2+41*J-12)/1440 \
    + (-(5*J**3-15*J**2*Sigma+17*J**2-6*J*Sigma-15*J*tau**2+18*tau**2)/360-tau*(3*J-1)*x/60)/s \
    + (-(5*J**2*Sigma-21*J**2+26*J*Sigma+20*J*tau**2-42*J-15*Sigma*tau**2+27*tau**2)/120+tau*(15*J**2+15*J*Sigma-21*J+8*Sigma-5*tau**2-8)*x/180)/s**2 \
    + ((18*J*Sigma-42*J-5*Sigma**3+10*Sigma**2-60*Sigma*tau**2-5*Sigma+76*tau**2)/90-tau*(36*J-15*Sigma**2+38*Sigma-20*tau**2-47)*x/90)/s**3 \
    + ((Sigma**3-2*Sigma**2+12*Sigma*tau**2+Sigma-8*tau**2)/18-tau*(3*Sigma**2-4*Sigma+4*tau**2+1)*x/9)/s**4
k4=(J+sp.Rational(1,4))**2
At=k4*x*(7-3*x**2)/(60*s)
Bt=k4/(1440*s**2)*(41*x**4-94*x**2+225-20*J*x**2*s+20*Sigma*x**2-40*tau*x**3)
def degree(expr):
    num=sp.cancel(expr).as_numer_denom()[0]
    return sp.Poly(sp.expand(num),J).degree()
da,db=degree(A4-At),degree(B4-Bt)
assert da<=1 and db<=2
print('D4 secular verification passed')
print('deg_J(A4-A4_tilde) =',da)
print('deg_J(B4-B4_tilde) =',db)
