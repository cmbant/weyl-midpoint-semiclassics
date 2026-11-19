#!/usr/bin/env python3
"""Finite checks of the chamber symmetries in prop:q6j-class.
# log: verify_q6j_chambers.txt

Independent KR sum: exact rational squared values and signs at q^(1/2)=3/2.
Independent Wigner functions: 50-digit SU(2) matrix exponentials.
Integer, half-integer, boundary, and exceptional channels; all four chamber
images at three ranks each. A wrong generator sign is a negative control.
These are finite regressions of the symmetry proof, not an all-label proof.
"""
from fractions import Fraction as Q
from functools import lru_cache

from mpmath import mp
mp.dps = 50
# Independent KR evaluation: exact rational q-numbers, with q^(1/2)=3/2.
r = Q(3, 2)


def qn(x):
    z = 2*x
    assert z.denominator == 1
    return (r**int(z) - r**(-int(z)))/(r*r - r**-2)


@lru_cache(None)
def qfac(x):
    assert x.denominator == 1 and x >= 0
    result = Q(1)
    for k in range(1, int(x)+1):
        result *= qn(Q(k))
    return result


def kr(a, b, c, m, n, f):
    triads = [(a, b, c), (b+m, a+n, c), (a, a+n, f), (b+m, b, f)]
    al = [sum(t) for t in triads]
    be = [2*a+2*b+m+n, a+b+c+n+f, a+b+c+m+f]
    terms = []
    for z in range(int(max(al)), int(min(be))+1):
        den = Q(1)
        for v in al:
            den *= qfac(z-v)
        for v in be:
            den *= qfac(v-z)
        terms.append((-1)**z*qfac(Q(z+1))/den)
    total = sum(terms, Q(0))
    norm = Q(1)
    for u, v, w in triads:
        norm *= qfac(u+v-w)*qfac(u-v+w)*qfac(-u+v+w)/qfac(u+v+w+1)
    exponent = a+b+c+n+f
    assert exponent.denominator == 1
    gauge = 1 if int(exponent) % 2 == 0 else -1
    norm *= qn(2*a+n+1)*qn(2*b+m+1)*qn(2*f+1)
    return gauge*(1 if total > 0 else -1 if total < 0 else 0), norm*total**2


def rotation(f, m, n, theta):
    size = int(2*f+1)
    mat = mp.matrix(size)
    rank = mp.mpf(f.numerator)/f.denominator
    for i in range(size-1):
        mm = rank-i
        val = mp.sqrt((rank+mm)*(rank-mm+1))/2
        mat[i, i+1] = -val
        mat[i+1, i] = val
    return mp.expm(theta*mat)[int(f-m), int(f-n)]


FAMILIES = [
    (Q(7), Q(6), Q(8), Q(2), Q(1)),
    (Q(7), Q(6), Q(8), Q(2), Q(-1)),
    (Q(13, 2), Q(11, 2), Q(7), Q(3, 2), Q(1, 2)),
    (Q(13, 2), Q(11, 2), Q(7), Q(3, 2), Q(-1, 2)),
    (Q(13, 2), Q(11, 2), Q(7), Q(1, 2), Q(-1, 2)),
    (Q(13, 2), Q(11, 2), Q(7), Q(1, 2), Q(1, 2)),
    (Q(7), Q(6), Q(8), Q(0), Q(0)),
]
count = 0
negative_controls = 0
for a, b, c, m, n in FAMILIES:
    generator_sign = 1 if int(m-n) % 2 == 0 else -1
    images = [
        (a, b, m, n, 1),
        (b, a, n, m, generator_sign),
        (a+n, b+m, -m, -n, generator_sign),
        (b+m, a+n, -n, -m, 1),
    ]
    for f in [max(abs(m), abs(n))+i for i in range(3)]:
        sign, square = kr(a, b, c, m, n, f)
        d = rotation(f, m, n, mp.mpf('1.1'))
        for aa, bb, mm, nn, sgn in images:
            si, sq = kr(aa, bb, c, mm, nn, f)
            assert sq == square and si == sgn*sign
            assert abs(rotation(f, mm, nn, mp.mpf('1.1'))-sgn*d) < mp.mpf('1e-45')
            count += 1
            if sgn == -1 and sign != 0:
                assert si != sign
                negative_controls += 1
print('PASS independent exact KR squared values and signs, and SU(2) '
      f'matrix exponentials: {count} chamber images')
assert negative_controls > 0
print('PASS negative control: dropping the generator sign fails in '
      f'{negative_controls} nonzero images')
