"""
Python implementation of Lai's 3j program originally developed for Maple.
As nearly as I can tell, there are no checks of the triangle conditions, so
it's up to the user to verify that all arguments lead to an allowed 3j symbol.

The sum of the azimuthal quantum numbers is explicitly checked to be zero.

The output has been tested against Anthony Stone's Wigner Coefficient Calculator

http://www-stone.ch.cam.ac.uk/wigner.html

for a few cases.  More extensive testing would be needed to ensure that this
module was road worthy.

Literature: Lai, S.-T. "Computation of Algebraic Formulas for Wigner 3-j, 6-j,
and 9-j Symbols by Maple" International Journal of Quantum Chemistry,
52:593--607 (1994)

Written:  KAE Department of Physics, University at Albany 26 Oct 08
Uses:     numpy, scipy
Returns:  Wigner 3J coefficient in s and sets flag true if arguments OK.
Future:   Explicitly check triangle conditions
Modified: 27 Oct 08 (implements Future)

"""

from numpy import *
from scipy import *
from scipy.misc import factorial
from decimal import Decimal
def W3J(J1,J,J2,M1,M,M2):

    # set flags to determine allowed 3j symbols based on calling arguments
    
    if ((abs(J1-J) <= J2) and (J2 <= J1+J)):
        tri = True
    else:
        tri = False

    if ((M1 in arange(-J1, J1+1)) and (M in arange(-J, J+1)) and \
        (M2 in arange(-J2, J2+1))):
        mem = True
    else:
        mem = False

    if (floor(J1 + J + J2) == J1 + J + J2):
        perim = True
    else:
        perim = False

    if (M1 + M + M2 == 0):
        mag = True
    else:
        mag = False

    # now compute allowed 3j symbol, return 0 if not allowed

    flag = tri and mem and perim and mag
    
    if (flag):
        delta = sqrt( Decimal( factorial(J1+J2-J,exact=1)*factorial(J1-J2+J,exact=1)* \
                     factorial(-J1+J2+J,exact=1))/Decimal(factorial(J1+J+J2+1,exact=1)))

        a = sqrt(Decimal( factorial(J2+M2,exact=1)*factorial(J2-M2,exact=1)/factorial(J+M,exact=1)/\
                 factorial(J-M,exact=1)/factorial(J1-M-M2,exact=1))/Decimal(factorial(J1+M+M2,exact=1)))
        s0 = Decimal(0)

        u  = -J1+J2+J
                
        z1 = max(-J1-M-M2,u-J2-M2,0)
        z2 = min(J+J2-M-M2,J2-M2,u)
        
        for z in arange(z1,z2+1):
            if ((0 <= J+J2-M-M2-z) and (0 <= J2-M2-z) and \
                (0 <= u-z) and (0 <= J2+M2-u+z)):
                s0 = s0 + Decimal((-1)**(2*J-J1-M1+z)) * \
                     Decimal(factorial(J+J2-M-M2-z,exact=1)*factorial(J1+M+M2+z,exact=1))/\
                     Decimal(factorial(z,exact=1))/Decimal(factorial(J2-M2-z,exact=1))/Decimal(factorial(u-z,exact=1))/\
                     Decimal(factorial(J2+M2-u+z,exact=1))

        s = float(a*delta*s0)

    else:
        s = 0

    return s,flag

