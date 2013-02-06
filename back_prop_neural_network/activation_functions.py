__author__ = 'samyvilar'

import numpy

# output = 1/(1 + e**-value)
# output*(1 + e**-value) = 1
# (1 + e**-value) = 1/output
# e**-value = (1/output) - 1
# -value = ln((1/output) - 1)
#  value -ln((1/output) - 1)
#
# value = ln( ((1/output) - 1)**-1)
# value = ln ( 1/((1/output) - 1) )
# value = ln ( 1/((1 - output)/output)
# value = ln (output/(1 - output))

def sigmoid(value):
    return 1.0/(1.0 + (numpy.e**(-value)))

def inv_sigmoid(output):
    return numpy.log(output/(1 - output))

