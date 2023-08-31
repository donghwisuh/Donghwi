import math
import numpy as np

def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return (a, b, c, d)
    
def angle_between_plane_and_vector(a, b, c, d, l, m, n):
    #print('enter func')
    sin_theta = abs(a*l + b*m + c*n)/(math.sqrt(a**2 + b**2 + c**2) * math.sqrt(l**2 + m**2 + n**2))
    return np.arcsin(sin_theta)*180/math.pi


def project_vector_onto_plane(x, y, z, a, b, c, d): #vector: xyz, plane: abcd
    dot_prod = x*a +y*b + z*c
    plane_norm_squared = a*a + b*b + c*c
    d = dot_prod / plane_norm_squared
    px = d*a
    py = d*b
    pz = d*c
    (px, py, pz) = (x - px, y-py, z-pz)
    return (px, py, pz)

def angle_between_two_planes(a1, b1, c1, d1, a2, b2, c2, d2):
    cos_theta = abs(a1*a2 + b1*b2 + c1*c2)/(math.sqrt(a1*a1 + b1*b1 + c1*c1) * math.sqrt(a2*a2+ b2*b2+ c2*c2))
    return np.arccos(cos_theta)*180/math.pi

"""
x1 =-1
y1 = 2
z1 = 1
x2 = 0
y2 =-3
z2 = 2
x3 = 1
y3 = 1
z3 =-4
a,b,c,d = equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3)
l,m,n,o = equation_plane(z1, y1, x1, z2, y2, x2, z3, y3, x3)
print(a,b,c,d)
print(l,m,n,o)
px,py,pz = project_vector_onto_plane(1,1,1,a,b,c,d)
angle = angle_between_plane_and_vector(a,b,c,d,px,py,pz)
angle2 = angle_between_two_planes(a,b,c,d,l,m,n,o)
print(px,'x',py,'y',pz,'z')
print(angle2)
"""
