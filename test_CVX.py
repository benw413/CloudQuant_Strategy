from cvxopt import matrix, solvers
A = matrix([ [ 0.3, -0.4, -0.2, -0.4, 1.3 ],
                 [ 0.6, 1.2, -1.7, 0.3, -0.3 ],
                 [-0.3,  0.0,   0.6, -1.2, -2.0 ] ])
b = matrix([ 1.5, .0, -1.2, -.7, .0])
m, n = A.size
I = matrix(0.0, (n,n))
e = matrix(1, (1, 3))
I[::n+1] = 1.0
G = matrix([-I, matrix(0.0, (1, n)), I])
h = matrix(n*[0.0] + [1.0] + n*[0.0])
dims = {'l': n, 'q': [n+1], 's': []}
sol = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)
a = matrix(sol["x"])
print(a)
print(e)
print(e*a)
print(sol["x"]/(e*a))


