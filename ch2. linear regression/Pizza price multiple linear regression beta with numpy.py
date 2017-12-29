from numpy.linalg import inv
from numpy import dot, transpose

#Training instance 
X = [[1,6,2] , [1,8,1] , [1,10,0], [1,14,2], [1,18,0]]
y = [[7], [9], [13], [17.5], [18]]
# caculate beta
beta = dot(inv(dot(transpose(X),X)), dot(transpose(X),y))
print (beta)
