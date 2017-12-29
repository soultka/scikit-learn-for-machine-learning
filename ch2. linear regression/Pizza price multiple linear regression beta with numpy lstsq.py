from numpy.linalg import lstsq
#Training instance 
X = [[1,6,2] , [1,8,1] , [1,10,0], [1,14,2], [1,18,0]]
y = [[7], [9], [13], [17.5], [18]]
# caculate beta
print (lstsq(X,y)[0])
