#Importing the required Libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sympy import *

with open('data3_new.pkl', 'rb') as a:   			 #Un-pickling the given Dataset
	data = pickle.load(a)   				 #Loading the pickle file

#Creating two empty lists---------------------------------
x1 = []
y1 = []

#Feeding the data into the lists--------------------------
for i in range(0,len(data)):
	x1.append(data[i][0])    
	y1.append(data[i][1])

#Creating the lists into numpy array----------------------
z1 = np.array(x1)   
z2 = np.array(y1)

#Finding the mean of the array and subtracting from the given list to create a new list
xmean1 = 0
ymean1 = 0

for i in range(0, len(z1)):
	xmean1 = xmean1+z1[i]
xmean1 = xmean1/len(z1)
X1 =  z1-xmean1

for j in range(0, len(z2)):
	ymean1 = ymean1+z2[j]
ymean1 = ymean1/len(z2)
Y1 =  z2-ymean1

#Stacking the updated lists together and finding the Transpose----------------------------
U1=(np.vstack((X1,Y1))).T
U1t=U1.T

#Finding the Co-variance Matrix-----------------------------------------------------------
cov=(np.matmul(U1t,U1))/len(data)
print (cov)

#Finding the Eigen Values-------------------------------------------------
Lambda = symbols('Lambda')  
I = eye(2)
A = Matrix([[cov[0][0], cov[0][1]], [cov[1][0], cov[1][1]]])
equation = Eq(det(Lambda*I-A), 0)
D = solve(equation)
print([N(element, 4) for element in D]) # Eigenvalues in decimal form
#print(pretty(D)) # Eigenvalues in exact form

m1= cov[0][0]
m2= cov[0][1]
m3= cov[1][0]
m4= cov[1][1]

#Visualizing and finding the Eigen Vectors------------------------------------------
v1= -(m3/(m1-D[0]))                                             #finding one eigev vector interms of the other 
v=[v1, 1]                                                       # assuming the other to be 1
print(v)
d= 1/(math.sqrt((v1**2+1))) # normalizing the vector 
v1=v1*d
v12 = 1*d 
V1=[v1,v12]                                                     # writing after nomalizing 
r1 = (0, 25*v1)                                                 # scaling the vector
r2 = (0, 25*v12)                                                # scaling the vector

plt.plot(r1,r2, color = 'r',label="Eigen-Vector 1")             #plaotting the eigen vector 


v2= -(m3/(m1-D[1]))                                             #finding one eigev vector interms of the other 
v=[v2, 1]                                                       # assuming the other to be 1
print(v)
d= 1/(math.sqrt((v2**2+1)))                                     # normalizing the vector
v2=v2*d
v22 = 1*d
V2=[v2,v22]                                                     # writing after nomalizing
r3 = (0, 100*v2)                                                # scaling the vector
r4 = (0, 100*v22)                                               # scaling the vector

plt.plot(r3,r4, color = 'g',label="Eigen-Vector 2",)            #plaotting the eigen vector 


plt.scatter(x1,y1, color = 'k')
plt.title("Visualizing Eigen-Vectors")
plt.legend()
plt.show()
 
#Part(ii): Least Square using Vertical Distances---------------------------------------


data1=np.column_stack((x1,y1))    				#Stacking the two lists together
d=(len(data1),1)   					        #Initializing an array of given dimensions
d=np.ones(d)   						        #Filling the array with ones throughout

A=np.column_stack((x1,d))   				        #Stacking x1 and d together in column
B=data1[:,[1]]   						#Extracting the second column of the list data1

S1=np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T) 		#Finding the Pseudo-Inverse of A

X=np.matmul(S1, B)   					        #Finding the values for ‘m’ and ‘c’ for the equation of a line
Y=X[0]*x1+X[1]                     			        #Finding the values of Y, to plot a line
plt.scatter(x1,y1, color = 'k')   			        #Plotting the points from the dataset
plt.plot(x1, Y, '-r',label="LS-V")   				#Plotting the Least Square line, using Vertical Distances

#Part(ii): least square with regularization----------------------------------------------------

q=np.eye(2)             					#constructing the identity matrix
eig,eigv = np.linalg.eig(cov) 				        #finding the eigen values
S1=np.matmul(np.linalg.inv(np.matmul(A.T,A)+np.matmul(eig,q)),A.T) #formual forregularization
X=np.matmul(S1, B)
Y=X[0]*x1+X[1] 					                #equation of the line
plt.scatter(x1,y1,color = 'k')
plt.plot(x1, Y, 'g',label="LS-R")    				#plotting the fitted line



#Part(ii): Least Square using Orthogonal Distances-----------------------------------------------------------


c1=x1-xmean1   					                #Subtracting the mean from the given dataset x
c2=y1-ymean1   					                #Subtracting the mean from the given dataset y
   	 
U=np.column_stack((c1,c2))    				        #Stacking the two columns, c1 & c2 together
N=np.matmul(U.T, U)    					        #Matrix Multiplication of U(transpose) with U



#u, s, v = np.linalg.svd(N, full_matrices=True)   		#Solving SVD to extract Eigen-Values
a=V1[0]   
b=V1[1]

#Equation of line between each data point and the best fit line ax + by = d---------------------------

d1=-(a*xmean1+b*ymean1)    				        #Solving for ‘d’ denoted by d1 here

p=d1/b   							#Finding the y-intercept of the line
q=-a/b   							#Finding the slope of the line

y2=[]   							#Creating an empty list
y2=q*z1+p   						        #Defining the y-coordinates to plot the line

                                                                #Plotting the points from the dataset
plt.plot(x1, y2, '-b',label="TLS")#Plotting the Least Square line, using Orthogonal Distances
plt.title("Least Square Line Fitting")
plt.legend()
plt.show()   						        #Displying the Graph


#Outlier reject using regularization--------------------------------
q=np.eye(2)             					#constructing the identity matrix
eig,eigv = np.linalg.eig(cov) 				        # finding the eigen values
S1=np.matmul(np.linalg.inv(np.matmul(A.T,A)+np.matmul(eig,q)),A.T) #formual forregularization
X=np.matmul(S1, B)
Y=X[0]*x1+X[1] 					                #equation of the line
#plt.scatter(x1,y1,color = 'k')
#plt.plot(x1, Y, 'g')    				        #plotting the fitted line


#eliminating points after regularization------------------------------------

count=0 # again putting count's value
inlier=[] #inlitializing the list
for i in range(0,200): # for loop for all the data points
    dis=((X[0]*z1[i])+((-1)*z2[i])+X[1])/(math.sqrt(X[0]**2 + (-1)**2)) # calculating distance
    if abs(dis)<10:  					 	# checking if distance less than 10
   	 inlier.append(i)   			                # appending the list
   	 count=count+1   				        # counting the points
  					         	 
#print(count)  						 


plt.plot(x1,Y,'g')     	                                        #plottig regularized line.
#plt.scatter(x1,y1, color = 'k')   	                        #plotting the data points  			 

#To eliminate the outliers--------------------------------------------------------------------------------------

for j in inlier:  						 
	plt.scatter(x1[j],y1[j],color = 'k')
plt.title("Outlier Rejection Using Regularization")
plt.show() 

# RANSAC---------------------------------------------------------------

#Part(iii): Outlier Rejection using RANSAC------------------------------------------------------------------


while True:    			                                #Running the loop continuously until the the minimum count is reached
        count = 0   		                                #initializing the count as zero
        inlier=[]   		                                #Creating a empty list of Inliers
        d1,d2= random.sample(data,2)    	                #Defining the dimension for d1 and d2
        for i in range(0,200):   			        #Creating a for loop to run ‘n’ iterations
                a1=-(d2[1]-d1[1])   				#Extracting the value for a1
                b1=(d2[0]-d1[0])   				#Extracting the value for b1
                c1=((-d2[0]+d1[0])*d1[1])+((d2[1]-d1[1])*d1[0]) #Extracting the value for c1
                M=((-a1)/b1)   			                #Finding the slope of from a1 and b1
                C=((-c1)/b1)   			                #Finding the y-intercept from c1 and b1
                x3=np.array([[min(z1)],[max(z1)]])              #Finding the max and min value on the x-axis
                y3=(M*x3)+C	                                #Finding the y coordinate from the values of x3 to plot a line
                dis=((a1*z1[i])+(b1*z2[i])+c1)/(math.sqrt(a1**2 + b1**2)) #Calculating the orthogonal distance
                if abs(dis)<10:  	                        #Defining the threshold value    
                                inlier.append(i)                #Appending the values satisfied in the threshold
                                count=count+1   	        #Increase the count in every loop
        if count>71:  		                                #Defining the minimum number of inliers before exiting the loop
                break   				        #Getting out of the loop


plt.title("RANSAC")
plt.plot(x3,y3,'r')    					        #Plotting the best fit line
plt.scatter(x1,y1, color = 'k')   				#Plotting the points from the dataset
plt.show()   						        #Displaying the graph
#print(len(inlier))   				                #Printing the length of inlier to cross verify

#To eliminate the outliers--------------------------------------------------------------------------------------
for j in inlier:  				                #Extracting the values from inlier
	plt.scatter(x1[j],y1[j],color = 'k')	                #Scatter plot the points obtained from inlier

plt.title("RANSAC excluding outliers")
plt.plot(x3,y3,'r')					        #Plotting the best fit line
plt.show()    						        #Displaying the graph

#Line fitting after RANSAC----------------------------
xn=[]
yn=[]
for j in inlier:  				                #Extracting the values from inlier
	#plt.scatter(x1[j],y1[j],color = 'k')
	xn.append(x1[j])
	yn.append(y1[j])
	
data1=np.column_stack((xn,yn))    				#Stacking the two lists together
d=(len(data1),1)   					        #Initializing an array of given dimensions
d=np.ones(d)   						        #Filling the array with ones throughout

A=np.column_stack((xn,d))   				        #Stacking x1 and d together in column
B=data1[:,[1]]   						#Extracting the second column of the list data1

S1=np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T) 		#Finding the Pseudo-Inverse of A

X=np.matmul(S1, B)   					        #Finding the values for ‘m’ and ‘c’ for the equation of a line
Y=X[0]*xn+X[1]                                                  #Finding the values of Y, to plot a line
plt.title("Line Fitting after RANSAC")
plt.scatter(xn,yn, color = 'k')   			        #Plotting the points from the dataset
plt.plot(xn, Y, 'r',label="LS-V")
plt.show()


