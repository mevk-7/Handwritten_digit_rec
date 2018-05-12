import numpy as np
import pandas as pd
import os 
class Network(object):
	#Initialize parameters
	def __init__(self,size):
		self.n_x = size[0]#input vector
		self.n_h1 = size[1]#hidden layer 1 units
		self.n_h2 = size[2]#hidden layer 2 units
		self.n_y =  size[3]#Output layer units
		
		self.W1 = np.random.randn(self.n_h1,self.n_x)*0.005
		self.b1 = np.zeros((self.n_h1,1))
		self.W2 = np.random.randn(self.n_h2,self.n_h1)*0.005
		self.b2 = np.zeros((self.n_h2,1))
		self.W3 = np.random.randn(self.n_y,self.n_h2)*0.005
		self.b3 = np.zeros((self.n_y,1))
	
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))
	def sigmoid_derv(self,a):
		return np.multiply(a,(1-a))

	def relu(self,z):
		return z * (z>0)
	def relu_derv(self,a):
		return 1*(a>0)


	def feed_forward(self,X):
		Z1 =  np.dot(self.W1,X)+self.b1
		A1 =  self.relu(Z1)
		Z2 =  np.dot(self.W2,A1) +self.b2
		A2 =  self.relu(Z2)
		Z3 =  np.dot(self.W3,A2) +self.b3
		A3 =  self.sigmoid(Z3)

		return (Z1,A1,Z2,A2,Z3,A3)

	def backprop(self,cache,Y,X):
		Z1,A1,Z2,A2,Z3,A3 =cache
		m = Y.shape[1]
		dZ3 = A3-Y
		dW3 = np.dot(dZ3,A2.T)/m
		db3 = np.sum(dZ3,axis=1,keepdims=True)/m

		dZ2 = np.dot(self.W3.T,dZ3)*self.relu_derv(A2)
		dW2 = np.dot(dZ2,A1.T)/m
		db2 = np.sum(dZ2,axis=1,keepdims=True)/m

		dZ1 = np.dot(self.W2.T,dZ2)*self.relu_derv(A1)
		dW1 = np.dot(dZ1,X.T)/m
		db1 = np.sum(dZ1,axis=1,keepdims=True)/m

		cost = self.compute_cost(Y,A3)
		
		return (dW1,db1,dW2,db2,dW3,db3,cost)

	def train(self,X,Y,learning_rate,epoch):
		#decay_rate = 0.02
		lambda_reg =0.0001
		m = Y.shape[1]
		for i in range(epoch):
			
			cache = self.feed_forward(X)

			#learning_rate = alpha/(1+decay_rate*i)
			dW1,db1,dW2,db2,dW3,db3,cost= self.backprop(cache,Y,X)
			#reguralize parameter lamda*alpha/m
			self.W1 =self.W1*(1-(lambda_reg*learning_rate/m))- learning_rate*dW1
			self.b1 -= learning_rate*db1
			self.W2 =self.W2*(1-(lambda_reg*learning_rate/m)) - learning_rate*dW2
			self.b2 -= learning_rate*db2
			self.W3 =self.W3*(1-(lambda_reg*learning_rate/m)) -  learning_rate*dW3
			self.b3 -= learning_rate*db3


			print("epoch :",i+1," cost :",cost)

	def predict(self,X_test):
		temp_Z1 =  np.dot(self.W1,X_test)+self.b1
		temp_A1 =  self.relu(temp_Z1)
		temp_Z2 =  np.dot(self.W2,temp_A1) +self.b2
		temp_A2 =  self.relu(temp_Z2)
		temp_Z3 =  np.dot(self.W3,temp_A2) +self.b3
		temp_A3 =  self.sigmoid(temp_Z3)
		#print(temp_A2)
		Out = 1 * (temp_A2>0.5)
		return Out
	def predict_mult(self,X_test):
		temp_Z1 =  np.dot(self.W1,X_test)+self.b1
		temp_A1 =  self.relu(temp_Z1)
		temp_Z2 =  np.dot(self.W2,temp_A1) +self.b2
		temp_A2 =  self.relu(temp_Z2)
		temp_Z3 =  np.dot(self.W3,temp_A2) +self.b3
		temp_A3 =  self.sigmoid(temp_Z3)
		#print(temp_A2)
		Out = np.argmax(temp_A3,axis=0)
		return Out

	def  compute_cost(self,Y,A3):
		try:
			logprobs = -np.multiply(np.log(A3),Y)-np.multiply(np.log(1-A3),1-Y)
		except:
			print("error :",A3)
		m = Y.shape[1]
		#print(Y.shape[1])
		cost = np.sum(logprobs)/m
		cost = np.squeeze(cost)
		return cost
	def compute_cost2(self,Y,A3):
		m = Y.shape[1]
		#A=np.argmax(A2,axis=0)
		cost = np.power(A3-Y,2)
		cost = np.sum(cost)/m

	def save(self,name):
		folder=name+" parameters"
		os.mkdir(folder)
		os.chdir(folder)
		np.savetxt("parameterW1",self.W1)
		np.savetxt("parameterb1",self.b1)
		np.savetxt("parameterW2",self.W2)
		np.savetxt("parameterb2",self.b2)
		np.savetxt("parameterW3",self.W3)
		np.savetxt("parameterb3",self.b3)

		
