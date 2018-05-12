import  NN3L
import numpy as np
import pandas as pd
def make_data_set(data):
	#first column is label Y
	n = data.shape[1]
	train_len = int(n*0.6)
	data_train = data[:,:train_len]
	data_test  = data[:,train_len:]
	X_train = data_train[1:,:]/255
	m = X_train.shape[1]
	X_train.reshape((784,m))
	Y_train = data_train[0,:]
	Y_train=np.reshape(Y_train,(1,m))

	Y_train_preprocess=OneHotEncoder(Y_train)
	X_test  = data_test[1:,:]/255
	n = X_test.shape[1]
	X_test.reshape((784,n))
	Y_test  = data_test[0,:]
	Y_test=np.reshape(Y_test,(1,n))
	return (X_train,Y_train_preprocess,Y_train,X_test,Y_test)

def OneHotEncoder(data):
	m = data.shape[1]
	print(data.shape)
	encoder =[[0]*10]*m
	Y = np.array(encoder)
	print(Y.shape)
	for i in range(m):
		Y[i,data[0,i]]=1
	return Y.T


model = NN3L.Network([784,100,100,10])
df = pd.read_csv('train.csv')
data = df.T 
#print(data.head())
data = data.values
#np.random.shuffle(data)
#print(data[0:5,0:5])
X_train,Y_train,Y_train_test,X_test,Y_test = make_data_set(data)
print('log: \n','*'*50)
model.train(X_train,Y_train,0.6,500)
print('*'*50)

Y_pre_train = model.predict_mult(X_train)
Y_pre_train = np.squeeze(Y_pre_train)
Y_train_test = np.squeeze(Y_train_test)

Y_pre_dev=model.predict_mult(X_test)
Y_pre_dev=np.squeeze(Y_pre_dev)
Y_test=np.squeeze(Y_test)
#print(Y_pre.shape,Y_test.shape)
from sklearn.metrics import accuracy_score
print("Train set accuracy: ",accuracy_score(Y_train_test,Y_pre_train)*100,"%")
print("Dev set accuracy :",accuracy_score(Y_test,Y_pre_dev)*100,"%")
model.save("NN3L_digit")

#Test dataset ------------------------------------------------------------------>