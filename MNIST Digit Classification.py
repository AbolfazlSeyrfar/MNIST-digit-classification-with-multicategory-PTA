
import numpy as np
import struct
import matplotlib.pyplot as plt

#-----------------------------------------Reading and Processing the MNIST files----------------------------------
def process_data():
    with open('./train-labels.idx1-ubyte', 'rb') as binary_file: 
        binary_file.seek(0) 
        magic_number = binary_file.read(4)
        nitems = binary_file.read(4)
        nitems = struct.unpack('>i',nitems)
        labl_training = np.fromfile(binary_file, np.uint8)
    with open('./train-images.idx3-ubyte', 'rb') as binary_file: 
        input_data = binary_file.read(16) 
        magic_number, train_n, rows, cols = struct.unpack('>iiii', input_data)
        img_training = np.fromfile(binary_file, np.uint8).reshape(train_n, rows, cols)
    with open('./t10k-labels.idx1-ubyte', 'rb') as binary_file:
        input_data = binary_file.read(8)
        magic_number, ntelabels = struct.unpack('>ii',input_data)
        labl_test = np.fromfile(binary_file, np.uint8) 
    with open('./t10k-images.idx3-ubyte', 'rb') as binary_file:
        input_data = binary_file.read(16)
        magic_number, testn, rows, cols = struct.unpack('>iiii', input_data)
        img_test = np.fromfile(binary_file, np.uint8).reshape(testn, rows, cols)
    return img_training, labl_training, train_n, img_test, labl_test, testn

#----------------------------------Define Activation Function, input (train and test sets), and output vectors-----------
def u(x):
    if x>=0:
        return 1
    else:
        return 0
    
global output_vectors
output_vectors = np.array([1,0,0,0,0,0,0,0,0,0], dtype=np.int)
output_r = lambda i: np.roll(output_vectors,i) 

img_training, labl_training, train_n, img_test, labl_test, testn = process_data()
errors = np.zeros(train_n)
x=np.empty((784,train_n))
x = x.T
vu = np.vectorize(u) 

#--------------------------------------------------------part d: Train the model------------------------------------------------------

# Define weights, learning rate, threshold, etc
eta=1
eps=0.15
n=12000
epoch = 0
W = np.random.uniform(-1,1,7840)
W = np.reshape(W,(10,784))
W = np.matrix(W)

flag=1;
while (flag):
    for i in range(n):
        x[i] = np.ravel(img_training[i])
        v = np.dot(W,x[i]) 
        j = np.argmax(v) 
        if(j != labl_training[i]):
            errors[epoch]+=1   
    print('Epoch Number: %d\Error: %f'%(epoch, errors[epoch]/n))
    epoch+=1
    for i in range(n):
        W = W + np.dot( eta*( output_r(labl_training[i]) - vu(np.dot(W,x[i]))  ).T, np.array(x[i]).reshape((1,784)) )
    if(errors[epoch-1]/n < eps):
        flag=0
print('learning rate=%f\tthreshold=%f\nNumber of Epochs: %d\nError Percentage: %f'%(eta, eps, epoch, errors[epoch-1]/n))


#-------------------------------------------------PART (e): Test the model -------------------------------------------------------------------

misclass_e = np.zeros(testn)        
def test(W): 
    misclass_e = 0
    for i in range(testn): 
        x[i] = np.ravel(img_test[i])
        v = np.dot(W,x[i])
        j = np.argmax(v) 
        if(j != labl_test[i]):
            misclass_e+=1
    print('#Misclassified: %d\nError Percentage: %f'%(misclass_e, misclass_e/10000))
    return misclass_e

test(W)


#--------------------PART (f): Train again with different settings to show an overfitted model----------------------------
l_rate_f=1
threshold=0.001
n_f=50
epoch_f = 0
errors = np.zeros(train_n)
W = np.random.uniform(-1,1,7840)
W = np.reshape(W,(10,784))
W = np.matrix(W)

flag=1;
while (flag):
    for i in range(n_f):
        x[i] = np.ravel(img_training[i])
        v = np.dot(W,x[i]) 
        j = np.argmax(v)
        if(j != labl_training[i]):
            errors[epoch_f]+=1
    epoch_f+=1
    for i in range(n_f):
        W = W + np.dot( l_rate_f*( output_r(labl_training[i]) - vu(np.dot(W,x[i]))  ).T, np.array(x[i]).reshape((1,784)) )
    if(errors[epoch_f-1]/n_f < threshold):
        flag=0
print('learning rate=%f\tthreshold=%f\nNumber of Epochs: %d\nError Percentage: %f'%(l_rate_f, threshold, epoch_f, errors[epoch_f-1]/n_f))

#Plot
plt.plot(np.linspace(0,epoch_f,epoch_f), errors[0:epoch_f], '-o')
plt.xlabel('Epoch Number')
plt.ylabel('Number of Misclassifications')


#for test set --------
misclass_e = 0
for i in range(testn): 
        x[i] = np.ravel(img_test[i])
        v = np.dot(W,x[i])
        j = np.argmax(v) 

        if(j != labl_test[i]):
            misclass_e+=1
print('#Misclassified: %d\tError Percentage: %f'%(misclass_e, misclass_e/10000))          


