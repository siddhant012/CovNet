import numpy as np
import math
import time

#functions for data preprocessing
def one_hot_encoded(num_of_classes,data,start_num=0):
    data=data.reshape([data.shape[0],1])
    new_data=np.zeros([data.shape[0],num_of_classes],dtype=data.dtype)
    for i in range(data.shape[0]):
        new_data[i][int(data[i][0])-start_num]=1
    return new_data

def normalize(data,max=None,min=None,axis=0):
    if(max is None or min is None):
        min=np.min(data,axis=axis)
        max=np.max(data,axis=axis)
        new_data=(data-min)/(max-min)
        return new_data,max,min
    else:
        new_data=(data-min)/(max-min)
        return new_data,max,min

def denormalize(data,max=None,min=None,axis=0):
    if(max is None or min is None):
        min=np.min(data,axis=axis)
        max=np.max(data,axis=axis)
    new_data=data*(max-min)+min
    return new_data

def standardize(data,mean=None,variance=None,axis=0):
    if(mean is None or variance is None):
        mean=data.mean(axis=axis)
        variance=((data**2).mean(axis=axis))**0.5
        epsilon=1.0e-8
        new_data=(data-mean)/(variance+epsilon)
        return new_data,mean,variance
    else:
        epsilon=1.0e-8
        new_data=(data-mean)/(variance+epsilon)
        return new_data,mean,variance

def destandardize(data,mean=None,variance=None,axis=0):
    if(mean is None or variance is None):
        mean=data.mean(axis=axis)
        variance=((data**2).mean(axis=axis))**0.5
    epsilon=1.0e-8
    new_data=data*(variance+epsilon)+mean
    return new_data

def shuffle(input_data,output_data):
    shuffled_data=np.append(input_data,output_data,axis=1)
    np.random.seed(0)
    np.random.shuffle(shuffled_data)
    input_data,output_data=shuffled_data[:,:-1],shuffled_data[:,-1:]
    return input_data,output_data


#deleted yet some useful functions
'''def acti_derivative(acti_type,num):
    if(acti_type=="sigmoid"):
        if(num<-100):num=-100
        if(num>100):num=100
        t=1/(1+np.exp(-num))
        return t*(1-t)
    elif(acti_type=="relu"):
        if(num>0) : return 1
        else : return 0
    elif(acti_type=="leaky_relu"):
        if(num>0) : return 1
        else : return 0.1
    elif(acti_type=="linear"):
        return num
    elif(acti_type=="softmax"):
        return None
    else:
        pass
acti_derivative=np.vectorize(acti_derivative,otypes=[float])

def acti(acti_type,num):
    if(acti_type=="sigmoid"):
        if(num<-100):num=-100
        if(num>100):num=100
        return 1/(1+np.exp(-num))
    elif(acti_type=="relu"):
        return max(0,num)
    elif(acti_type=="leaky_relu"):
        return max(num/10,num)
    elif(acti_type=="linear"):
        return num
    elif(acti_type=="softmax"):
        return None
    else:
        pass
acti=np.vectorize(acti,otypes=[float])

def softmax_acti(arr):     #exclusive function to calculate softmax activation function since it is different from other activation functions
    def apply_exp(num):
        if(num>100):num=100
        if(num<-100):num=-100
        return np.exp(num)
    apply_exp=np.vectorize(apply_exp,otypes=[float])
    temp1=apply_exp(arr)
    temp2=temp1.sum(axis=0).reshape(1,len(arr[0]))
    arr=temp1*(1/temp2)
    return arr'''

'''def apply_padding(inp_images,padding=1):
    out_images=np.zeros([inp_images.shape[0],inp_images.shape[1],inp_images.shape[2]+2*padding,inp_images.shape[3]+2*padding])
    out_images[:,:,padding:out_images.shape[2]-padding,padding:out_images.shape[3]-padding] +=inp_images
    return out_images'''

'''def remove_padding(inp_images,padding=1):
    out_images=inp_images[:,:,padding:inp_images.shape[2]-padding,padding:inp_images.shape[3]-padding]
    return out_images'''

'''def softmax_acti(arr):     #exclusive function to calculate softmax activation function since it is different from other activation functions
    arr[arr>100.0]=100.0
    arr[arr<-100.0]=-100.0
    temp1=np.exp(arr)
    temp2=temp1.sum(axis=0).reshape(1,len(arr[0]))
    return temp1*(1/temp2)'''


#functions used in forward prop and back prop algorithms
def acti(acti_type,arr):
    if(acti_type=="sigmoid"):
        arr=np.clip(arr,-80,80)
        temp=1/(1+np.exp(-arr))
        return temp
    elif(acti_type=="relu"):
        return np.maximum(arr,0)
        #arr[arr<0.0]=0.0
        #return arr
    elif(acti_type=="leaky_relu"):
        arr[arr<0]=arr/10
        return arr
    elif(acti_type=="linear"):
        return arr
    elif(acti_type=="softmax"):
        arr=np.clip(arr,-80,80)
        temp1=np.exp(arr)
        temp2=temp1.sum(axis=0).reshape(1,len(arr[0]))
        return temp1/temp2
    elif(acti_type=="tanh"):
        arr=np.clip(arr,-80,80)
        temp=1/np.exp(arr)**2
        return (1-temp)/(1+temp)
    else:
        return None

def acti_derivative(acti_type,arr):
    if(acti_type=="sigmoid"):
        arr=np.clip(arr,-80,80)
        temp=1/(1+np.exp(-arr))
        return temp*(1-temp)
    elif(acti_type=="relu"):
        return (np.greater(arr, 0).astype(int)).astype(arr.dtype)
        #arr[arr<0.0]=0.0
        #arr[arr>0.0]=1.0
        #arr[arr==0.0]=0.0
        #return arr
    elif(acti_type=="leaky_relu"):
        arr[arr<0.0]=0.1
        arr[arr>0.0]=1.0
        arr[arr==0.0]=1.0
        return arr
    elif(acti_type=="linear"):
        return arr
    elif(acti_type=="softmax"):
        return None
    elif(acti_type=="tanh"):
        arr=np.clip(arr,-80,80)
        temp=1/( np.exp(arr) )**2
        return 1-( (1-temp)/(1+temp) )**2
    else:
        return None

'''def arr_to_vec(x,filter_shape,stride=1,padding=0):
    f1,f2=filter_shape
    num_of_images,depth,in1,in2=x.shape

    temp=np.zeros([num_of_images,depth,in1+2*padding,in2+2*padding],dtype=x.dtype)
    temp[:,:,padding:in1+padding+1,padding:in2+padding+1]+=x
    x=temp
    #if(padding) : x=apply_padding(x,padding)

    valid_idx=[(i,j) for i in range(in1+2*padding-f1+1) for j in range(in2+2*padding-f2+1) if j%stride==0 and i%stride==0]
    new_x=np.concatenate( [ x[:,:,i:i+f1,j:j+f2].reshape(num_of_images,1,f1*f2*depth) for i,j in valid_idx ] ,axis=1)
    #new_x=np.concatenate( [ x[:,:,i:i+f1,j:j+f1].reshape(1,num_of_images*f1*f2*depth) for i,j in valid_idx ] ,axis=0)
    #return new_x.transpose().reshape(num_of_images,f1*f2*depth,((in1+2*padding-f1)//stride+1)*((in2+2*padding-f2)//stride+1)).transpose(0,2,1)

    return new_x'''

'''def vec_to_arr(x,filter_shape,new_shape,stride=1,padding=0):    #here the padding and stride are those used while converting the original image to vectorized form 
    f1,f2=filter_shape
    n,p=new_shape
    r,s=(n+2*padding-f1)//stride + 1 , (p+2*padding-f2)//stride + 1
    num_of_images,conv_num,conv_size=x.shape
    depth=conv_size//(f1*f2)

    new_x=np.zeros([num_of_images,depth,new_shape[0]+2*padding,new_shape[1]+2*padding],dtype=x.dtype)     #with padding applied
    #new_x=np.zeros([num_of_images,depth,new_shape[0],new_shape[1]])
    #if(padding) : new_x=apply_padding(new_x,padding)

    def func(i):
        a,b=divmod(i,r)
        ri,ci=a*stride,b*stride
        new_x[:,:,ri:ri+f1,ci:ci+f2]+= x[:,i:i+1,:].reshape(num_of_images,depth,f1,f2)

    func=np.vectorize(func,otypes=[x.dtype])
    indices=np.arange(r*s)
    func(indices)

    #if(padding) : new_x=remove_padding(new_x,padding)
    return new_x[:,:,padding:new_x.shape[2]-padding,padding:new_x.shape[3]-padding]     #remove padding'''

def arr_to_vec(x,filter_shape,stride=1,padding=0):
    f1,f2=filter_shape
    num_of_images,depth,in1,in2=x.shape
    out1,out2=((in1+2*padding-f1)//stride+1) , ((in2+2*padding-f2)//stride+1)

    x=np.pad(x,[(0,0),(0,0),(padding,padding),(padding,padding)],mode="constant")
    valid_idx=[(i+k1)*(in2+2*padding)+j+k2 +d*(in1+2*padding)*(in2+2*padding) for i in range(in1+2*padding-f1+1) for j in range(in2+2*padding-f2+1) if (j%stride==0 and i%stride==0) for d in range(depth) for k1 in range(f1) for k2 in range(f2)]
    new_x=x.reshape(num_of_images,-1).take(valid_idx,axis=1).reshape(num_of_images,out1*out2,depth*f1*f2)
    return new_x

def vec_to_arr(x,filter_shape,new_shape,stride=1,padding=0):    #here the padding and stride are those used while converting the original image to vectorized form 
    f1,f2=filter_shape
    n,p=new_shape
    r,s=(n+2*padding-f1)//stride + 1 , (p+2*padding-f2)//stride + 1
    num_of_images,conv_num,conv_size=x.shape
    depth=conv_size//(f1*f2)
    d_type=np.float32
    x_d_type=x.dtype
    x=x.astype(d_type)

    new_x=np.zeros([num_of_images,depth,new_shape[0]+2*padding,new_shape[1]+2*padding],dtype=d_type)     #with padding applied

    indices=np.arange(r*s)
    r_c=[( (i//r)*stride,(i%r)*stride ) for i in indices]
    x=x.reshape(num_of_images,conv_num,depth,f1,f2).transpose(1,0,2,3,4)

    def func(i):
        ri,ci=r_c[i]
        new_x[:,:,ri:ri+f1,ci:ci+f2]+= x[i]

    func=np.vectorize(func,otypes=[d_type])
    func(indices)
    return new_x[:,:,padding:new_x.shape[2]-padding,padding:new_x.shape[3]-padding].astype(x_d_type)     #remove padding


#optimizers
class adam:
    def __init__(self,beta1=0.9,beta2=0.99,t=2,epsilon=1.0e-8):
        self.beta1=beta1
        self.beta2=beta2
        self.t=t
        self.epsilon=epsilon
        self.num_of_layers=0
        self.dB1=None
        self.dB2=None
        self.dW1=None
        self.dW2=None
    def load(self,num_of_layers,d_type=np.float32):
        self.num_of_layers=num_of_layers
        self.d_type=d_type
        self.dB1=[0]*self.num_of_layers
        self.dB2=[0]*self.num_of_layers
        self.dW1=[0]*self.num_of_layers
        self.dW2=[0]*self.num_of_layers
    def calculate_parameters(self,dB,dW,i):
        self.dB1[i] = self.beta1*self.dB1[i] + (1-self.beta1)*dB
        self.dB2[i] = self.beta2*self.dB2[i] + (1-self.beta2)*(dB**2)
        self.dW1[i] = self.beta1*self.dW1[i] + (1-self.beta1)*dW
        self.dW2[i] = self.beta2*self.dW2[i] + (1-self.beta2)*(dW**2)
        dB1=self.dB1[i]/(1-self.beta1**self.t)
        dW1=self.dW1[i]/(1-self.beta1**self.t)
        dB2=self.dB2[i]/(1-self.beta2**self.t)
        dW2=self.dW2[i]/(1-self.beta2**self.t)
        return ( dB1/((dB2**0.5)+self.epsilon),dW1/((dW2**0.5)+self.epsilon) )

class rms_prop:
    def __init__(self,beta2=0.99,epsilon=1.0e-8):     #learning rate has to be very less in case of rms_prop and adam optimizers
        self.beta2=beta2
        self.epsilon=epsilon
        self.dW=None
        self.dB=None
        self.num_of_layers=0
    def load(self,num_of_layers,d_type=np.float32):
        self.num_of_layers=num_of_layers
        self.d_type=d_type
        self.dB=[0]*self.num_of_layers
        self.dW=[0]*self.num_of_layers
    def calculate_parameters(self,dB,dW,i):
        self.dB[i] = self.beta2*self.dB[i] + (1-self.beta2) * (dB**2)
        self.dW[i] = self.beta2*self.dW[i] + (1-self.beta2) * (dW**2)
        return ( dB/((self.dB[i]**0.5)+self.epsilon),dW/((self.dW[i]**0.5)+self.epsilon) )

class momentum:
    def __init__(self,beta1=0.9):
        self.beta1=beta1
    def load(self,num_of_layers,d_type=np.float32):
        self.num_of_layers=num_of_layers
        self.d_type=d_type
        self.dB=[0]*self.num_of_layers
        self.dW=[0]*self.num_of_layers
    def calculate_parameters(self,dB,dW,i):
        self.dB[i] = self.beta1*self.dB[i] + (1-self.beta1)*dB
        self.dW[i] = self.beta1*self.dW[i] + (1-self.beta1)*dW
        return (self.dB[i],self.dW[i])

#learning rate decay methods
class expo_decay:
    def __init__(self,k):     #k is a number less than 1
        self.k=k
    def new_lr(self,lr,epoch):
        return (self.k**epoch)*lr

class normal_decay:
    def __init__(self,decay_rate):
        self.decay_rate=decay_rate
    def new_lr(self,lr,epoch):
        return lr*(1/(1+lr*self.decay_rate))

class sqr_root_decay:
    def __init__(self,k):
        self.k=k
    def new_lr(self,lr,epoch):
        return (self.k/(epoch**0.5))*lr

class staircase_decay:
    def __init__(self,num_of_epochs):
        self.num_of_epochs=num_of_epochs
    def new_lr(self,lr,epoch):
        return lr/2 if epoch%self.num_of_epochs==0 else lr

class linear_decay:
    def __init__(self,num_of_epochs,k):
        self.num_of_epochs=num_of_epochs
        self.k=k
    def new_lr(self,lr,epoch):
        return lr/self.k if epoch%self.num_of_epochs==0 else lr

#regularizations (dropout is implemented as a separate layer)
class l2:
    def __init__(self,lamda):
        self.lamda=lamda
    def load(self,d_type):
        self.d_type=d_type
    def reg_cost(self,layers):
        batch_size=layers[-1].A.shape[1]
        sum=0.0
        for i in range(len(layers)):
            if(layers[i].layer_name in ("pool","drp","flatten","batch_norm")) : continue
            if(layers[i].layer_name=="fc") : sum+=(layers[i].W**2).sum()
            elif(layers[i].layer_name=="conv") : sum+=(layers[i].F**2).sum()
            elif(layers[i].layer_name=="batch_norm") : sum+=(layers[i].Y**2).sum()
        return (self.lamda/(2*batch_size))*sum
    def add_to_weights(self,W,batch_size):
        return (self.lamda/batch_size)*W

class l1:
    pass

#initializers
class zeros:
    def __init__(self):
        pass
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,*args):
        return np.zeros(shape,dtype=self.d_type)

class ones:
    def __init__(self):
        pass
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,*args):
        return np.ones(shape,dtype=self.d_type)

class constant:
    def __init__(self,value):
        self.value=value
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,*args): 
        return np.full(shape,self.value,dtype=self.d_type)

class random_normal:
    def __init__(self,mean,stdev,seed=None):
        self.mean=mean
        self.stdev=stdev
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.normal(loc=self.mean,scale=self.stdev,size=shape).astype(self.d_type)   #loc=mean,scale=standard deviation and size=shape
        return out

class random_uniform:
    def __init__(self,minimum_val,maximum_val,seed=None):
        self.minimum_val=minimum_val
        self.maximum_val=maximum_val
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.uniform(low=self.minimum_val,high=self.maximum_val,size=shape).astype(self.d_type)
        return out
    
class he_normal:
    def __init__(self,seed=None):
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,in_neurons,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.normal(loc=0,scale=(2/in_neurons)**0.5,size=shape).astype(self.d_type)
        return out

class he_uniform:
    def __init__(self,seed=None):
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,in_neurons,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.normal(low=-(6/in_neurons)**0.5,high=(6/in_neurons)**0.5,size=shape).astype(self.d_type)
        return out

class glorot_normal:     #xavier's distribution
    def __init__(self,seed=None):
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,in_neurons,out_neurons,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.normal(loc=0,scale=(2/(in_neurons+out_neurons))**0.5,size=shape).astype(self.d_type)
        return out

class glorot_uniform:     #xavier's distribution
    def __init__(self,seed=None):
        self.seed=seed
    def load(self,d_type=np.float32):
        self.d_type=d_type
    def init(self,shape,in_neurons,out_neurons,*args):
        if self.seed is not None : np.random.seed(self.seed)
        out=np.random.uniform(low=-(6/(in_neurons+out_neurons))**0.5,high=(6/(in_neurons+out_neurons))**0.5,size=shape).astype(self.d_type)
        return out


class CovNet:

    def __init__(self):
        self.layers=[]
        self.num_of_epochs=0
        self.batch_size=0
        self.learning_rate=0.0
        self.input_data=None
        self.output_data=None
        self.regularization=l2(0.0)
        self.lr_decay=normal_decay(0.0)
        self.optimizer=adam()
        self.cost_function_type="categorical_cross_entropy"     #various possible types are binary_cross_entropy,mean_absolute_error,mean_square_error,softmax etc
        self.display_type="batch_type"     #various possible types are batch_type or epoch_type


        self.epsilon=1.0e-6     #a small value added to denominators during divisions to prevent division by zero error
        self.avg_rmse=0.0     #the average rmse over several mini batches/batches
        self.avg_accuracy=0.0     #the average accuracy over several mini batches/batches
        self.avg_cost=0.0     #the average cost over several mini batches/batches
        self.precision=7     #the maximum precision with which to display output values except in gradient checking which requires a lot more precision

        #current variables
        self.num_of_batches=0
        self.curr_batch_size=0
        self.curr_epoch=0
        self.curr_batch=0
        self.curr_ind=0     #starting index of the current batch in the input data
        self.curr_inp=None     #current input data
        self.curr_out=None     #current real output data
        self.d_type=np.float32     #the dtype of the numpy arrays used.Higher dtypes means more accuracy/performance and lower dtypes means more speed.
                                   #It sets itself according to the dtype of the input data passed and is used to maintain same dtype across the whole code.


    def calculate_rmse(self,real_data,predicted_data):
        temp=( (real_data-predicted_data)**2 ).sum()/real_data.shape[0]
        return (temp**0.5)

    def calculate_accuracy(self,real_data,predicted_data):
        if(self.layers[-1].acti_type!="sigmoid" and self.layers[-1].acti_type!="softmax") : return 0     #means it is not classification but regression problem
        if(self.layers[-1].acti_type=="sigmoid" and self.layers[-1].num_of_neurons>1) : return 0    #classification problem but the value of output layer for different classes not understood

        if(self.layers[-1].acti_type=="sigmoid" and self.layers[-1].num_of_neurons==1):
            predicted_data=predicted_data.reshape(-1,)
            real_data=real_data.reshape(-1,)
            predicted_data=np.round(predicted_data)
            non_zero=np.count_nonzero( np.abs(predicted_data-real_data) )
            return (predicted_data.shape[0]-non_zero)/real_data.shape[0]

        elif(self.layers[-1].acti_type=="softmax"):
            indices=np.argmax(predicted_data,axis=1).reshape(real_data.shape[0],1)
            vals=np.take_along_axis(real_data,indices,axis=1)
            return np.count_nonzero(vals)/real_data.shape[0]

    def calculate_cost(self,real_data,predicted_data):

        if(self.cost_function_type=="binary_cross_entropy"):
            cost=( ( (real_data*-1) * np.log(predicted_data+self.epsilon) ) + (1-real_data) * np.log(1-predicted_data+self.epsilon) ).sum()/real_data.shape[0]

        elif(self.cost_function_type=="mean_square_error"):
            cost=( (predicted_data-real_data)**2 ).sum()/real_data.shape[0]

        elif(self.cost_function_type=="categorical_cross_entropy"):
            cost=( (real_data*-1) * np.log(predicted_data+self.epsilon) ).sum()/real_data.shape[0]

        elif(self.cost_function_type=="sparse_categorical_cross_entropy"):
            pass

        elif(self.cost_function_type=="mean_absolute_error"):
            cost=np.abs(predicted_data-real_data).sum()/real_data.shape[0]
        
        cost+= self.regularization.reg_cost(self.layers) if self.regularization is not None else 0
        return cost

    def display_info(self):
        real_data=self.curr_out
        predicted_data=self.layers[-1].A.T

        batch_rmse=self.calculate_rmse(real_data,predicted_data)
        batch_accuracy=self.calculate_accuracy(real_data,predicted_data)
        batch_cost=self.calculate_cost(real_data,predicted_data)
        
        pr=self.precision     #because the name self.precision is too long
        sp=" "*3*pr     #spaces after printing the learning rate to deal with the carriage return problem


        if(self.curr_batch==1) : self.avg_rmse,self.avg_cost,self.avg_accuracy = 0.0 , 0.0 , 0.0
        self.avg_cost= (self.avg_cost*self.curr_ind+batch_cost*self.curr_batch_size) / (self.curr_ind+self.curr_batch_size)
        self.avg_rmse= ( ((self.avg_rmse**2)*self.curr_ind+batch_rmse*self.curr_batch_size) / (self.curr_ind+self.curr_batch_size) )**0.5
        self.avg_accuracy= (self.avg_accuracy*self.curr_ind+batch_accuracy*self.curr_batch_size) / (self.curr_ind+self.curr_batch_size)


        if(self.display_type=="batch_type"):
            if(self.curr_batch==1) : print("\nEPOCH ","[",self.curr_epoch,"/ ",self.num_of_epochs,"]",":")
            print("batch ","[",self.curr_batch,"/",self.num_of_batches,"]",":","  ","batch rmse=",round(batch_rmse,self.precision),"  ","batch cost=",round(batch_cost,self.precision),"  ","batch accuracy:",round(batch_accuracy,self.precision),"  ","learning rate=",round(self.learning_rate,self.precision))
            if(self.curr_batch==self.num_of_batches):
                print("average rmse:",round(self.avg_rmse,self.precision),"  ","average cost:",round(self.avg_cost,self.precision),"  ","average accuracy:",round(self.avg_accuracy,self.precision))

        elif(self.display_type=="epoch_type"):
            if(self.curr_batch==1) : print("\n\nepoch ",self.curr_epoch,"/",self.num_of_epochs,":")
            print("[",self.curr_ind+self.curr_batch_size,"/",self.input_data.shape[0],"]"," ","[",self.curr_batch,"/",self.num_of_batches,"]"
            ,"  ","average rmse:",round(self.avg_rmse,self.precision),"  ","average cost:",round(self.avg_cost,self.precision),"  ","average accuracy",round(self.avg_accuracy,self.precision),"  ","learning rate:",round(self.learning_rate,self.precision),sp,end="\r")

    def dA_from_dJ(self,real_data,predicted_data):

        if(self.cost_function_type=="binary_cross_entropy"):
            dA= ( -real_data/(predicted_data+self.epsilon) + (1-real_data)/(1-predicted_data+self.epsilon) ) / real_data.shape[0]
            return dA

        elif(self.cost_function_type=="mean_square_error"):
            dA=(predicted_data-real_data) * (2/real_data.shape[0])
            return dA

        elif(self.cost_function_type=="categorical_cross_entropy"):
            dA=predicted_data-real_data
            #dA=-real_data*(1/(predicted_data+self.epsilon))
            return dA

        elif(self.cost_function_type=="mean_absolute_error"):
            dA=np.ones(real_data.shape,dtype=self.d_type)/real_data.shape[0]
            return dA

        else:
            return None

    def load_network(self):

        #variables initialization
        self.num_of_batches=math.ceil(self.input_data.shape[0]/self.batch_size)
        self.d_type=self.input_data.dtype

        #loading the optimizers,initializers and regularizations
        if(self.optimizer is not None) : self.optimizer.load(len(self.layers),self.d_type)     #loading the optimizer
        for lyr in self.layers:     #loading the initializers
            if(lyr.layer_name in ("flatten","drp","pool")) : continue
            if(lyr.weight_init is not None) : lyr.weight_init.load(self.d_type)
            if(lyr.bias_init is not None) : lyr.bias_init.load(self.d_type)
        if(self.regularization is not None) : self.regularization.load(self.d_type)     #loading the regularization

        #parameter initialization
        input_shape=[None]+list(self.input_data.shape[1:])
        for lyr in self.layers : input_shape=lyr.init_parameters(input_shape)

        #providing training data to batch_norm layer to calculate mean and variance
        for lyr in self.layers:
            if(lyr.layer_name=="batch_norm") : lyr.set_mean_var(self.input_data)     #batch norm layer calculates training data mean and variance to be used during testing


    def train(self):

        #changing status of dropout and batch_norm layers to training
        for lyr in self.layers: 
            if(lyr.layer_name in ("batch_norm","drp")) : lyr.status="training"     #changing status to training

        print("training...")

        for ep in range( self.num_of_epochs*math.ceil(self.input_data.shape[0]/self.batch_size) ):

            #current variables calculation
            self.curr_batch=ep%self.num_of_batches+1
            self.curr_epoch=ep//self.num_of_batches+1
            self.curr_batch_size=min(self.curr_batch*self.batch_size,self.input_data.shape[0])-(self.curr_batch-1)*self.batch_size
            self.curr_ind=(self.curr_batch-1)*self.batch_size
            self.curr_inp=self.input_data[self.curr_ind:self.curr_ind+self.curr_batch_size]
            self.curr_out=self.output_data[self.curr_ind:self.curr_ind+self.curr_batch_size]

            #forward pass
            if(self.layers[0].layer_name in("conv","pool","flatten")) : prev_A=self.curr_inp
            elif(self.layers[0].layer_name in("fc")) : prev_A=self.curr_inp.T
            for lyr in range(0,len(self.layers)):
                prev_A=self.layers[lyr].forward_pass(prev_A)

            #backward pass
            dA=self.dA_from_dJ( self.curr_out,self.layers[-1].A.T ).T
            for lyr in range(len(self.layers)-1,-1,-1):
                dA=self.layers[lyr].backward_pass(dA)

            #learning rate update
            if(self.curr_batch==self.num_of_batches) : self.learning_rate=self.lr_decay.new_lr(self.learning_rate,ep) if self.lr_decay is not None else self.learning_rate

            #update parameters
            for lyr in range(0,len(self.layers)):
                self.layers[lyr].update_parameters(self.learning_rate,optimizer=self.optimizer,regularization=self.regularization,lyr_no=lyr)

            #display info
            self.display_info()

    def model_summary(self):
        H="Model Summary"   #main heading
        h1="Layer Name"     #sub heading 1
        s1="          "     #spaces 1
        h2="Output Shape"   #sub heading 2
        s2="          "     #spaces 2
        h3="Parameters"     #sub heading 3

        sp1=len(h1)+len(s1)     #variable for spaces
        sp2=len(h2)+len(s2)     #variable for spaces

        print("\n\n")
        print(H)
        print(h1,s1,h2,s2,h3)

        tot_para_num=0
        para_num=0
        for lyr in self.layers:
            if(lyr.layer_name=="conv") : para_num=np.prod(lyr.F.shape)+np.prod(lyr.B.shape)
            elif(lyr.layer_name=="fc") : para_num=np.prod(lyr.W.shape)+np.prod(lyr.B.shape)
            elif(lyr.layer_name=="drp") : para_num=0
            elif(lyr.layer_name=="flatten") : para_num=0
            elif(lyr.layer_name=="pool") : para_num=0
            elif(lyr.layer_name=="batch_norm") : para_num=np.prod(lyr.Y.shape)+np.prod(lyr.B.shape)
            else : pass
            tot_para_num+=para_num
            print(lyr.layer_name," "*(sp1-len(lyr.layer_name)),lyr.output_shape," "*(sp2-len(str(lyr.output_shape))),para_num)
        print("\n")
        print("total parameters:",tot_para_num,end="\n\n")

    def test(self,input_data,output_data,batch_size=128):

        #changing status of dropout and batch_norm layers to testing
        for lyr in self.layers:
            if(lyr.layer_name in ("batch_norm","drp")) : lyr.status="testing"

        batch_rmse,batch_cost,batch_accuracy=0.0,0.0,0.0
        avg_rmse,avg_cost,avg_accuracy=0.0,0.0,0.0
        batch_input,batch_output=None,None

        print("\nrunning for test data...")

        for batch in range(math.ceil(input_data.shape[0]/batch_size)):
            batch_input=input_data[batch*batch_size:min((batch+1)*batch_size,input_data.shape[0])]
            batch_output=output_data[batch*batch_size:min((batch+1)*batch_size,input_data.shape[0])]

            if(self.layers[0].layer_name in("conv","pool","flatten")) : prev_A=batch_input
            elif(self.layers[0].layer_name in("fc")) : prev_A=batch_input.T
            for lyr in range(0,len(self.layers)):
                prev_A=self.layers[lyr].forward_pass(prev_A)

            batch_rmse=self.calculate_rmse(batch_output,self.layers[-1].A.T)
            batch_cost=self.calculate_cost(batch_output,self.layers[-1].A.T)
            batch_accuracy=self.calculate_accuracy(batch_output,self.layers[-1].A.T)

            avg_rmse+= (batch_rmse**2)*self.layers[-1].A.shape[1]
            avg_cost+= batch_cost*self.layers[-1].A.shape[1]
            avg_accuracy+= batch_accuracy*self.layers[-1].A.shape[1]
        
        avg_rmse=(avg_rmse/input_data.shape[0])**0.5
        avg_cost=avg_cost/input_data.shape[0]
        avg_accuracy=avg_accuracy/input_data.shape[0]

        print("for test data:")
        print("average rmse:",round(avg_rmse,self.precision),"  ","average cost:",round(avg_cost,self.precision),"  ","average accuracy:",round(avg_accuracy,self.precision),end="\n\n")

    def predict(self,input_data):
        #changing status of dropout and batch_norm layers to testing
        for lyr in self.layers:
            if(lyr.layer_name in ("batch_norm","drp")) : lyr.status="testing"

        input_data=np.array([input_data],dtype=self.d_type)

        if(self.layers[0].layer_name in("conv","pool","flatten")) : prev_A=input_data
        elif(self.layers[0].layer_name in("fc")) : prev_A=input_data.T
        for lyr in range(0,len(self.layers)) : prev_A=self.layers[lyr].forward_pass(prev_A)
        print("predicted output:")
        print(np.round(self.layers[-1].A.T.reshape(-1,),self.precision),end="\n\n")

    def grad_check(self,sample_input=None,sample_output=None,epsilon=1.0e-4):

        #changing status of dropout and batch_norm layers to grad_check
        for lyr in self.layers:
            if(lyr.layer_name in ("batch_norm","drp")) : lyr.status="grad_check"
        
        #make sure that training is not done with dropout keep_probs<1.0 with l2/l1regularization also present otherwise grad_check will give incorrect results because training with dropout<1 alters weights which alters cost calculation due to regularization

        print("running gradient checking...")

        #initializing local variables
        total_para=0
        curr_para=0
        real_grads,predicted_grads=[],[]
        weights,bias=None,None
        if(sample_input is None) : sample_input=self.input_data[0:10]
        if(sample_output is None) : sample_output=self.output_data[0:10]

        #calculating total parameters
        for lyr in self.layers:
            if(lyr.layer_name in ("pool","drp","flatten")) : continue
            if(lyr.layer_name=="fc") : total_para+=np.prod(lyr.W.shape)+np.prod(lyr.B.shape)
            elif(lyr.layer_name=="conv") : total_para+=np.prod(lyr.F.shape)+np.prod(lyr.B.shape)
            elif(lyr.layer_name=="batch_norm") : total_para+=np.prod(lyr.Y.shape)+np.prod(lyr.B.shape)

        #defining local functions to be used
        def forward_pass():
            if(self.layers[0].layer_name in("conv","pool","flatten")) : prev_A=sample_input
            elif(self.layers[0].layer_name in("fc")) : prev_A=sample_input.T
            for lyr in self.layers : prev_A=lyr.forward_pass(prev_A)
        
        def backward_pass():
            dA=self.dA_from_dJ(sample_output,self.layers[-1].A.T).T
            for lyr in range(len(self.layers)-1,-1,-1) : dA=self.layers[lyr].backward_pass(dA)

        def display_info():
            print("[",curr_para,"/",total_para," parameters","]","          ",end="\r")

        forward_pass()
        backward_pass()

        #iterating through all layers
        for lyr in self.layers:

            if(lyr.layer_name in ("pool","drp","flatten")) : continue
            if(lyr.layer_name=="fc"):
                weights,bias=lyr.W,lyr.B
                real_grads+=list(lyr.dW.reshape(-1,))
                real_grads+=list(lyr.dB.reshape(-1,))
            elif(lyr.layer_name=="conv"): 
                weights,bias=lyr.F,lyr.B
                real_grads+=list(lyr.dF.reshape(-1,))
                real_grads+=list(lyr.dB.reshape(-1,))
            elif(lyr.layer_name=="batch_norm"):
                weights,bias=lyr.Y,lyr.B
                real_grads+=list(lyr.dY.reshape(-1,))
                real_grads+=list(lyr.dB.reshape(-1,))


            for w in np.nditer(weights,op_flags=[['readwrite']]):
                w-=epsilon
                forward_pass()
                J1=self.calculate_cost(sample_output,self.layers[-1].A.T)
                w+=2*epsilon
                forward_pass()
                J2=self.calculate_cost(sample_output,self.layers[-1].A.T)
                predicted_grads.append((J2-J1)/(2*epsilon))
                w-=epsilon
                curr_para+=1
                display_info()

            for b in np.nditer(bias,op_flags=[['readwrite']]):
                b-=epsilon
                forward_pass()
                J1=self.calculate_cost(sample_output,self.layers[-1].A.T)
                b+=2*epsilon
                forward_pass()
                J2=self.calculate_cost(sample_output,self.layers[-1].A.T)
                predicted_grads.append((J2-J1)/(2*epsilon))
                b-=epsilon
                curr_para+=1
                display_info()

        real_grads=np.array(real_grads,dtype=self.d_type)
        predicted_grads=np.array(predicted_grads,dtype=self.d_type)
        mean_abs_error=np.abs(real_grads-predicted_grads).mean()
        norm_error=((real_grads-predicted_grads)**2).sum()**0.5 / ((real_grads+predicted_grads)**2).sum()**0.5     #alternate formulas for the norm error are used as well

        print("[",curr_para,"/",total_para," parameters","]")
        print("Errors between actual derivatives and the derivatives calculated by backpropogation:")
        print("mean absolute error:",mean_abs_error)
        print("norm error:",norm_error)



class Conv:

    def __init__(self):
        self.acti_type="sigmoid"     #various possible types are sigmoid,relu,softmax,leaky_relu,linear etc
        self.f_dims=[]
        self.num_of_filters=0
        self.stride=1
        self.padding="valid"     #various possible types are same,valid or any integer
        self.bias_init=zeros()
        self.weight_init=glorot_normal()

        self.layer_name="conv"
        self.input_shape=None
        self.output_shape=None
        self.Z=None
        self.A=None
        self.F=None
        self.B=None
        self.dZ=None
        self.dA=None
        self.dF=None
        self.dB=None
        self.vec_prev_A=None     #vectorized version of prev_A

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        if(self.padding=="same") : self.padding=(self.stride*(self.input_shape[2]-1)+self.f_dims[0]-self.input_shape[2])//2
        elif(self.padding=="valid") : self.padding=0
        self.output_shape=[self.input_shape[0],self.num_of_filters,(self.input_shape[2]+2*self.padding-self.f_dims[0])//self.stride+1,(self.input_shape[3]+2*self.padding-self.f_dims[1])//self.stride+1]  

        #parameter initialization
        in_neurons=np.prod(np.array(self.input_shape[1:]))
        out_neurons=np.prod(np.array(self.output_shape[1:]))
        self.B=self.bias_init.init([self.num_of_filters],in_neurons,out_neurons)
        self.F=self.weight_init.init([self.num_of_filters,self.input_shape[1]*self.f_dims[0]*self.f_dims[1]],in_neurons,out_neurons)
        return self.output_shape

    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        new_dB,new_dF=optimizer.calculate_parameters(self.dB,self.dF,lyr_no) if optimizer is not None else (self.dB,self.dF)
        self.F+= -lr*regularization.add_to_weights(self.F,self.A.shape[0]) if regularization is not None else 0

        self.B+= lr*-new_dB
        self.F+= lr*-new_dF

    def forward_pass(self,prev_A):
        self.vec_prev_A=arr_to_vec(prev_A,self.f_dims,self.stride,self.padding)
        self.Z=(self.vec_prev_A @ self.F.T).transpose(0,2,1) + self.B.reshape(self.num_of_filters,1)
        self.A=acti(self.acti_type,self.Z)
        return self.A.reshape(self.A.shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3])

    def backward_pass(self,dA):
        self.dA=dA.reshape(dA.shape[0],dA.shape[1],-1)
        self.dZ=self.dA * acti_derivative(self.acti_type,self.Z)
        self.dF=(self.dZ  @ self.vec_prev_A).mean(axis=0)
        self.dB=self.dZ.sum(axis=2).mean(axis=0)
        prev_dA=self.dZ.transpose(0,2,1)  @  self.F.reshape(self.num_of_filters,-1)
        return vec_to_arr(prev_dA,self.f_dims,self.input_shape[2:4],self.stride,self.padding)


class Pool:

    def __init__(self):
        self.f_dims=[]
        self.stride=1
        self.padding="valid"     #various possible types are same,valid or any integer
        self.pool_type="max"     #various possible types are None,max,avg
        
        self.layer_name="pool"
        self.input_shape=None
        self.output_shape=None
        self.A=None
        self.dA=None
        self.req_idx=None     #the required max indices


    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        return

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        if(self.padding=="same") : self.padding=(self.stride*(self.input_shape[2]-1)+self.f_dims[0]-self.input_shape[2])//2
        elif(self.padding=="valid") : self.padding=0
        self.output_shape=[self.input_shape[0],self.input_shape[1],(self.input_shape[2]+2*self.padding-self.f_dims[0])//self.stride+1,(self.input_shape[3]+2*self.padding-self.f_dims[1])//self.stride+1]
        return self.output_shape

    def forward_pass(self,prev_A):
        temp=arr_to_vec(prev_A,self.f_dims,self.stride,self.padding).reshape(prev_A.shape[0],self.output_shape[2]*self.output_shape[3],self.input_shape[1],self.f_dims[0]*self.f_dims[1])

        if(self.pool_type=="max"):
            self.req_idx=np.argmax(temp,axis=3)[:,:,:,None]
            self.A=np.take_along_axis(temp,self.req_idx,axis=3)[:,:,:,0].transpose(0,2,1)
        elif(self.pool_type=="avg"):
            pass
        elif(self.pool_type==None):
            self.A=prev_A
            return self.A
        else:
            pass
        return self.A.reshape(prev_A.shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3])

    def backward_pass(self,dA):
        self.dA=dA.reshape(dA.shape[0],dA.shape[1],-1)
        if(self.pool_type=="max"):
            prev_dA=np.zeros([self.dA.shape[0],self.output_shape[2]*self.output_shape[3],self.input_shape[1],self.f_dims[0]*self.f_dims[1]],dtype=self.A.dtype)
            np.put_along_axis(prev_dA,self.req_idx,self.dA.transpose(0,2,1)[:,:,:,None],axis=3)
            prev_dA=vec_to_arr(prev_dA.reshape(self.dA.shape[0],self.output_shape[2]*self.output_shape[3],-1),self.f_dims,self.input_shape[2:4],self.stride,self.padding)
        elif(self.pool_type=="avg"):
            pass
        elif(self.pool_type==None):
            prev_dA=self.dA
        else:
            pass
        return prev_dA


class FC:

    def __init__(self):
        self.acti_type="sigmoid"     #various possible types are sigmoid,relu,softmax,leaky_relu,linear etc
        self.num_of_neurons=0
        self.bias_init=zeros()
        self.weight_init=glorot_normal()

        self.layer_name="fc"
        self.input_shape=None
        self.output_shape=None
        self.Z=None
        self.A=None
        self.B=None
        self.W=None
        self.dZ=None
        self.dA=None
        self.dW=None
        self.dB=None
        self.prev_A=None     #here vec_prev_A is not present as it is not needed instead prev_A is present

    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        new_dB,new_dW=optimizer.calculate_parameters(self.dB,self.dW,lyr_no) if optimizer is not None else (self.dB,self.dW)
        self.W+= -lr*regularization.add_to_weights(self.W,self.A.shape[1]) if regularization is not None else 0

        self.B+= lr*-new_dB
        self.W+= lr*-new_dW

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        self.output_shape=[self.num_of_neurons,None]

        #parameter initialization
        in_neurons=self.input_shape[0]
        out_neurons=self.output_shape[0]
        self.B=self.bias_init.init([1,out_neurons],in_neurons,out_neurons)
        self.W=self.weight_init.init([out_neurons,in_neurons],in_neurons,out_neurons)
        return self.output_shape

    def forward_pass(self,prev_A):
        self.prev_A=prev_A
        self.Z=(self.W @ self.prev_A)+self.B.T
        self.A=acti(self.acti_type,self.Z)
        return self.A

    def backward_pass(self,dA):
        self.dA=dA
        if(self.acti_type=="softmax") : self.dZ=self.dA
        else : self.dZ=self.dA * acti_derivative(self.acti_type,self.Z)
        self.dW=(self.dZ @ self.prev_A.T) * (1/self.dA.shape[1])
        self.dB=self.dZ.sum(axis=1) * (1/self.dA.shape[1])
        prev_dA=self.W.T @ self.dZ
        return prev_dA


class Drp:
    def __init__(self):  
        self.keep_probs=1.0
        self.seed=None

        self.layer_name="drp"
        self.status="training"     #various possible types are training,testing and grad_check
        self.input_shape=None
        self.output_shape=None
        self.A=None
        self.dA=None
        self.temp_var=0.0     #temporary variable to backup the value of keep_probs when it is set to 1 during testing and grad_check
        self.epsilon=1.0e-6     #a small value added to denominators during divisions to prevent division by zero error    

    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        return

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        self.output_shape=input_shape
        self.temp_var=self.keep_probs
        return self.output_shape

    def forward_pass(self,prev_A):
        self.keep_probs= 1.0 if self.status in ("testing","grad_check") else self.temp_var
        if(self.seed is not None) : np.random.seed(self.seed)
        #prev_A_shape=prev_A.shape
        #prev_A=prev_A.reshape(prev_A.shape[0],-1)
        self.A=(prev_A.reshape(-1,1) * (np.random.rand(np.prod(prev_A.shape),1)<self.keep_probs)/(self.keep_probs+self.epsilon)).reshape(prev_A.shape).astype(prev_A.dtype)
        return self.A
    
    def backward_pass(self,dA):
        if(self.seed is not None) : np.random.seed(self.seed)
        #dA=dA.reshape(self.A.shape)
        self.dA=dA
        prev_dA=(self.dA.reshape(-1,1) * (np.random.rand(np.prod(self.dA.shape),1)<self.keep_probs)/(self.keep_probs+self.epsilon)).reshape(self.dA.shape).astype(dA.dtype)
        return prev_dA


class Flatten:
    def __init__(self):
        self.layer_name="flatten"
        self.input_shape=None
        self.output_shape=None

    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        return

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        self.output_shape=[np.prod(np.array(self.input_shape[1:])),None]
        return self.output_shape

    def forward_pass(self,prev_A):
        return prev_A.reshape(prev_A.shape[0],-1).T
    
    def backward_pass(self,dA):
        return dA.T.reshape(dA.shape[1],self.input_shape[1],self.input_shape[2],self.input_shape[3])


class Batch_Norm:
    def __init__(self):  
        self.layer_name="batch_norm"
        self.input_shape=None
        self.output_shape=None
        self.prev_A=None
        self.bias_init=zeros()
        self.weight_init=glorot_normal()
        self.status="training"     #various possible types are training,testing and grad_check

        self.A=None
        self.dA=None
        self.Y=None
        self.B=None
        self.dY=None
        self.dB=None
        self.prev_A=None
        self.training_mean=0.0     #mean of the entire training set to be used during testing
        self.training_variance=0.0     #variance of the entire training set to be used during testing

    def set_mean_var(self,input_data):     #set training data mean and variance to be used during testing
        if(len(self.input_shape)==2):
            _,self.training_mean,self.training_variance=standardize(input_data)
        elif(len(self.input_shape)==4):
            _,self.training_mean,self.training_variance=standardize( input_data.transpose(1,0,2,3).reshape(self.input_shape[1],-1).transpose() )

    def init_parameters(self,input_shape):
        #input output shape calculation
        self.input_shape=input_shape
        self.output_shape=input_shape

        #parameter initialization
        if(len(self.input_shape)==4):     #meaning batch norm layer is being used after a conv or pool layer
            out_neurons=np.prod(np.array(self.output_shape[1:]))
            in_neurons=np.prod(np.array(self.input_shape[1:]))
            self.B=self.bias_init.init([self.input_shape[1]],in_neurons,out_neurons)
            self.Y=self.weight_init.init([self.input_shape[1]],in_neurons,out_neurons)
        elif(len(self.input_shape)==2):     #meaning batch norm layer is being used after a fc or flatten layer
            out_neurons=self.output_shape[0]
            in_neurons=self.input_shape[0]
            self.B=self.bias_init.init([self.input_shape[0]],in_neurons,out_neurons)
            self.Y=self.weight_init.init([self.input_shape[0]],in_neurons,out_neurons)

        return self.output_shape

    def update_parameters(self,lr,optimizer=None,regularization=None,lyr_no=0):
        new_dB,new_dY=optimizer.calculate_parameters(self.dB,self.dY,lyr_no) if optimizer is not None else (self.dB,self.dY)
        self.Y+= -lr*regularization.add_to_weights(self.Y , self.A.shape[0] if len(self.A.shape)==4 else self.A.shape[1] ) if regularization is not None else 0     #regularization not to be used with batch norm

        self.B+= lr*-new_dB
        self.Y+= lr*-new_dY

    def forward_pass(self,prev_A):
        if(len(self.input_shape)==4):     #meaning batch norm layer is being used after a conv or pool layer
            num_of_examples=prev_A.shape[0]
            prev_A=prev_A.transpose(1,0,2,3).reshape(self.input_shape[1],-1).T
            if(self.status=="training") : prev_A,_,__=standardize(prev_A)
            elif(self.status=="testing") : prev_A,_,__=standardize(prev_A,self.training_mean,self.training_variance)
            elif(self.status=="grad_check") : prev_A=prev_A
            self.prev_A=np.copy( prev_A.T.reshape(self.input_shape[1],num_of_examples,self.input_shape[2],self.input_shape[3]).transpose(1,0,2,3) )
            self.A=(self.Y*prev_A+self.B).T.reshape(self.input_shape[1],self.prev_A.shape[0],self.input_shape[2],self.input_shape[3]).transpose(1,0,2,3)
            return self.A

        elif(len(self.input_shape)==2):     #meaning batch norm layer is being used after a fc or flatten layer
            if(self.status=="training") : prev_A,_,__=standardize(prev_A)
            elif(self.status=="testing") : prev_A,_,__=standardize(prev_A,self.training_mean,self.training_variance)
            elif(self.status=="grad_check") : prev_A=prev_A
            self.prev_A=np.copy(prev_A)
            self.A=self.Y*prev_A+self.B
            return self.A
    
    def backward_pass(self,dA):
        if(len(self.input_shape)==4):     #meaning batch norm layer is being used after a conv or pool layer
            self.dB=dA.sum(axis=3).sum(axis=2).mean(axis=0)
            self.dY=(dA*self.prev_A).sum(axis=3).sum(axis=2).mean(axis=0)
            prev_dA=(dA.transpose(1,0,2,3).reshape(self.input_shape[1],-1).T * self.Y)
            return prev_dA.T.reshape(self.input_shape[1],self.A.shape[0],self.input_shape[2],self.input_shape[3]).transpose(1,0,2,3)

        elif(len(self.input_shape)==2):     #meaning batch norm layer is being used after a fc or flatten layer
            self.dB=dA.mean(axis=1)
            self.dY=(dA*self.prev_A).mean(axis=1)
            prev_dA=(dA*self.Y)
            return prev_dA