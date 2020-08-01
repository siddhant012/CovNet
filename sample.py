#A sample file for making a model to train cifar10 dataset to help you understand how to use this library.
#Not everything is specified here.You can lookup the actual code to know more about it. 
#Keep the CovNet.py file in the same folder as this sample file so that it can be imported here.

import CovNet as CV
from keras.datasets import cifar10     #just to load the dataset
import numpy as np
import math

def main():

    #creating object of the CovNet class
    cv=CV.CovNet()
    
    #specifying required values
    cv.num_of_epochs=20
    cv.batch_size=128
    cv.learning_rate=0.01
    cv.optimizer=CV.adam()                                #several types can be adam(),rms_prop() and momentum() with required arguments.It can be set to None as well
    cv.lr_decay=None#CV.linear_decay(1,2)                 #several types can be linear_decay(),expo_decay(),sqr_root_decay,staircase_decay() and linear_decay() with required arguments.It can be set to None as well.
    cv.regularization=None#CV.l2(0.9)                     #several types can be l2() and l1() with required arguments. Dropout is implemented as a separate layer.It can be set to None as well.
    cv.cost_function_type="categorical_cross_entropy"     #several types can be "binary_cross_entropy" , "categorical_cross_entropy" , "mean_square_error" and  "mean_absolute_error"
    cv.display_type="epoch_type"                          #how the results are to be displayed at each epoch. "epoch_type" displays info for each batch on a new line.   "batch_type" displays info for each iteration on a new line and is more keras like.

    #loading the dataset into numpy arrays from keras
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    total_images=trainX.shape[0]
    
    #converting to depth first format.(depth last is not supported)
    trainX=np.concatenate([ trainX[:,:,:,0].reshape(total_images,1,32,32),trainX[:,:,:,1].reshape(total_images,1,32,32),trainX[:,:,:,2].reshape(total_images,1,32,32) ],axis=1)
    
    #shuffling randomly
    input_data=np.array(trainX).reshape(total_images,-1)
    output_data=np.array(trainY).reshape(total_images,1)
    input_data,output_data=CV.shuffle(input_data,output_data)     #shuffles randomly the input and output data keeping the one to one relation maintained.It takes two numpy arrays as input so make sure that the input and output data is 2 dimensional.
    
    #uncomment if you want to train on only some data not all.
    #input_data=input_data[0:40000]
    #output_data=output_data[0:40000]
    #total_images=40000
    
    #reshaping back after shuffling
    input_data=input_data.reshape(total_images,trainX.shape[1],trainX.shape[2],trainX.shape[3])
    output_data=output_data.reshape(total_images,1)

    #normalizing/standardizing
    input_data=input_data/255.0                                    #alternate form of normalizing the training data in case of images.
    #input_data,_,__=CV.normalize(input_data)                      #denormalize function also exist.If the arguments max , min are specified then data will be normalized by these values otherwise max and min will be found out from data provided.It is helpful when We want to do prediction on new data and need it to be normalized in a manner similar to training data.
    #input_data,_,__=CV.standardize(input_data)                    #destandardize function also exist.If the arguments mean ,variance are specified then data will be standardized by these values otherwise mean and variance will be found out from data provided.It is helpful when We want to do prediction on new data and need it to be standardized in a manner similar to training data.
    output_data=CV.one_hot_encoded(10,output_data,start_num=0)     #convert to onehot.Argument start_num means the lowest value of class label. Sometimes class labels are specified from 0 and sometimes from 1.

    #specifying the input and output training data to the CV object
    cv.input_data=input_data
    cv.output_data=output_data


    l1=CV.Conv()            #object for the Conv class layer which represents a convolutional layer
    l1.f_dims=[3,3]         #filter/kernel shape
    l1.stride=1             #stride to be used
    l1.padding="valid"      #padding can be "valid" or "same"
    l1.num_of_filters=32    #number of filters to be used
    l1.acti_type="relu"     #various possible types are "relu","sigmoid","leaky_relu","linear","softmax","tanh"
    l1.bias_init=CV.zeros() #various possible types are zeros(),ones(),constant(),random_normal(),random_uniform(),glorot_normal(),glorot_uniform(),he_normal() and he_uniform() with required arguments.
    l1.weight_init=CV.glorot_normal()     #various possible types are zeros(),ones(),constant(),random_normal(),random_uniform(),glorot_normal(),glorot_uniform(),he_normal() and he_uniform() with required arguments.
    #32,30,30

    l2=CV.Conv()
    l2.f_dims=[3,3]
    l2.stride=1
    l2.padding="valid"
    l2.num_of_filters=32
    l2.acti_type="relu"
    l2.bias_init=CV.zeros()
    l2.weight_init=CV.glorot_normal()
    #32,28,28

    l3=CV.Pool()            #object for the Pool class layer which represents a pooling layer
    l3.f_dims=[2,2]         #the filter shape used for pooling
    l3.padding="valid"      #the padding used for pooling
    l3.stride=2             #the stride used for pooling
    #32,14,14

    l4=CV.Batch_Norm()     #object for the Batch_Norm class layer which represents a batch_norm layer

    l5=CV.Conv()
    l5.f_dims=[3,3]
    l5.stride=1
    l5.padding="valid"
    l5.num_of_filters=64
    l5.acti_type="relu"
    l5.bias_init=CV.zeros()
    l5.weight_init=CV.glorot_normal()
    #64,12,12

    l6=CV.Conv()
    l6.f_dims=[3,3]
    l6.stride=1
    l6.padding="valid"
    l6.num_of_filters=64
    l6.acti_type="relu"
    l6.bias_init=CV.zeros()
    l6.weight_init=CV.glorot_normal()
    #64,10,10

    l7=CV.Pool()
    l7.f_dims=[2,2]
    l7.padding=0
    l7.stride=2
    l7.pool_type="max"
    #64,5,5

    l8=CV.Flatten()                 #object for the Flatten class layer which represents a flatten layer

    l9=CV.Drp()                     #object for the Dropout class layer which represents a dropout layer.It can be placed after any layer.
    l9.keep_probs=0.4               #probability of keeping a node.Similar to that in keras.

    l10=CV.FC()                     #object for the FC class layer which represents a fully_connected layer
    l10.num_of_neurons=128          #the number of neurons in the fc layer
    l10.acti_type="relu"
    l10.bias_init=CV.zeros()
    l10.weight_init=CV.glorot_uniform()
    
    l11=CV.FC()
    l11.num_of_neurons=10
    l11.acti_type="softmax"
    l11.bias_init=CV.zeros()
    l11.weight_init=CV.glorot_normal()


    #appending layer objects in the cv.layers variable.The sequence of adding matters.
    cv.layers.append(l1)
    cv.layers.append(l2)
    cv.layers.append(l3)
    cv.layers.append(l4)
    cv.layers.append(l5)
    cv.layers.append(l6)
    cv.layers.append(l7)
    cv.layers.append(l8)
    cv.layers.append(l9)
    cv.layers.append(l10)
    cv.layers.append(l11)


    cv.load_network()          #For vuilding the model.similar to build in keras.
    #cv.model_summary()        #print the model summary if required
    cv.train()                 #To start training the model.Similar to fit in keras.

    # Here I have used train data for testing and prediction which is not generally the case (it is done on test data) .However before doing thay make sure that the test data is preprocessed in a similar way as the training data for good results.
    #cv.test(trainX,trainY)    #Test on different data
    #cv.predict(trainX)        #Make a prediction
    
    #Gradient checking function just in case you decide to modify the library for your personal work , you will have a way to know if the code you wrote is correct.But it takes a lot of time to run.Try testing it on smaller models.Its essentially a for loop over all the weights in the model.
    #cv.grad_check()

if(__name__=="__main__") : main()