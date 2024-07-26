import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import cv2
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import os
import zipfile
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
def load_img(img_dir, img_list,t1,t2):
    images=[]
  
    for i,image_name in enumerate(img_list):    
            image = np.load(img_dir+image_name + '/image_'+image_name + t1 +'.npy')
            image2 = np.load(img_dir+image_name + '/image_'+image_name +t2 +'.npy')
            im= np.stack([image, image2], axis=3)
           #  128*128*128*2
         
                      
            images.append(im)
       
           
    images = np.array(images)
   
    
    return images

def load_mask(img_dir, img_list):
    
    maskes=[]
    for i,image_name in enumerate(img_list):    
           
            mask = np.load(img_dir+image_name + '/mask_' +image_name+'.npy')
           
            maskes.append(mask)
         

    maskes = np.array(maskes)
    
    return maskes

 
            
            

train_img_dir = 'output/train/images/'
val_img_dir = 'output/val/images/'


train_img_list=os.listdir(train_img_dir)
val_img_list=os.listdir(val_img_dir)
# type_image=0 >t1 , type_image=1 >t1ce, type_image=2 >flair, type_image=3 >flair

train_img_datagen_X = load_img(train_img_dir, train_img_list,'t1ce','flair')
train_img_datagen_Y = load_mask(train_img_dir, train_img_list)
val_img_datagen_X = load_img(val_img_dir, val_img_list,'t1ce','flair')
val_img_datagen_Y = load_mask(val_img_dir, val_img_list)


train_img_datagen_X2 = load_img(train_img_dir, train_img_list,'t2','t1ce')

val_img_datagen_X2 = load_img(val_img_dir, val_img_list,'t2','t1ce')



print("xxxxxxxxxxxxxxxxxxxx", train_img_datagen_X.shape)
print("yyyyyyyyyyyyyyyyyyyy", train_img_datagen_Y.shape)

import random
#img, msk = train_img_datagen.__next__()
 
img_num =1
test_img=train_img_datagen_X[img_num]
test_img2=train_img_datagen_X2[img_num]
test_mask=train_img_datagen_Y[img_num]

test_mask=np.argmax(test_mask, axis=3)
n_slice=70
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0 ], cmap='gray')
plt.title('Image t1ce')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1 ], cmap='gray')
plt.title('Image flair')
plt.subplot(223)
plt.imshow(test_img2[:,:,n_slice, 0 ], cmap='gray')
plt.title('T2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')

plt.savefig('Combine_two_model/1.png', format='png')

###########################################################################

# dice loss as defined above for 4 classes
import keras.backend as K



#33333333333333333333333333333333333333333333
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
   
    return loss


#333333333333333333333333333333333333333333333333
 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
#def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
#    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
#    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

#def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
#    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
#    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

#def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
#    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
#    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' 
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH,c, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH,c))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2,2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
 
    return model

#Test if everything is working ok. 
model1 = simple_unet_model(128, 128, 128,2, 4)
model2 = simple_unet_model(128, 128, 128,2, 4)


##callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
  ##    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
 ##                             patience=2, min_lr=0.000001, verbose=1),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
 ##       csv_logger
 ##   ]


import tensorflow as tf
model1.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001), 
              metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), 
                         dice_coef, precision, sensitivity, specificity] )
model2.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001), 
              metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), 
                         dice_coef, precision, sensitivity, specificity] )                         





model1.load_weights('3D_segmentation_t1ce_flair_best_weight.hdf5')

model2.load_weights('3D_segmentation_T2_t1ce_best_weight.hdf5')


model1.summary()

for layer in model1.layers:
            layer.trainable = False
            
for layer in model2.layers:
            layer.trainable = False            

model1.summary()


m= layers.Concatenate()([model1.layers[-2].output, model2.layers[-2].output])
m= Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(m)

x = Conv3D(4, (1, 1, 1), activation='softmax') (m)#(model1.output)



print("***********************************")

model3 =  Model( inputs=[model1.input, model2.input], outputs=x ) 
model3.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001), 
              metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), 
                         dice_coef, precision, sensitivity, specificity] )  
model3.summary()





from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.inputs.keras import PlotLossesCallback

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='Combine_two_model_best_weight.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping - monitors the performance of the model and stopping the training process prevents overtraining
early_stop = EarlyStopping(monitor='val_loss',
                         patience=5,
                          restore_best_weights=True,
                           mode='min')

batch_size=4

steps_per_epoch =len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size




K.clear_session()


history=model3.fit(x=[train_img_datagen_X,train_img_datagen_X2], y=train_img_datagen_Y, steps_per_epoch=steps_per_epoch, epochs=40,verbose=1,
                  callbacks=[tl_checkpoint_1, early_stop],
                  batch_size=batch_size,
                  validation_data=([val_img_datagen_X,val_img_datagen_X2],val_img_datagen_Y),
                  validation_steps=val_steps_per_epoch,
                 )

 
 
######################################################################################################3
########################################################################################################
###################################################################################################################

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure(figsize=(12, 8))
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.savefig('Combine_two_model/2.png', format='png')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(12, 8))
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
 
plt.savefig('Combine_two_model/3.png', format='png')




acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
plt.figure(figsize=(12, 8))
plt.plot(epochs, acc, 'y', label='Training dice_coef')
plt.plot(epochs, val_acc, 'r', label='Validation dice_coef')
plt.title('Training and validation dice_coef')
plt.xlabel('Epochs')
plt.ylabel('dice_coef')
plt.legend()
 
plt.savefig('Combine_two_model/33.png', format='png')


acc = history.history['mean_io_u_2']
val_acc = history.history['val_mean_io_u_2']
plt.figure(figsize=(12, 8))
plt.plot(epochs, acc, 'y', label='Training mean IOU')
plt.plot(epochs, val_acc, 'r', label='Validation mean IOU')
plt.title('Training and validation mean IOU')
plt.xlabel('Epochs')
plt.ylabel('mean IOU')
plt.legend()
 
plt.savefig('Combine_two_model/333.png', format='png')

model3.load_weights('Combine_two_model_best_weight.hdf5')


#model.load_weights('our_segmentation_BraTS2020/3D_segmentation_T2_best_weight.hdf5')
#################################################
n_classes = 4

test_img_dir = 'output/test/images/'



test_img_list=os.listdir(test_img_dir)




test_img_datagen_X = load_img(test_img_dir, test_img_list,'t1ce','flair')
test_img_datagen_X2 = load_img(test_img_dir, test_img_list,'t2','t1ce')
test_img_datagen_Y = load_mask(test_img_dir, test_img_list)





Averae_test_result= np.zeros((7))  
Mean_IoU=0  
results = model3.evaluate(x=[test_img_datagen_X,test_img_datagen_X2],y=test_img_datagen_Y, batch_size=batch_size,)

Averae_test_result[0]= results[0]
Averae_test_result[1]= results[1]
Averae_test_result[2]= results[2]
Averae_test_result[3]= results[3] 
Averae_test_result[4]= results[4]
Averae_test_result[5]= results[5] 
Averae_test_result[6]= results[6] 
 
 
print("***************************************************")    
 

print("loss =", Averae_test_result[0])
print("accuracy =", Averae_test_result[1])
print("mean_io_u =", Averae_test_result[2])
print("dice_coef =", Averae_test_result[3])
print("precision =", Averae_test_result[4])
print("sensitivity =", Averae_test_result[5])
print("specificity =", Averae_test_result[6])
 

print("***************************************************")
    










img_num =2

test_img1 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t1ce.npy")

test_img2 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "flair.npy")

test_img3 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t2.npy")

test_img= np.stack([test_img1, test_img2], axis=3)
test_img_2= np.stack([test_img3, test_img1], axis=3)
            
print("test_img:", test_img.shape)

test_mask = np.load("output/test/images/"+str(img_num)+ "/mask_"+str(img_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_img_input2 = np.expand_dims(test_img_2, axis=0)
test_prediction = model3.predict([test_img_input,test_img_input2])
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


# print(test_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(test_prediction_argmax))


#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image t1ce')
plt.imshow(test_img[:,:,n_slice,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Image flair')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(233)
plt.title('Mask')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(234)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
 
plt.savefig('Combine_two_model/4.png', format='png')




img_num = 11

test_img1 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t1ce.npy")

test_img2 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "flair.npy")

test_img3 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t2.npy")

test_img= np.stack([test_img1, test_img2], axis=3)
test_img_2= np.stack([test_img3, test_img1], axis=3)
            
print("test_img:", test_img.shape)

test_mask = np.load("output/test/images/"+str(img_num)+ "/mask_"+str(img_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_img_input2 = np.expand_dims(test_img_2, axis=0)
test_prediction = model3.predict([test_img_input,test_img_input2])
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


# print(test_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(test_prediction_argmax))


#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image t1ce')
plt.imshow(test_img[:,:,n_slice,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Image flair')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(233)
plt.title('Mask')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(234)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
 
plt.savefig('Combine_two_model/5.png', format='png')


img_num = 90

test_img1 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t1ce.npy")

test_img2 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "flair.npy")

test_img3 = np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t2.npy")

test_img= np.stack([test_img1, test_img2], axis=3)
test_img_2= np.stack([test_img3, test_img1], axis=3)
            
print("test_img:", test_img.shape)

test_mask = np.load("output/test/images/"+str(img_num)+ "/mask_"+str(img_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_img_input2 = np.expand_dims(test_img_2, axis=0)
test_prediction = model3.predict([test_img_input,test_img_input2])
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


# print(test_prediction_argmax.shape)
# print(test_mask_argmax.shape)
# print(np.unique(test_prediction_argmax))


#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image t1ce')
plt.imshow(test_img[:,:,n_slice,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Image flair')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(233)
plt.title('Mask')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(234)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
 
plt.savefig('Combine_two_model/6.png', format='png')



def showPredictsById( img_num ):

 #   X = imageLoaderX(test_img_dir, test_img_list,case, img_num,1)
 #   Y = imageLoaderY(test_img_dir, test_img_list,case, img_num,1)
 
 
    X1=np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t1ce.npy")
    X2=np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "flair.npy")
    X3=np.load("output/test/images/" +str(img_num)+ "/image_"+str(img_num)+ "t2.npy")
    X= np.stack([X1, X2], axis=3)
    X_2=np.stack([X3, X1], axis=3)
    
    Y= np.load("output/test/images/"+str(img_num)+ "/mask_"+str(img_num)+".npy")
    print('x=', X.shape)
    print('Y=', Y.shape)
    test_img_input = np.expand_dims(X, axis=0)
    test_img_input2 = np.expand_dims(X_2, axis=0)
    print('xis=', X.shape)
    test_mask_batch_argmax = Y #.reshape(128,128,128,4) 
    
 
    test_mask_argmax=np.argmax(test_mask_batch_argmax, axis=3)

   # test_img_input = np.expand_dims(X, axis=0)
    test_prediction = model3.predict([test_img_input,test_img_input2])
#    test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
 
   # p = test_pred_batch_argmax
    p=test_prediction.reshape(128,128,128,4) 
    print('p=', p.shape)
    pp=np.argmax(p, axis=3)
    core = p[:,:,:,1]
    edema= p[:,:,:,2]
    enhancing = p[:,:,:,3]

    print('test_mask_argmax=',test_mask_argmax.shape)
    plt.figure(figsize=(18, 10))
    f, axarr = plt.subplots(1,8, figsize = (18, 5)) 
   
    
    print('x2=', X.shape)
    axarr[0].imshow( X[:,:,n_slice,0], cmap="gray", interpolation='none')
    axarr[1].imshow( X[:,:,n_slice,1], cmap="gray", interpolation='none')
    axarr[2].imshow( X_2[:,:,n_slice,0], cmap="gray", interpolation='none')
    for i in range(3,8): # for each image, add brain background
        axarr[i].imshow( X[:,:,n_slice,0], cmap="gray", interpolation='none')
    
    axarr[0].imshow(X[:,:,n_slice,0], cmap="gray")
    axarr[0].title.set_text( ' MRI image t1ce')
    axarr[1].imshow(X[:,:,n_slice,1], cmap="gray")
    axarr[1].title.set_text(  ' MRI image flair')
    axarr[2].imshow(X_2[:,:,n_slice,0], cmap="gray")
    axarr[2].title.set_text(  ' MRI image t2')
    curr_gt=test_mask_argmax[:,:,n_slice]
    axarr[3].imshow(curr_gt, cmap="Wistia", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[3].title.set_text('Mask')
    axarr[4].imshow(pp[:,:,n_slice], cmap="Greens", interpolation='none', alpha=0.3)
    axarr[4].title.set_text('predicted')
    axarr[5].imshow(edema[:,:,n_slice], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'Core predicted')
    axarr[6].imshow(core[:,:,n_slice], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[6].title.set_text(f' Edema predicted')
    axarr[7].imshow(enhancing[:,:, n_slice], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[7].title.set_text(f'Enhancing predicted')
    
    plt.savefig('Combine_two_model/7' + str(img_num) + '.png', format='png')
    
showPredictsById(2)
showPredictsById(11)
showPredictsById(90)
 


