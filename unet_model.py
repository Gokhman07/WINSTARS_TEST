
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.layers import Input

def unet_model(input_shape):
  inputs=Input(input_shape)

  #ENCODER
  #layer1
  conv1=Conv2D(64, 3,activation='relu',padding='same')(inputs)
  conv1=Conv2D(64, 3,activation='relu',padding='same')(conv1)
  pool1=MaxPooling2D(pool_size=(2,2))(conv1)


  #layer2
  conv2=Conv2D(128, 3,activation='relu',padding='same')(pool1)
  conv2=Conv2D(128, 3,activation='relu',padding='same')(conv2)
  pool2=MaxPooling2D(pool_size=(2,2))(conv2)

  #layer3
  conv3=Conv2D(256, 3,activation='relu',padding='same')(pool2)
  conv3=Conv2D(256, 3,activation='relu',padding='same')(conv3)
  pool3=MaxPooling2D(pool_size=(2,2))(conv3)

  #layer4
  conv4=Conv2D(512, 3,activation='relu',padding='same')(pool3)
  conv4=Conv2D(512, 3,activation='relu',padding='same')(conv4)
  drop1=Dropout(0.5)(conv4)
  pool4=MaxPooling2D(pool_size=(2,2))(conv4)



  #Botttom
  conv5=Conv2D(1024, 3,activation='relu',padding='same')(pool4)
  conv5=Conv2D(1024, 3,activation='relu',padding='same')(conv5)
  drop2=Dropout(0.5)(conv5)

  #Decoder
  #layer6
  up6=Conv2DTranspose(512,2,strides=(2,2),padding='same')(drop2)
  merge6=concatenate([conv4,up6],axis=3)
  conv6=Conv2D(512,3,activation='relu',padding='same')(merge6)
  conv6=Conv2D(512,3,activation='relu',padding='same')(conv6)

  #layer7
  up7=Conv2DTranspose(256,2,strides=(2,2),padding='same')(conv6)
  merge7=concatenate([conv3,up7],axis=3)
  conv7=Conv2D(256,3,activation='relu',padding='same')(merge7)
  conv7=Conv2D(256,3,activation='relu',padding='same')(conv7)

  #layer8
  up8=Conv2DTranspose(128,2,strides=(2,2),padding='same')(conv7)
  merge8=concatenate([conv2,up8],axis=3)
  conv8=Conv2D(128,3,activation='relu',padding='same')(merge8)
  conv8=Conv2D(128,3,activation='relu',padding='same')(conv8)

  #layer9
  up9=Conv2DTranspose(64,2,strides=(2,2),padding='same')(conv8)
  merge9=concatenate([conv1,up9],axis=3)
  conv9=Conv2D(64,3,activation='relu',padding='same')(merge9)
  conv9=Conv2D(64,3,activation='relu',padding='same')(conv9)

  outputs=Conv2D(1,1,activation='sigmoid')(conv9)

  model=Model(inputs,outputs)

  return model
