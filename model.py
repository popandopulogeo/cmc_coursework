from keras import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, LeakyReLU
from keras.losses import binary_crossentropy
from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from keras import backend as K

def build_model(size, start_neurons=8):
    input_layer = Input(size + (1,))
    
    conv1 = Conv2D(start_neurons*1,(3,3), padding="same")(input_layer)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(start_neurons*1,(3,3), padding="same")(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons*2,(3,3), padding="same")(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(start_neurons*2,(3,3), padding="same")(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons*4,(3,3), padding="same")(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(start_neurons*4,(3,3), padding="same")(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons*8,(3,3), padding="same")(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(start_neurons*8,(3,3), padding="same")(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(start_neurons*16,(3,3), padding="same")(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(start_neurons*16,(3,3), padding="same")(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(0.5)(pool5)

    # Middle
    convm = Conv2D(start_neurons*32,(3,3), padding="same")(pool5)
    convm = LeakyReLU(alpha=0.1)(convm)
    convm = Conv2D(start_neurons*32,(3,3), padding="same")(convm)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(0.5)(uconv5)
    uconv5 = Conv2D(start_neurons*16,(3,3), padding="same")(uconv5)
    uconv5 = LeakyReLU(alpha=0.1)(uconv5)
    uconv5 = Conv2D(start_neurons*16,(3,3), padding="same")(uconv5)
    uconv5 = LeakyReLU(alpha=0.1)(uconv5)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8,(3,3), padding="same")(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    uconv4 = Conv2D(start_neurons*8,(3,3), padding="same")(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons*4,(3,3),strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons*4,(3,3), padding="same")(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    uconv3 = Conv2D(start_neurons*4,(3,3), padding="same")(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons*2,(3,3),strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons*2,(3,3), padding="same")(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    uconv2 = Conv2D(start_neurons*2,(3,3), padding="same")(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons*1,(3,3),strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons*1,(3,3), padding="same")(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    uconv1 = Conv2D(start_neurons*1,(3,3), padding="same")(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uncov1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1,(1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = Model(input_layer, output_layer)
    return model

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))/K.log(10.0)

def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def apply_metrcis():
	get_custom_objects().update({'SSIM': SSIM})
	get_custom_objects().update({'PSNR': PSNR})

