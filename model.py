from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, MaxPooling2D, Lambda
import tensorflow as tf

def reorg(input_tensor, stride=2, darknet=True):
    shapes = tf.shape(input_tensor)
    channel_first = tf.transpose(input_tensor,(0,3,1,2))
    reshape_tensor = tf.reshape(channel_first, (-1,shapes[3] // (stride ** 2), shapes[1], stride, shapes[2], stride))
    permute_tensor = tf.transpose(reshape_tensor,(0,3,5,1,2,4))
    target_tensor = tf.reshape(permute_tensor, (-1, shapes[3]*stride**2,shapes[1] // stride, shapes[2] // stride))
    channel_last = tf.transpose(target_tensor,(0,2,3,1))
    result = tf.reshape(channel_last, (-1,shapes[1]//stride, shapes[2]//stride, tf.cast(input_tensor.shape[3]*4, tf.int32)))
    return result
    
class FeatureExtractor:
    def __init__(self):
        self.true_boxes = Input(shape=(1, 1, 1, 15 , 4))
        self.anchors_map  = Input(shape=(None, None, 5, 1))

    def tiny_yolo_convolutional_net(self):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))

        net = Conv2D(16, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(32, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)
        
        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="9_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same", name="11_max")(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        

        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])        
        net = Conv2D((20 + 5) * 5, 1, padding="same", use_bias=True, name="last_conv")(net)
        return Model([inputs, self.true_boxes, self.anchors_map], net)        

    def yolo_convolutional_net(self):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="23_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        conv_id = 25
        c16 = Conv2D(64, 1, padding="same", use_bias=False, name="{}_conv".format(conv_id))(c16)
        conv_id += 1
        c16 = BatchNormalization(epsilon=bn_epsilon)(c16)
        c16 = LeakyReLU(alpha=0.1)(c16)            

        c16 = Lambda(reorg)(c16)        
        net = Concatenate()([c16, net])

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="{}_conv".format(conv_id))(net)
        conv_id += 1
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        

        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        net = Conv2D((20 + 5) * 5, 1, padding="same", use_bias=True, name="last_conv")(net)
        return Model([inputs, self.true_boxes, self.anchors_map], net)        
