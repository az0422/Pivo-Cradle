import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

YOLO_TYPE = 0

def YOLOv4_custom(input_layer, NUM_CLASS):
    if (YOLO_TYPE == 0):
        return YOLOv4_tiny_original(input_layer, NUM_CLASS)
    elif YOLO_TYPE == 1:
        return YOLOv4_tiny_custom(input_layer, NUM_CLASS)

def YOLOv4_tiny_original(input_layer, NUM_CLASS):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def custom_backbone(input_layer):
    input_layer = common.convolutional(input_layer, (3, 32), downsample=True)
    
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)
    input_layer = common.convolutional(input_layer, (3, 64))
    
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)
    input_layer = common.convolutional(input_layer, (3, 128))
    input_layer = common.route_group(input_layer, 2, 1)
    input_layer = common.convolutional(input_layer, (3, 128))
    
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)
    input_layer = common.convolutional(input_layer, (3, 256))
    input_layer = common.route_group(input_layer, 2, 1)
    input_layer = common.convolutional(input_layer, (3, 128))
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 128))
    input_layer = tf.concat([input_layer, route_1,], axis=-1)
    input_layer = common.convolutional(input_layer, (3, 256))
    route_r = input_layer
    
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)
    
    return [route_r, input_layer]

def YOLOv4_tiny_custom(input_layer, NUM_CLASS):
    route_1, conv = custom_backbone(input_layer)

    conv = common.convolutional(conv, (1, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 128))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    print(conv_mbbox, conv_lbbox)
    print(conv.shape[-1])
    return [conv_mbbox, conv_lbbox]
    
    
'''
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)	#11 13
    conv = common.convolutional(input_layer, (3, 3, 128, 128))		#12
    route_1 = conv

    conv_lobj_branch = common.convolutional(conv, (3, 3, 128, 256))		#13
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)	#14

    conv = common.convolutional(route_1, (3, 3, 128, 128))			#16
    conv = common.upsample(conv)					#17 26

    conv = tf.concat([conv, route], axis=-1)				#18
    conv = common.convolutional(conv, (3, 3, 256, 256))			#19
    
    conv_mbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False) #20

    return [conv_mbbox, conv_lbbox]
    '''

'''
def YOLOv4_tiny_custom(input_layer, NUM_CLASS):
    input_layer = common.convolutional(input_layer, (3, 3, 3, 24), downsample=True)	#0
    input_layer = common.convolutional(input_layer, (3, 3, 24, 48), downsample=True) 	#1
    input_layer = common.convolutional(input_layer, (3, 3, 48, 48))	#2

    route = input_layer
    input_layer = common.route_group(input_layer, 2, 1)		#3
    input_layer = common.convolutional(input_layer, (3, 3, 24, 24))	#4
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 24, 24))	#5
    input_layer = tf.concat([input_layer, route_1], axis=-1)		#6
    input_layer = common.convolutional(input_layer, (1, 1, 48, 48))	#7
    input_layer = tf.concat([route, input_layer], axis=-1)		#8
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)	#9

    input_layer = common.convolutional(input_layer, (3, 3, 96, 96))	#10
    route = input_layer
    input_layer = common.route_group(input_layer, 2, 1)		#11
    input_layer = common.convolutional(input_layer, (3, 3, 48, 48))	#12
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 48, 48))	#13
    input_layer = tf.concat([input_layer, route_1], axis=-1)		#14
    input_layer = common.convolutional(input_layer, (1, 1, 96, 96))	#15
    input_layer = tf.concat([route, input_layer], axis=-1)		#16
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)	#17

    input_layer = common.convolutional(input_layer, (3, 3, 192, 192))	#18
    route = input_layer
    input_layer = common.route_group(input_layer, 2, 1)		#19
    input_layer = common.convolutional(input_layer, (3, 3, 96, 96))	#20
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 96, 96))	#21
    input_layer = tf.concat([input_layer, route_1], axis=-1)		#22
    route_2 = input_layer
    input_layer = common.convolutional(input_layer, (1, 1, 192, 192))	#23
    route_1 = input_layer
    input_layer = tf.concat([route, input_layer], axis=-1)		#24
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)	#25

    conv = common.convolutional(input_layer, (3, 3, 384, 384))	#26

##########################################################
    conv = common.convolutional(conv, (1, 1, 384, 192))		#27

    conv_lobj_branch = common.convolutional(conv, (3, 3, 192, 384))	#28
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 384, 3 * (NUM_CLASS + 5)), activate=False, bn=False)	#29

    conv = common.convolutional(conv, (1, 1, 192, 192))	#32
    conv = common.upsample(conv)			#33
    conv = tf.concat([conv, route_1], axis=-1)		#34

    conv_mobj_branch = common.convolutional(conv, (3, 3, 384, 96))	#35
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 96, 3 * (NUM_CLASS + 5)), activate=False, bn=False) #36

    return [conv_mbbox, conv_lbbox]
'''
'''
def YOLOv4_tiny_custom(input_layer, NUM_CLASS):
    input_layer = common.convolutional(input_layer, (3, 3, 3, 32), downsample=True)
    input_layer = common.convolutional(input_layer, (3, 3, 32, 64), downsample=True)
    input_layer = common.convolutional(input_layer, (3, 3, 64, 64))

    route = input_layer
    input_layer = common.route_group(input_layer, 4, 1)
    input_layer = common.convolutional(input_layer, (3, 3, 16, 16))
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 16, 16))
    input_layer = tf.concat([input_layer, route_1], axis=-1)
    input_layer = common.convolutional(input_layer, (1, 1, 32, 32))
    input_layer = tf.concat([route, input_layer], axis=-1)
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)

    input_layer = common.convolutional(input_layer, (3, 3, 96, 96))
    route = input_layer
    input_layer = common.route_group(input_layer, 2, 1)
    input_layer = common.convolutional(input_layer, (3, 3, 48, 48))
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 48, 48))
    input_layer = tf.concat([input_layer, route_1], axis=-1)
    input_layer = common.convolutional(input_layer, (1, 1, 96, 96))
    input_layer = tf.concat([route, input_layer], axis=-1)
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)

    input_layer = common.convolutional(input_layer, (3, 3, 192, 192))
    route = input_layer
    input_layer = common.route_group(input_layer, 3, 1)
    input_layer = common.convolutional(input_layer, (3, 3, 64, 64))
    route_1 = input_layer
    input_layer = common.convolutional(input_layer, (3, 3, 64, 64))
    input_layer = tf.concat([input_layer, route_1], axis=-1)
    route_2 = input_layer
    input_layer = common.convolutional(input_layer, (1, 1, 128, 128))
    route_1 = input_layer
    input_layer = tf.concat([route, input_layer], axis=-1)
    input_layer = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_layer)

    conv = common.convolutional(input_layer, (3, 3, 256, 256))

##########################################################
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 128, 96))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 224, 224))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 224, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]
'''