import tensorflow as tf

def bbox_iog(y_true,y_pred):
    x11, y11, x12, y12 = tf.split(y_pred, 4, axis=1)
    x21, y21, x22, y22 = tf.split(y_true, 4, axis=1)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    intersect_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    gt_area = (x22 - x21 + 1) * (y22 - y21 + 1)


    return tf.maximum(intersect_area / gt_area, 0.0)

def smooth_ln(x, smooth):
    return tf.where(
        tf.less_equal(x, smooth),
        -tf.log(1 - x),
        ((x - smooth) / (1 - smooth)) - tf.log(1 - smooth)
)

def creat_repulsion(y_true,y_pred,smooth):
    return smooth_ln(bbox_iog(y_true,y_pred),smooth)