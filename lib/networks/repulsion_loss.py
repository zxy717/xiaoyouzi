import tensorflow as tf
import numpy as np


def bbox_overlap_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    #x11 = tf.transpose(x11, (1, 2, 0))
    #y11 = tf.transpose(y11, (1, 2, 0))
    #x12 = tf.transpose(x12, (1, 2, 0))
    #y12 = tf.transpose(y12, (1, 2, 0))

    #x21 = tf.transpose(x21, (1, 2, 0))
    #y21 = tf.transpose(y21, (1, 2, 0))
    #x22 = tf.transpose(x22, (1, 2, 0))
    #y22 = tf.transpose(y22, (1, 2, 0))

    #xI1 = tf.maximum(x11, tf.transpose(x21))
    #yI1 = tf.maximum(y11, tf.transpose(y21))

    #xI2 = tf.minimum(x12, tf.transpose(x22))
    #yI2 = tf.minimum(y12, tf.transpose(y22))

    #x11 = tf.transpose(x11)
    #y11 = tf.transpose(y11)
    #x12 = tf.transpose(x12)
    #y12 = tf.transpose(y12)

    x21 = tf.transpose(x21)
    y21 = tf.transpose(y21)
    x22 = tf.transpose(x22)
    y22 = tf.transpose(y22)

    xI1 = tf.maximum(x11, x21)
    yI1 = tf.maximum(y11, y21)

    xI2 = tf.minimum(x12, x22)
    yI2 = tf.minimum(y12, y22)

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (bboxes1_area + bboxes2_area) - inter_area

    return tf.maximum(inter_area / union, 0.0)

def bbox_iog(ground_truth, predicted):
    x11, y11, x12, y12 = tf.split(predicted, 4, axis=1)
    x21, y21, x22, y22 = tf.split(ground_truth, 4, axis=1)

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



def repulsion_term_gt(y_pred, ious_over_truth_boxes, smooth, weights):
    _, indices_2highest_iou = tf.nn.top_k(ious_over_truth_boxes[..., 0], k=2)
    ious_over_truth_boxes = ious_over_truth_boxes[..., 1:]
    indices_2highest_iou = indices_2highest_iou[..., 1]

    gt_boxes_with_max_ious = None

    for batch_num in np.arange(1, dtype=np.int64):
        #indices = tf.stack([tf.cast(tf.tile([batch_num], [tf.shape(y_pred)[1]]), dtype=tf.int64),
                            #tf.range(tf.cast(tf.shape(y_pred)[1], dtype=tf.int64)),
                            #tf.cast(indices_2highest_iou[batch_num], dtype=tf.int64)])
        indices = tf.stack([tf.range(tf.cast(tf.shape(y_pred)[0],dtype=tf.int64)),
                            tf.cast(indices_2highest_iou, dtype=tf.int64)])
        indices = tf.transpose(indices)

        #if gt_boxes_with_max_ious is None:
            #gt_boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
        #else:
            #boxes_with_max_ious = tf.expand_dims(tf.gather_nd(ious_over_truth_boxes, indices), axis=0)
            #gt_boxes_with_max_ious = tf.concat([gt_boxes_with_max_ious, boxes_with_max_ious], axis=0)
        gt_boxes_with_max_ious = tf.gather_nd(ious_over_truth_boxes,indices)

    ln_distances_for_iog = weights * smooth_ln(bbox_iog(gt_boxes_with_max_ious, y_pred), smooth)
    return tf.reduce_sum(ln_distances_for_iog) / tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)


def repulsion_term_box(ious, smooth):
    iou_over_predicted_indices = tf.where(tf.less(ious, 1.0))
    ious = tf.gather_nd(ious, iou_over_predicted_indices)

    dist_sum = tf.reduce_sum(smooth_ln(ious, smooth))
    iou_sum = tf.reduce_sum(ious)

    return dist_sum / tf.maximum(iou_sum, 0.000001)



def create_repulsion_loss(y_true, y_pred, score, alpha=0.5, betta=0.5,
                          smooth_rep_gt=0.99, smooth_rep_box=0.01,
                          objectness_to_filter=0.5):
    def _filter_predictions(y_pred, score):
        y_pred_indices = tf.where(tf.greater_equal(score[..., 0], objectness_to_filter))
        return tf.gather_nd(y_pred, [y_pred_indices])

    def _preprocess_inputs(y_true, y_pred):
        return y_true[..., :4], y_pred[..., :4]

    def _repulsion_impl(y_true, y_pred,weights):
        len_of_predicted = tf.shape(y_pred)[0]

        ious = bbox_overlap_iou(y_pred, y_true)
        tiled_for_concat = tf.tile(tf.expand_dims(y_true, axis=0), [len_of_predicted, 1, 1])
        ious_over_truth_boxes = tf.concat([tf.expand_dims(ious, axis=2), tiled_for_concat], axis=2)

        return tf.reduce_sum([
            alpha * repulsion_term_gt(y_pred, ious_over_truth_boxes, smooth_rep_gt,weights),
            betta * repulsion_term_box(ious, smooth_rep_box)
        ])

    def _repulsion_loss(y_true, y_pred, score):
        #y_pred = _filter_predictions(y_pred, score)
        #y_true, y_pred = _preprocess_inputs(y_true, y_pred)

        return tf.cond(tf.logical_or(tf.equal(tf.shape(y_pred)[0], 0), tf.equal(tf.shape(y_true)[0], 0)),
                       lambda: tf.Variable(0.0, dtype=tf.float32),
                       lambda: _repulsion_impl(y_true, y_pred,score))

    #repulsion = _repulsion_loss(score,y_true, y_pred, outside_weights,tmp)
    repulsion = _repulsion_loss(y_true, y_pred, score)
    return repulsion