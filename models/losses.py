import numpy as np
import tensorflow as tf
# from datasets.data_processing import sequential_slice, sequential_concat
# from utils.performance_metrics import calc_llrs, calc_oblivious_llrs, threshold_generator


def CE_lite(logits, labels, num_class=2):
    # softmax-CE
    return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                tf.one_hot(labels.astype('int'), num_class), 
                logits))
        

def LLLR_lite(logits, labels):
    # substraction is equal to LLR
    LLR = logits[:, 1] - logits[:, 0]
    return tf.reduce_mean(
        tf.math.abs(
            labels - tf.math.sigmoid(LLR)))


def KLIEP_lite(logits, labels):
    LLR = logits[:, 1] - logits[:, 0]
    KLIEP = tf.reduce_mean(((labels * -2) + 1) * LLR)

    sm_logits = tf.nn.softmax(logits)
    LR = tf.math.divide(sm_logits[:, 1], sm_logits[:, 0])
    LR_class1 = tf.boolean_mask(LR, labels)
    LR = tf.math.divide(sm_logits[:, 0], sm_logits[:, 1])
    LR_class0 = tf.boolean_mask(LR, [not elem for elem in labels])
    KLIEP += (tf.reduce_sum(LR_class1) - 1) + (tf.reduce_sum(LR_class0) - 1)
    return KLIEP
   



def get_gradient_DRE(model, x, y, training, flag_wd, calc_grad,
                    param_CE_loss, param_LLR_loss, param_KLIEP_loss, param_wd):
    """Calculate loss and/or gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        flag_wd: A boolean. Weight decay or not.
        calc_grad: A boolean. Calculate gradient or not.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad. 
                The weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
            aucsat_loss: A scalar Tensor.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output of 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        - All the losses below will be calculated if not calc_grad 
          to log them to TensorBoard.
            total_loss 
            multiplet_loss
            llr_loss 
    """
    
    # For training
    if calc_grad:
        with tf.GradientTape() as tape:
            x = model(x, training)
            total_loss = 0.

            # crossentropy-loss
            crossentropy_loss = CE_lite(x, y)
            total_loss += param_CE_loss * crossentropy_loss
            
            # Loss for Log Likelihood Ratio estimation (LLLR)
            loss_LLR = LLLR_lite(x, y)
            total_loss += param_LLR_loss * loss_LLR
            
            # KLIEP loss
            loss_KLIEP = KLIEP_lite(x, y)
            total_loss += param_KLIEP_loss * loss_KLIEP
            
            # wd_reg
            wd_reg = 0.
            if flag_wd:
                for variables in model.trainable_variables:
                    wd_reg += tf.nn.l2_loss(variables)
                    total_loss += param_wd * wd_reg

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, crossentropy_loss, loss_LLR, loss_KLIEP, wd_reg]

    # For validation and test
    else: 
        x = model(x, training)

        crossentropy_loss = CE_lite(x, y)
        loss_LLR = LLLR_lite(x, y)
        loss_KLIEP = KLIEP_lite(x, y)
        
        total_loss = crossentropy_loss + loss_LLR + loss_KLIEP
        # wd_reg
        wd_reg = 0.
        # for variables in model.trainable_variables:
        #     wd_reg += tf.nn.l2_loss(variables)

        gradients = None
        losses = [total_loss, crossentropy_loss, loss_LLR, loss_KLIEP, wd_reg]

    return gradients, losses, x
