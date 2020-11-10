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






def multiplet_loss_func(logits_slice, labels_slice):
    """Multiplet loss for density estimation of time-series data.
    Args:
        model: A model.backbones_lstm.LSTMModel object. 
        logits_slice: An logit Tensor with shape 
            ((effective) batch size, order of SPRT + 1, num classes). 
            This is the output of LSTMModel.call(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,)
    Returns:
        xent: A scalar Tensor. Sum of multiplet losses.
    """
    # Calc multiplet losses     
    logits = tf.transpose(logits_slice, [1, 0, 2])
    logits = tf.reshape(logits, [-1, logits.shape[2]])
    labels = tf.tile(labels_slice, [logits_slice.shape[1],])
    xent = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits) 
        #A scalar averaged in a batch and sliding windows

    return xent


def margin_generator(llrs, labels_oh):
    """ Used in LLLR.
    Args:
        llrs: A Tensor with shape
            (batch, duration, num cls, num cls).
        labels_oh: A Tensor with shape
            (batch, 1, num cls, 1).
    Returns:
        random_margin: A Tensor with shape
            (batch, duration, num cls, num cls)
            - is negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
            - is posotive for negative class row (\lambda_{y_i k} (l \in [num_classes]))
            - All the margins share the same absolute value. 
    """
    labels_oh = 1 - (2 * labels_oh)
        # positive = -1, negative = 1
        # (batch, 1, num cls, 1)
    random_margin = threshold_generator(llrs, 1, "unirandom")
        # (1, duration, num cls, num cls)
        # positive values (diag = 0)
    random_margin = random_margin * labels_oh
        # (batch, duration, num cls, num cls)
        # negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
        # posotive for negative class row (\lambda_{y_i k} (l \in [num_classes])) 

    return random_margin


def LLLR(logits_concat, labels_concat, oblivious, version, flag_mgn):
    """LLLR for early multi-classification of time series.
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A label Tensor with shape (batch size,). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        oblivious: A bool, whether to use TANDEMwO or not (= TANDEM).
        version: "A", "B", "C", or "D".
        flag_mgn: Use margin or not.
    Return:
        llr_loss: A scalar Tensor that represents the log-likelihoood ratio loss.
    Remark:
        - version A: original (use all LLRs)
        - version B: simple LLLR (extract the row of the positive class)
        - version C: logistic LLLR (logistic loss instead of sigmoid)
        - version D: simple logistic LLLR
        - margin is generated uniformly-randomly.
    """
    shapes = logits_concat.shape
    order_sprt = shapes[2] - 1
    duration = shapes[1] + order_sprt
    num_classes = shapes[3]
    
    labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1, dtype=tf.float32)
    labels_oh = tf.reshape(labels_oh, [-1, 1, num_classes, 1])
        # (batch, 1, num cls, 1)

    if oblivious:
        llrs = calc_oblivious_llrs(logits_concat)
    else:
        llrs = calc_llrs(logits_concat) 
            # (batch, duration, num cls, num cls)

    if flag_mgn:
        random_margin = margin_generator(llrs, labels_oh)
        llrs += random_margin
            # negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
            # posotive for negative class row (\lambda_{y_i k} (l \in [num_classes])) 

    if version == "A":
        lllr = tf.abs(labels_oh - tf.sigmoid(llrs))
            # (batch, duration, num cls, num cls)
        lllr = 0.5 * (num_classes / (num_classes - 1.)) * tf.reduce_mean(lllr)

    elif version == "B":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        lllr = tf.abs(1. - tf.sigmoid(llrs))
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)

    elif version == "C":
        labels_oh = tf.tile(labels_oh, [1, duration, 1, num_classes])
            # (batch, duration, num cls, num cls)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_oh, logits=llrs)
            # (batch, duration, num cls, num cls)
        lllr = 0.5 * (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)        

    elif version == "D":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.reshape(llrs, [-1, num_classes])
        z = tf.ones_like(llrs, dtype=tf.float32)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=llrs)
            # (batch * duration, num cls)
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)

    else:
        raise ValueError("version={} must be either of 'A', 'B', 'C', or 'D'.".format(version))

    return lllr # scalar
    

def get_gradient_lstm(model, x, y, training, order_sprt, duration, 
    oblivious, version, flag_wd, flag_mgn, calc_grad, 
    param_multiplet_loss, param_llr_loss, param_wd):
    """Calculate loss and/or gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames in a sequence.
        oblivious: A bool. TANDEMwO or normal TANDEM.
        version: A string, "A", "B", "C", or "D".
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
        flag_mgn: A boolean. Use margin in LLLR or not.
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
    x_slice, y_slice = sequential_slice(x, y, order_sprt)

    # For training
    if calc_grad:
        with tf.GradientTape() as tape:
            logits_slice = model(x_slice, training)
            logits_concat, y_concat = sequential_concat(
                logits_slice, y_slice, duration)
            total_loss = 0.

            # multiplet_loss, llr_loss
            if param_multiplet_loss != 0:
                multiplet_loss = multiplet_loss_func(logits_slice, y_slice)
                total_loss += param_multiplet_loss * multiplet_loss
            else:
                multiplet_loss = 0.

            if param_llr_loss != 0:
                lllr = LLLR(logits_concat, y_concat, oblivious, version, flag_mgn)
                total_loss += param_llr_loss * lllr
            else:
                lllr = 0.

            # wd_reg
            wd_reg = 0.
            if flag_wd:
                for variables in model.trainable_variables:
                    wd_reg += tf.nn.l2_loss(variables)
                    total_loss += param_wd * wd_reg

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, multiplet_loss, lllr, wd_reg]

    # For validation and test
    else: 
        logits_slice = model(x_slice, training)
        logits_concat, y_concat = sequential_concat(
            logits_slice, y_slice, duration)
        total_loss = 0.

        # multiplet_loss, llr_loss
        multiplet_loss = multiplet_loss_func(logits_slice, y_slice)
        lllr = LLLR(logits_concat, y_concat, oblivious, version, flag_mgn)

        # wd_reg
        wd_reg = 0.
        # for variables in model.trainable_variables:
        #     wd_reg += tf.nn.l2_loss(variables)

        gradients = None
        losses = [0., multiplet_loss, lllr, wd_reg]

    return gradients, losses, logits_concat


def get_loss_fe(model, x, y, flag_wd, training, calc_grad, param_wd):
    """
    Args:
        model: A tf.keras.Model object.
        x: A Tensor with shape=(batch, H, W, C).
        y: A Tensor with shape (batch,).
        flag_wd: A boolean, whether to decay weight here.
        training: A boolean, the training flag.
        calc_grad: A boolean, whether to calculate gradient.
    """
    if calc_grad:
        with tf.GradientTape() as tape:
            logits, bottleneck_feat = model(x, training)
                # (batch, 2) and (batch, final_size)

            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=logits))
            
            total_loss = xent

            if flag_wd:
                for variables in model.trainable_variables:
                    total_loss += param_wd * tf.nn.l2_loss(variables)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, xent]

    else:
        logits, bottleneck_feat = model(x, training)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))

        gradients = None 
        losses = [0., xent]

    return gradients, losses, logits, bottleneck_feat


# def threshold_loss(bac, mht, bac_base, mht_base, duration, m_bac=0.01, m_mht=1.):
#     """ Numpy calc threshold loss.
#     Args:
#         bac: A float. 
#         mht: A float. 
#         bac_base: A float. 
#         mht_base: A float.
#         duration: An int. 
#         m_bac: A float. Too large values lead to non-vanishing biases in thresh_loss. Default=0.01.
#         m_mht: An int or float. A frame-based margin for mean hitting time. Too large values lead to non-vanishing biases in thresh_loss. Default=1.
#     Return:
#         thresh_loss: A numpy scalar.
#     Remarks:
#         - If m_bac=0.01, then 1% improvement of balanced accuracy is expected by thresholding.
#         - If .m_mht=1, then 1 frame improvement of mean hitting time is expected by thresholding.
#     """
#     loss1 = np.max([bac_base + m_bac - bac, 0.])
#     loss2 = np.max([mht + m_mht - mht_base, 0.]) / duration
#     thresh_loss = loss1 + loss2
    
#     return thresh_loss
