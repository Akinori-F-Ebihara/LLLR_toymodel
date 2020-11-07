# Density Ratio Estimation Experiment

# In this article, we test whether the prPoposed LLLR can help a neural network estimating
# the true log-likelihood ratio. The inputs and networks are simplified to multidimensional
# Gaussian random variables and a 3-layer fully-connected network with nonlinear activation,
# respectively.

# Tested environment: tensorflow 2.0.0.

# Nov. 4th, 2020. Akinori F. Ebihara.


# Experiment Settings
# Our experimental settings are based on Sugiyama et al. 2008, "Direct importance estimation for covariate shift adaptation":

# Let 𝑝0(𝑥) be the 𝑑-dimensional Gaussian density with mean (𝑎,0,0,...,0) and covariance identity, and 𝑝1(𝑥) be the 𝑑-dimensional Gaussian density with mean (0,𝑎,0,...,0) and covariance identity.

# The task for the neural network is to estimate the density ratio:

# 𝑟̂ (𝑥)=𝑝̂ 1(𝑥)𝑝̂ 0(𝑥).

# Here, 𝑥 is sampled either from 𝑝0 or 𝑝1. We compared 3 loss functions: (1) KLIEP, (2) LLLR, and (3) cross-entropy loss.


from __future__ import absolute_import, division, print_function
import datetime, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform
from utils.misc import load_yaml, set_gpu_devices, fix_random_seed
from utils.util_tensorboard import TensorboardLogger
from utils.util_ckpt import checkpoint_logger
from models.backbones_DRE import MLP4DRE
from models.optimizers import get_optimizer
from models.losses import get_gradient_DRE

# load parameters
config_path = '/raid6/ebihara/python/SPRTproject/Density_Estimation_with_LLLR/config/config_CE.yaml'
config = load_yaml(config_path)
stat_N = 10

# for train logs
def tblog_writer(tblogger, losses, eval_metrics, global_step, phase):
    # Losses
    tblogger.scalar_summary("{}_loss/Sum_loss".format(phase),
        losses[1] + losses[2] + losses[3], int(global_step))
    tblogger.scalar_summary("{}_loss/CE_loss".format(phase),
        losses[1], int(global_step))
    tblogger.scalar_summary("{}_loss/LLLR".format(phase),
        losses[2], int(global_step))
    tblogger.scalar_summary("{}_loss/LLLRv2".format(phase),
        losses[3], int(global_step))
    tblogger.scalar_summary("{}_loss/KLIEP".format(phase),
        losses[4], int(global_step))

    # results of density-ratio estimation
    # Normalized Mean Squared Error (NMSE)
    tblogger.scalar_summary("{}_metric/LR_NMSE".format(phase),
        eval_metrics[0], int(global_step))
    tblogger.scalar_summary("{}_metric/LLR_NMSE".format(phase),
        eval_metrics[1], int(global_step))
    # Mean Absolute Error (MABS)
    tblogger.scalar_summary("{}_metric/LR_MABS".format(phase),
        eval_metrics[2], int(global_step))
    tblogger.scalar_summary("{}_metric/LLR_MABS".format(phase),
        eval_metrics[3], int(global_step))


def LLR(x, pdf0, pdf1):
    LLRs = np.log(pdf1.pdf(x) / pdf0.pdf(x))
    LRs = pdf1.pdf(x) / pdf0.pdf(x)
    return LLRs, LRs


def generate_data(mean1, mean2, covmat, batch_size):
    # sample from p1 and p2
    x0 = np.random.multivariate_normal(mean1, covmat, batch_size//2).astype('float32')
    x1 = np.random.multivariate_normal(mean2, covmat, batch_size//2).astype('float32')

    y0 = np.zeros((batch_size//2))
    y1 = np.ones((batch_size//2))

    pdf0 = multivariate_normal(mean1, covmat)
    pdf1 = multivariate_normal(mean2, covmat)

    LLR0, LR0 = LLR(x0, pdf0, pdf1)
    LLR1, LR1 = LLR(x1, pdf0, pdf1)

    X = np.concatenate((x0, x1), axis=0)
    Y = np.concatenate((y0, y1), axis=0)

    LLRs = np.concatenate((LLR0, LLR1), axis=0)
    LRs = np.concatenate((LR0, LR1), axis=0)

    return X, Y, LLRs, LRs

# eval_metrics = calc_NMSE_MABS(GT_LLRs, logits)
def calc_NMSE_MABS(GT_LRs, GT_LLRs, logits):
    estimated_LRs = tf.nn.softmax(logits).numpy()
    estimated_LRs = estimated_LRs[:, 1] / estimated_LRs[:, 0]

    estimated_LLRs = (logits[:, 1] - logits[:, 0]).numpy()

    LR_NMSE = np.mean((GT_LRs / np.sum(GT_LRs) - estimated_LRs / np.sum(estimated_LRs))**2)
    LLR_NMSE = np.mean((GT_LLRs / np.sum(GT_LLRs) - estimated_LLRs / np.sum(estimated_LLRs))**2)

    LR_MABS = np.mean(np.abs(GT_LRs, estimated_LRs))
    LLR_MABS = np.mean(np.abs(GT_LLRs, estimated_LLRs))

    return LR_NMSE, LLR_NMSE, LR_MABS, LLR_MABS



if __name__ == '__main__':

    data_dim = config['data_dim']
    density_offset = config['density_offset']
    batch_size = config['batch_size']

    # GPU settings
    set_gpu_devices(config["gpu"])

    # Set Random Seeds (Optional)
    fix_random_seed(flag_seed=config["flag_seed"], seed=config["seed"])

    # create the two probability density functions.
    covmat = np.eye(data_dim)
    mean0 = np.zeros((data_dim))
    mean0[0] = density_offset
    mean1 = np.zeros((data_dim))
    mean1[1] = density_offset

    pdf0 = multivariate_normal(mean0, covmat)
    pdf1 = multivariate_normal(mean1, covmat)

    # sample from p1 and p2
    x0 = np.random.multivariate_normal(mean0, covmat, batch_size//2).astype('float32')
    x1 = np.random.multivariate_normal(mean1, covmat, batch_size//2).astype('float32')

    LLR0 = LLR(x0, pdf0, pdf1)
    LLR1 = LLR(x1, pdf0, pdf1)

    # # (optional) visualize the first two dimensions of x1/x2
    # fig = plt.figure(figsize=(10,7))
    # fig.patch.set_facecolor('white')
    # plt.rcParams['font.size'] = 25
    # plt.scatter(x0[:, 0], x0[:, 1])
    # plt.scatter(x1[:, 0], x1[:, 1])
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.show()

    loss_pool = np.zeros((stat_N, config['num_iter'] // config['validation_step'], 2))
    NMSE_pool = np.zeros((stat_N, config['num_iter'] // config['validation_step']))

    for ind in range(stat_N):
        tf.compat.v1.reset_default_graph()

        evalpoint = 0
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

        # setup the network
        model = MLP4DRE(
            nb_cls=config["num_classes"],
            feat_dim=data_dim)

        # setup the optimizer
        optimizer, flag_wd_in_loss = get_optimizer(
            learning_rates=config["learning_rates"],
            decay_steps=config["lr_decay_steps"],
            name_optimizer=config["name_optimizer"],
            flag_wd=config["flag_wd"],
            weight_decay=config["weight_decay"])

        # Tensorboard and checkpoints
        ####################################
        # Define global step
        global_step = tf.Variable(0, name="global_step", dtype=tf.int32)

        # Checkpoint
        _, ckpt_manager = checkpoint_logger(
            global_step,
            model,
            optimizer,
            config["flag_resume"],
            config["root_ckptlogs"],
            config["subproject_name"],
            config["exp_phase"],
            config["comment"],
            now,
            config["path_resume"],
            config["max_to_keep"],
            config_path)

        # Tensorboard
        #tf.summary.experimental.set_step(global_step)
        tblogger = TensorboardLogger(
            root_tblogs=config["root_tblogs"],
            subproject_name=config["subproject_name"],
            exp_phase=config["exp_phase"],
            comment=config["comment"],
            time_stamp=now)


        # Start training
        with tblogger.writer.as_default():
            # Initialization
            estimation_error_pool = np.zeros((config['num_iter'], 2)) # 2 metrics: NMSE and MABS

            for epoch in range(config['num_iter']):
                # Training loop
                x_batch, y_batch, GT_LLRs, GT_LRs = generate_data(mean0, mean1, covmat, batch_size)

                # Show summary of model
                if epoch == 0:
                    model.build(input_shape=x_batch.shape)
                    model.summary()

                # Calc loss and grad, and backpropagation
                grads, losses, logits = get_gradient_DRE(
                    model,
                    x_batch,
                    y_batch,
                    training=True,
                    flag_wd=flag_wd_in_loss,
                    calc_grad=True,
                    param_CE_loss=config["param_CE_loss"],
                    param_LLR_loss=config["param_LLR_loss"],
                    param_LLLR_v2=config["param_LLLR_v2"],
                    param_KLIEP_loss=config['param_KLIEP_loss'],
                    param_wd=config["weight_decay"]
                    )

                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                LR_NMSE, LLR_NMSE, LR_MABS, LLR_MABS = calc_NMSE_MABS(GT_LRs, GT_LLRs, logits)
                global_step.assign_add(1)

                # train log
                if tf.equal(global_step % config['train_display_step'], 0) or tf.equal(global_step, 1):
                    print('Global Step={:7d}/{:7d}'.format(int(global_step), config['num_iter']))
                    print('Train CE loss:{:7.5f} * {}'.format(losses[1], str(config['param_CE_loss'])))
                    print('Train LLLR :{:7.5f} * {}'.format(losses[2], str(config['param_LLR_loss'])))
                    print('Train LLLR v2:{:7.5f} * {}'.format(losses[3], str(config['param_LLLR_v2'])))
                    print('Train KLIEP loss:{:7.5f} * {}'.format(losses[4], str(config['param_KLIEP_loss'])))

                    # Tensorboard
                    tblog_writer(
                        tblogger,
                        losses,
                        [LR_NMSE, LLR_NMSE, LR_MABS, LLR_MABS],
                        global_step,
                        phase='train')

                # validation
                if tf.equal(global_step % config['validation_step'], 0) or tf.equal(global_step, 1):

                    x_batch, y_batch, GT_LLRs, GT_LRs = generate_data(mean0, mean1, covmat, batch_size)
                    # Calc loss and grad, and backpropagation
                    _, losses, logits = get_gradient_DRE(
                        model,
                        x_batch,
                        y_batch,
                        training=False,
                        flag_wd=flag_wd_in_loss,
                        calc_grad=False,
                        param_CE_loss=config["param_CE_loss"],
                        param_LLR_loss=config["param_LLR_loss"],
                        param_LLLR_v2=config["param_LLLR_v2"],
                        param_KLIEP_loss=config['param_KLIEP_loss'],
                        param_wd=config["weight_decay"]
                        )

                    LR_NMSE, LLR_NMSE, LR_MABS, LLR_MABS = calc_NMSE_MABS(GT_LRs, GT_LLRs, logits)

                    NMSE_pool[ind, evalpoint] = LR_NMSE
                    loss_pool[ind, evalpoint, 0] = losses[1] # crossentropy loss
                    loss_pool[ind, evalpoint, 1] = losses[2] # LLLR
                    evalpoint += 1

                    if tf.equal(global_step, 1):
                        best = LR_NMSE

                    print('Global Step={:7d}/{:7d}'.format(int(global_step), config['num_iter']))
                    print('Validation CE loss:{:7.5f} * {}'.format(losses[1], str(config['param_CE_loss'])))
                    print('Validation LLLR :{:7.5f} * {}'.format(losses[2], str(config['param_LLR_loss'])))
                    print('Train LLLR v2:{:7.5f} * {}'.format(losses[3], str(config['param_LLLR_v2'])))
                    print('Validation KLIEP loss:{:7.5f} * {}'.format(losses[4], str(config['param_KLIEP_loss'])))

                    print('Validation LR_NMSE:{:.10f}'.format(LR_NMSE))
                    print('Validation LLR_NMSE:{:.10f}'.format(LLR_NMSE))
                    print('Validation LR_MABS:{:.10f}'.format(LR_MABS))
                    print('Validation LLR_MABS:{:.10f}\n'.format(LLR_MABS))

                    # Tensorboard
                    tblog_writer(
                        tblogger,
                        losses,
                        [LR_NMSE, LLR_NMSE, LR_MABS, LLR_MABS],
                        global_step,
                        phase='validation')

                    # Save checkpoint
                    if best > LR_NMSE and int(global_step) > 1:
                        best = LR_NMSE

                        ckpt_manager._checkpoint_prefix = \
                                    ckpt_manager._checkpoint_prefix[:ckpt_manager._checkpoint_prefix.rfind("/") + 1] + \
                                    "ckpt_step{}_LR_NMSE{:.10f}".format(int(global_step), best)
                        save_path_prefix = ckpt_manager.save()
                        print("Best value updated. Saved checkpoint for step {}: {}".format(
                            int(global_step), save_path_prefix))

    np.save(os.path.join(config["root_ckptlogs"], 'loss_pool_{}_{}_{}.npy'.format(
        config["param_LLR_loss"], config["param_CE_loss"], np.mean(loss_pool))), loss_pool)
    np.save(os.path.join(config["root_ckptlogs"], 'NMSE_pool_{}_{}_{}.npy'.format(
        config["param_LLR_loss"], config["param_CE_loss"], np.mean(NMSE_pool))), NMSE_pool)
    print('all done!')
