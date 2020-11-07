import os, yaml
import numpy as np
import tensorflow as tf 

def load_yaml(yaml_path):
    assert os.path.exists(yaml_path), "Yaml path does not exist: " + yaml_path
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def set_gpu_devices(gpu):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(physical_devices[gpu], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu], True)


def make_directories(path):
    if not os.path.exists(path):
        print("Path '{}' does not exist.".format(path))
        print("Make directory: " + path)
        os.makedirs(path)
        
    
def fix_random_seed(flag_seed, seed=None):
    if flag_seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print("Numpy and TensorFlow's random seeds fixed: seed=" + str(seed))
    
    else:
        print("Random seed not fixed.")


def show_layers(model):
    """Shows layers in model.
    Args:
        model: A tf.keras.Model object.
    """
    print("================= Model contains the followding layers ================")
    for iter_layer in model.layers:
        print("Layer: ", iter_layer.name)
    print("=======================================================================")


# Functions for graph plotting
def restrict_classes(llrs, labels, list_classes):
    """ 
    Args:
        list_classes: A list of integers.
    Remark:
        (batch, duration, num classes, num classes)
        -> (< batch, duration, num classes, num classes)
    """
    if list_classes == []:
        return llrs, labels

    assert tf.reduce_min(labels).numpy() <= np.min(list_classes)
    assert np.max(list_classes) <= tf.reduce_max(labels).numpy() 
    
    ls_idx = []
    for itr_cls in list_classes:
        ls_idx.append(tf.squeeze(tf.where(labels == itr_cls)))
    idx = tf.concat(ls_idx, axis=0)
    idx = tf.sort(idx)
    
    llrs_rest = tf.gather(llrs, idx, axis=0)
    lbls_rest = tf.gather(labels, idx, axis=0)

    return llrs_rest, lbls_rest


def extract_positive_row(llrs, labels):
    """ Extract y_i-th rows of LLR matrices.
    Args:
        llrs: (batch, duraiton, num classes, num classes)
        labels: (batch,)
    Returns:
        llrs_posrow: (batch, duration, num classes)
    """
    llrs_shape = llrs.shape
    duration = llrs_shape[1]
    num_classes = llrs_shape[2]
    
    labels_oh = tf.one_hot(labels, depth=num_classes, axis=1)
        # (batch, num cls)
    labels_oh = tf.reshape(labels_oh,[-1, 1, num_classes, 1])
    labels_oh = tf.tile(labels_oh, [1, duration, 1, 1])
        # (batch, duration, num cls, 1)

    llrs_pos = llrs * labels_oh
        # (batch, duration, num cls, num cls)
    llrs_posrow = tf.reduce_sum(llrs_pos, axis=2)
        # (batch, duration, num cls): = LLR_{:, :, y_i, :}
        
    return llrs_posrow


def add_max_to_diag(llrs):
    """
    Args:
        llrs: (batch, duration, num classes, num classes)
    Returns:
        llrs_maxdiag: (batch, duration, num classes, num classes),
            max(|llrs|) is added to diag of llrs.
    """
    num_classes = llrs.shape[2]
    
    llrs_abs = tf.abs(llrs)
    llrs_max = tf.reduce_max(llrs_abs)
        # max |LLRs|
    tmp = tf.linalg.tensor_diag([1.] * num_classes) * llrs_max
    tmp = tf.reshape(tmp, [1, 1, num_classes, num_classes])
    llrs_maxdiag = llrs + tmp

    return llrs_maxdiag



