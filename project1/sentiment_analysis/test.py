import os
import sys
import time
import traceback
import project1 as p1
import numpy as np

verbose = False

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_real(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if res != exp_res:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def equals(x, y):
    if type(y) == np.ndarray:
        return (x == y).all()
    return x == y

def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == tuple:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_array(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == list:
        log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def check_get_order():
    ex_name = "Get order"
    if check_list(
            ex_name, p1.get_order,
            [0], 1):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    if check_list(
            ex_name, p1.get_order,
            [1, 0], 2):
        log("You should revert `get_order` to its original implementation for this test to pass")
        return
    log(green("PASS"), ex_name, "")


def check_hinge_loss_single():
    ex_name = "Hinge loss single"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_single,
            exp_res, feature_vector, label, theta, theta_0):
        return
    log(green("PASS"), ex_name, "")


def check_hinge_loss_full():
    ex_name = "Hinge loss full"

    feature_vector = np.array([[1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
    exp_res = 1 - 0.8
    if check_real(
            ex_name, p1.hinge_loss_full,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_perceptron_single_update():
    ex_name = "Perceptron single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    exp_res = (np.array([0, 3]), -0.5)
    if check_tuple(
            ex_name, p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1
    exp_res = (np.array([0, 3]), 0)
    if check_tuple(
            ex_name + " (boundary case)", p1.perceptron_single_step_update,
            exp_res, feature_vector, label, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_perceptron():
    ex_name = "Perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 2
    exp_res = (np.array([0, 2]), 2)
    if check_tuple(
            ex_name, p1.perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")


def check_average_perceptron():
    ex_name = "Average perceptron"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2], [-1, 0]])
    labels = np.array([1, 1])
    T = 1
    exp_res = (np.array([-0.5, 1]), 1.5)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    feature_matrix = np.array([[ 0.,          0.,          0.,          0.,          0.,          0.,
   0.,          0.,          0.,          0.,        ],
 [-0.64452739, -0.8686877,  -0.16365435, -0.24538229, -0.09271405, -0.71546825,
  -0.8785896,  -0.5185619,  -0.19447116, -0.6958803 ],
 [ 0.82767276,  0.32737563,  0.7907481,   0.98479046,  0.42418538,  0.16130865,
   0.70270002,  0.48820146,  0.05909762,  0.0605582 ],
 [ 0.3314781,   0.40438026,  0.92653649,  0.23157254,  0.14131511,  0.70287698,
   0.34851775,  0.29255411,  0.37816325,  0.58655075],
 [ 0.04709424,  0.04638772,  0.85014331,  0.32091319,  0.83123805,  0.49344437,
   0.3005928,   0.78050199,  0.63589838,  0.62005912],
 [ 0.54205487,  0.76363792,  0.86665137,  0.91272214,  0.02170441,  0.15033594,
   0.25854785,  0.83033588,  0.3155424,  0.31395639],
 [-0.12511146, -0.67416201, -0.28205421, -0.29889979, -0.96778155, -0.20299508,
  -0.67962779, -0.50086843, -0.0878882,  -0.43814884],
 [-0.11815033, -0.9042513,  -0.5624174,  -0.04130574, -0.43576283, -0.22174509,
  -0.37497967, -0.41588909, -0.54356096, -0.98478622],
 [-0.67162023, -0.41017803, -0.26198418, -0.64288407, -0.31358074, -0.04250198,
  -0.1004534,  -0.6263229,  -0.74234598, -0.83899247],
 [ 0.73953117,  0.41271733,  0.04389472,  0.31698206,  0.89385302,  0.26323054,
   0.4680077,   0.28657905,  0.00138099,  0.54763486]])
    labels = np.array([-1, -1,  1,  1,  1,  1, -1, -1, -1,  1])
    T = 100
    exp_res = (np.array(['0.0221859', '0.0123815', '0.0013168', '0.0095095', '0.0268156', '0.0078969', '0.0140402', '0.0085974', '0.0000414', '0.0164290']), 1.75)
    if check_tuple(
            ex_name, p1.average_perceptron,
            exp_res, feature_matrix, labels, T):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos_single_update():
    ex_name = "Pegasos single update"

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.4)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 1])
    label, theta, theta_0 = 1, np.array([-1, 1]), 1
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.08]), 1.1)
    if check_tuple(
            ex_name +  " (boundary case)", p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    feature_vector = np.array([1, 2])
    label, theta, theta_0 = 1, np.array([-1, 1]), -2
    L = 0.2
    eta = 0.1
    exp_res = (np.array([-0.88, 1.18]), -1.9)
    if check_tuple(
            ex_name, p1.pegasos_single_step_update,
            exp_res,
            feature_vector, label, L, eta, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")


def check_pegasos():
    ex_name = "Pegasos"

    feature_matrix = np.array([[1, 2]])
    labels = np.array([1])
    T = 1
    L = 0.2
    exp_res = (np.array([1, 2]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    feature_matrix = np.array([[1, 1], [1, 1]])
    labels = np.array([1, 1])
    T = 1
    L = 1
    exp_res = (np.array([1-1/np.sqrt(2), 1-1/np.sqrt(2)]), 1)
    if check_tuple(
            ex_name, p1.pegasos,
            exp_res, feature_matrix, labels, T, L):
        return

    log(green("PASS"), ex_name, "")


def check_classify():
    ex_name = "Classify"

    feature_matrix = np.array([[1, 1], [1, 1], [1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([1, 1, 1])
    if check_array(
            ex_name, p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    feature_matrix = np.array([[-1, 1]])
    theta = np.array([1, 1])
    theta_0 = 0
    exp_res = np.array([-1])
    if check_array(
            ex_name + " (boundary case)", p1.classify,
            exp_res, feature_matrix, theta, theta_0):
        return

    log(green("PASS"), ex_name, "")

def check_classifier_accuracy():
    ex_name = "Classifier accuracy"

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.perceptron,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T):
        return

    train_feature_matrix = np.array([[1, 0], [1, -1], [2, 3]])
    val_feature_matrix = np.array([[1, 1], [2, -1]])
    train_labels = np.array([1, -1, 1])
    val_labels = np.array([-1, 1])
    exp_res = 1, 0
    T=1
    L=0.2
    if check_tuple(
            ex_name, p1.classifier_accuracy,
            exp_res,
            p1.pegasos,
            train_feature_matrix, val_feature_matrix,
            train_labels, val_labels,
            T=T, L=L):
        return

    log(green("PASS"), ex_name, "")

def check_bag_of_words():
    ex_name = "Bag of words"

    texts = [
        "He loves to walk on the beach",
        "There is nothing better"]

    try:
        res = p1.bag_of_words(texts)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return
    if not type(res) == dict:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return

    vals = sorted(res.values())
    exp_vals = list(range(len(res.keys())))
    if not vals == exp_vals:
        log(red("FAIL"), ex_name, ": wrong set of indices. Expected: ", exp_vals, " got ", vals)
        return

    log(green("PASS"), ex_name, "")

    keys = sorted(res.keys())
    exp_keys = ['beach', 'better', 'he', 'is', 'loves', 'nothing', 'on', 'the', 'there', 'to', 'walk']
    stop_keys = ['beach', 'better', 'loves', 'nothing', 'walk']

    if keys == exp_keys:
        log(yellow("WARN"), ex_name, ": does not remove stopwords:", [k for k in keys if k not in stop_keys])
    elif keys == stop_keys:
        log(green("PASS"), ex_name, " stopwords removed")
    else:
        log(red("FAIL"), ex_name, ": keys are missing:", [k for k in stop_keys if k not in keys], " or are not unexpected:", [k for k in keys if k not in stop_keys])


def check_extract_bow_feature_vectors():
    ex_name = "Extract bow feature vectors"
    texts = [
        "He loves her ",
        "He really really loves her"]
    keys = ["he", "loves", "her", "really"]
    dictionary = {k:i for i, k in enumerate(keys)}
    exp_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 1]])
    non_bin_res = np.array(
        [[1, 1, 1, 0],
        [1, 1, 1, 2]])


    try:
        res = p1.extract_bow_feature_vectors(texts, dictionary)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return

    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return

    log(green("PASS"), ex_name)

    if (res == exp_res).all():
        log(yellow("WARN"), ex_name, ": uses binary indicators as features")
    elif (res == non_bin_res).all():
        log(green("PASS"), ex_name, ": correct non binary features")
    else:
        log(red("FAIL"), ex_name, ": unexpected feature matrix")
        return

def main():
    log(green("PASS"), "Import project1")
    try:
        check_get_order()
        check_hinge_loss_single()
        check_hinge_loss_full()
        check_perceptron_single_update()
        check_perceptron()
        check_average_perceptron()
        check_pegasos_single_update()
        check_pegasos()
        check_classify()
        check_classifier_accuracy()
        check_bag_of_words()
        check_extract_bow_feature_vectors()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()
