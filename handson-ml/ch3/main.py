import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(42)

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


def sort_by_target(mnist):
    reorder_train = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[:60000])])
    )[:, 1]
    reorder_test = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[60000:])])
    )[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def save_mnist_data():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, cache=True)
    # we want ints instead of strings
    mnist.target = mnist.target.astype(np.int8)
    # fetch_openml returns an unsorted dataset
    sort_by_target(mnist)
    return mnist


mnist = save_mnist_data()
print(mnist["data"], mnist["target"])
