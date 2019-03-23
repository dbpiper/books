import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from json_tricks import dump, load

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


def download_mnist_data():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, cache=True)
    # we want ints instead of strings
    mnist.target = mnist.target.astype(np.int8)
    # fetch_openml returns an unsorted dataset
    sort_by_target(mnist)
    return mnist

# This is slower than the default version!!
# JSON serializing/deserializing the numpy sparse array seems to
# not work very well!
#
# def load_mnist_data():
#     if not os.path.exists("data"):
#         os.makedirs("data")

#     with open("data/mnist-data.json", "a+") as mnist_file:
#         # read/write from the start of the file
#         # overwriting if needed
#         mnist_file.seek(0)
#         if os.path.getsize("data/mnist-data.json") <= 0:
#             print("no file!!!!")
#             mnist_data = download_mnist_data()
#             dump(mnist_data, mnist_file)
#             return mnist_data
#         else:
#             print("file exists!!")
#             return load(mnist_file)


mnist = download_mnist_data()
print(mnist["data"], mnist["target"])

