import pandas as pd
from matplotlib import pyplot as plt

import fire

denmo_mnist_template = 'denmo_{}x{}_mnist_valid_accuracy.csv'
denmo_fashion_mnist_template = 'denmo_{}x{}_fashion_mnist_valid_accuracy.csv'
denmo_cifar_template = 'denmo_{}x{}_cifar10_valid_accuracy.csv'


def table_3():
    mnist = pd.read_csv(denmo_mnist_template.format(200, 200))
    fashion_mnist = pd.read_csv(denmo_fashion_mnist_template.format(400, 400))
    best_mnist = mnist['Value'].max()
    best_fmnist = fashion_mnist['Value'].max()
    print("MNIST test accuracy: {}".format(best_mnist))
    print("Fashion MNIST test accuracy: {}".format(best_fmnist))


def figure_5a():
    even = pd.read_csv(denmo_cifar_template.format(300, 300))
    baselines = pd.read_csv('baselines_cifar10.csv', index_col=0)
    num_epochs = 150

    fig, ax = plt.subplots()

    ax.plot(range(num_epochs), even['Value'] / 100, label='Morph Network', color='red')
    ax.plot(baselines['tanh'], label='NN-tanh', color='xkcd:blue')
    ax.plot(baselines['relu'], label='NN-Relu', color='xkcd:orange')
    ax.plot(baselines['maxout'], label='NN-Maxout', color='green')

    # ax.set_title('Comparison to Baselines(CIFAR-10)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(0, 155, 25))
    ax.legend()
    plt.ylim(0.22, 0.57)
    plt.xlim(-10, 155)
    plt.savefig('figure_5a_reproduction.png')


def figure_5b():
    """Compares runs of DenMo on CIFAR-10 with different distributions of dilations/erosions."""
    even = pd.read_csv(denmo_cifar_template.format(300, 300))
    all_d = pd.read_csv(denmo_cifar_template.format(0, 600))
    all_e = pd.read_csv(denmo_cifar_template.format(600, 0))
    num_epochs = 150

    fig, ax = plt.subplots()

    ax.plot(range(num_epochs), all_d['Value'] / 100, label='Dilation(600)', color='xkcd:azure')
    ax.plot(all_e['Value'] / 100, label='Erosion(600)', color='xkcd:orange')
    ax.plot(even['Value'] / 100, label='Erosion(300) and Dilation(300)', color='green')

    # ax.set_title('Varying Dilation/Erosion Distribution (CIFAR-10)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(0, 160, 25))
    ax.legend()
    plt.ylim(0.2, 0.57)
    plt.xlim(-10, 160)
    plt.savefig('figure_5b_reproduction.png')


def figure_4b():
    """Compares runs of DenMo on MNIST with different distributions of dilations/erosions."""
    even = pd.read_csv(denmo_mnist_template.format(200, 200))
    all_d = pd.read_csv(denmo_mnist_template.format(0, 400))
    all_e = pd.read_csv(denmo_mnist_template.format(400, 0))
    num_epochs = 200

    fig, ax = plt.subplots()

    ax.plot(range(num_epochs), all_d['Value'][:num_epochs] / 100, label='Dilation(400)', color='xkcd:azure')
    ax.plot(range(num_epochs), all_e['Value'][:num_epochs] / 100, label='Erosion(400)', color='orange')
    ax.plot(range(num_epochs), even['Value'][:num_epochs]  / 100, label='Erosion(200) and Dilation(200)', color='green')

    # ax.set_title('Varying Dilation/Erosion Distribution (MNIST)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend()
    plt.ylim(0.68, 0.99)
    plt.xlim(-10, 205)
    plt.savefig('figure_4b_reproduction.png')


def figure_4a():
    """Compares runs of DenMo on MNIST with different layer widths."""
    width_10 = pd.read_csv(denmo_mnist_template.format(5, 5))
    width_50 = pd.read_csv(denmo_mnist_template.format(25, 25))
    width_100 = pd.read_csv(denmo_mnist_template.format(50, 50))
    width_200 = pd.read_csv(denmo_mnist_template.format(100, 100))
    num_epochs = 400

    fig, ax = plt.subplots()

    ax.plot(range(num_epochs), width_10['Value'] / 100, label='l=10', color='magenta')
    ax.plot(range(num_epochs), width_50['Value'] / 100, label='l=50', color='blue')
    ax.plot(range(num_epochs), width_100['Value'] / 100, label='l=100', color='green')
    ax.plot(range(num_epochs), width_200['Value'] / 100, label='l=200', color='red')

    # ax.set_title('Varying Layer Width (MNIST)')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend()
    plt.ylim(0, 1.05)
    plt.xlim(-10, 150)
    plt.savefig('figure_4a_reproduction.png')


if __name__ == '__main__':
    fire.Fire()
