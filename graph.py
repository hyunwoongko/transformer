"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw():
    train = read('./result/train.txt')
    test = read('./result/test.txt')

    plt.plot(train, 'r', label='train')
    plt.plot(test, 'b', label='validation')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.legend(loc='lower left')

    plt.show()


if __name__ == '__main__':
    draw()
