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

    return [float(i) for idx, i in enumerate(file.split(',')) if idx <= 150]


train = read('./result/train.txt')
test = read('./result/test.txt')

plt.plot(train, 'r', label='train')
plt.plot(test, 'b', label='validation')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training result')
plt.grid(True, which='both', axis='both')
plt.legend(loc='lower left')
plt.xticks([i for i in range(0, 151, 10)])
plt.yticks([i * 0.2 for i in range(17, 30)])

plt.show()
