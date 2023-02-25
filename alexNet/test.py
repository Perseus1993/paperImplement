import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test = np.random.randint(1, 10, (4, 100, 2))
    for i in range(len(test)):
        test[i]
        plt.plot(test[i])
    plt.show()
