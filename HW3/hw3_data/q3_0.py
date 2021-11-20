'''
Question 3.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        means.append([])
        for j in range(len(i_digits[0])):
            means[i].append(np.mean(i_digits[:, j]))
        means[i] = np.array(means[i]).reshape(8,8)

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a3digits.zip', './HW3/hw3_data/')
    plot_means(train_data, train_labels)
