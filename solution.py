import math
import numpy
import numpy as np #(működik a Moodle-ben is)
import csv


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    if (n_cat1 == 0 or n_cat2 == 0):
        return 0
    n_cat1_prob: float = n_cat1 / (n_cat1+n_cat2)
    n_cat2_prob: float = n_cat2 / (n_cat1 + n_cat2)
    entropy: float = (- n_cat1_prob * math.log2(n_cat1_prob) - n_cat2_prob * math.log2(n_cat2_prob))
    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: numpy.ndarray,
                        labels: numpy.ndarray) -> (int, int):
    max_information_gain = -float(math.inf)
    best_value: int
    best_feature: int

    accepted_first: int = 0
    declined_first: int = 0
    accepted_second: int = 0
    declined_second: int = 0
    row_number: int = 0
    all_lines: int

    attribute_index: int = 0
    attribute_values: numpy.ndarray

    for x in range(8):
        attribute_values = np.unique(features[:,attribute_index])
        for value in attribute_values:
            for row in features[:, attribute_index]:
                if row <= value:
                    if labels[row_number] != 0:
                        accepted_first += 1
                    else:
                        declined_first += 1
                else:
                    if labels[row_number] != 0:
                        accepted_second += 1
                    else:
                        declined_second += 1

                row_number += 1
            all_lines = accepted_first + accepted_second + declined_first + declined_second
            if (1 - (((accepted_first + declined_first) / all_lines) * get_entropy(accepted_first, declined_first) +
                     ((accepted_second + declined_second) / all_lines) * get_entropy(accepted_second,
                                                                                     declined_second))) > max_information_gain:
                best_value = value
                best_feature = attribute_index
                max_information_gain = 1 - (
                            ((accepted_first + declined_first) / all_lines) * get_entropy(accepted_first,
                                                                                          declined_first) +
                            ((accepted_second + declined_second) / all_lines) * get_entropy(accepted_second,
                                                                                            declined_second))
            accepted_first = 0
            declined_first = 0
            accepted_second = 0
            declined_second = 0
            row_number = 0
        attribute_index += 1


    return best_feature, best_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    data = np.loadtxt('train.csv', delimiter=',')

    labels = data[:, 8]
    features = data[:,:8]

    result = get_best_separation(features, labels)

    test = np.loadtxt('test.csv', delimiter=',')

    testColumn = test[:, result[0]]

    output: list = []

    for row in testColumn:
        if(row >= result[1]):
            output.append(1)
        else:
            output.append(0)

    with open('results.csv', 'w') as file:
        for line in output:
            file.write(str(line))
            file.write('\n')

    return 0

if __name__ == "__main__":
    main()
