import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

# print(type(breast_cancer_data))
# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)

# print(breast_cancer_data.target, breast_cancer_data.target_names)

from sklearn.model_selection import train_test_split

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors=3)

# classifier.fit(training_data, training_labels)

# print(classifier.score(validation_data, validation_labels))
accuracies = []
for i in range(1,101):
  classifier = KNeighborsClassifier(i)
  classifier.fit(training_data, training_labels)
  # print(classifier.score(validation_data, validation_labels))
  accuracies.append(classifier.score(validation_data, validation_labels))

import matplotlib.pyplot as plt
k_list = range(1,101)

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
