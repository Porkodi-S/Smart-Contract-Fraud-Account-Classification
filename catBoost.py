!pip install catboost
!pip install scikit-learn
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

dataset = pd.read_csv('modified5.csv')
dataset.head()
dataset.describe()

X = dataset.iloc[0:,3:]
Y = dataset['FLAG']

print(X.dtypes)

categorical_features_indices = np.where(X.dtypes != float)[0]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score

model = CatBoostClassifier(
    custom_loss=[metrics.Accuracy()],
    random_seed=42,
    logging_level='Silent'
)

model.fit(
    X_train, Y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, Y_test),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
);

from google.colab import output
output.disable_custom_widget_manager()

cv_params = model.get_params()
cv_params.update({
    'loss_function': metrics.Logloss()
})
cv_data = cv(
    Pool(X, Y, cat_features=categorical_features_indices),
    cv_params,
    plot=True
)

Y_pred = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)

end = time.time()
print('Time taken to execute using Catboost algorithm:', end-start)


print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']),
    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
    np.argmax(cv_data['test-Accuracy-mean'])
))

print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))

from sklearn import metrics
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_test, Y_pred))

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(Y_test, Y_pred))
confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
print('Sensitivity:', TP / float(TP+FN))
print('Specificity:', TN / float(TN+FP))

import itertools
classes = [0, 1]
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix of Catboost")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = confusion.max() / 2.
for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
    plt.text(j, i, format(confusion[i, j], fmt),
             horizontalalignment="center",
             color="white" if confusion[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
ax.set_title('Precision-Recall Curve of XGBoost')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.show()

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
