import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())
 
# How many fraudulent transactions?
print(transactions['isFraud'].value_counts())

# Summary statistics on amount column
print(transactions['amount'].describe())


# Create isPayment field
transactions['isPayment'] = transactions['type'].apply(lambda x: 1 if x in ('PAYMENT','DEBIT') else 0)

# Create isMovement field
transactions['isMovement'] = transactions['type'].apply(lambda x: 1 if x in ('CASH_OUT','TRANSFER') else 0)

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceOrg']) - abs(transactions['oldbalanceDest'])

# Create features and label variables
features = transactions[['amount','isPayment','isMovement','accountDiff']]
label = transactions['isFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.3)

# Normalize the features variables
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(X_train_sc, y_train)

# Score the model on the training data
print(model.score(X_train, y_train))

# Score the model on the test data
print(model.score(X_test, y_test))

# Print the model coefficients
print(model.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
transaction4 = np.array([548.31, 1.0, 1.0, 5125.5])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2, transaction3, transaction4))

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(model.predict(sample_transactions))

# Show probabilities on the new transactions
print(model.predict_proba(sample_transactions))
