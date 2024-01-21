#Imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#Read CSV file into dataframe
df = pd.read_csv("diabetes.csv")

#Pre-process the data, check for missing values
print("\n* Checking for presence of zero values in columns:")
print(df.isnull().any())

#Print useful statistics about the data
pd.set_option("display.max_columns", 500)
print("\n*Descriptive Analysis:")
print(df.describe(include='all'))

#Determine row and columns containing zero values
print("\n* Number of rows with 0 values for each variable:")
for col in df.columns:
    missing_rows = df.loc[df[col] == 0].shape[0]
    print(col + ": " + str(missing_rows))

#Remove all incorrect values with NaN
print("\n* Replace all incorrect values with NaN values")
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

print("\n* Replace NaN values with mean values of respective column")
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

#Normalize the data via centering
# Use the scale() function from scikit-learn
print("\n* Centering the data")
df_scaled = preprocessing.scale(df)
# Result must be converted back to a pandas DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
# Do not want the Outcome column to be scaled, so keep the original
df_scaled['Outcome'] = df['Outcome']
df = df_scaled
print(df.describe().loc[['mean', 'std','max'],].round(2).abs())

print("\n* Generate training, validation and testing set")
# Split dataset into an input matrix (all columns but Outcome) and Outcome vector
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']
# Split input matrix to create the training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Second split on training set to create the validation set (20% of training set)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

#Building the multilayer perceptron
print("\n* Building Multilayer Perceptron")
model = Sequential()
print(" - Adding first hidden layer with 32 neurons")
model.add(Dense(32, activation='relu', input_dim=8))
print(" - Adding second hidden layer with 16 neurons")
model.add(Dense(16, activation='relu'))
print(" - Adding output layer")
model.add(Dense(1, activation='sigmoid'))

#Compile the network
print("\n* Compiling the network")
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

#Train the network
print("\n* Training the network")
model.fit(X_train, y_train, epochs=200)

# Evaluate the accuracy with respect to the training set
scores = model.evaluate(X_train, y_train)
print('Training Accuracy: %.2f%%\n' % (scores[1]*100))
# Evaluate the accuracy with respect to the testing set
scores = model.evaluate(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' % (scores[1]*100))

# Construct a confusion matrix
print("\n* Constructing confusion matrix")
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True,
xticklabels=['No Diabetes','Diabetes'],
yticklabels=['No Diabetes','Diabetes'],
cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()

#Construct ROC curve
print("\n* Constructing ROC curve")
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


