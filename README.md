# Qiskit

from qiskit import *
import pandas as pd
import seaborn as sns
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('tescochurn.csv')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


label_encoder = LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])
df['Churn'] = label_encoder.fit_transform(df['Churn'])


df_encoded = pd.get_dummies(df['MultipleLines'], prefix='MultipleLines')
df = pd.concat([df, df_encoded], axis=1)
df.drop('MultipleLines', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['InternetService'], prefix='InternetService')
df = pd.concat([df, df_encoded], axis=1)
df.drop('InternetService', axis=1, inplace=True)

# Apply one-hot encoding to 'OnlineSecurity' column
df_encoded = pd.get_dummies(df['OnlineSecurity'], prefix='OnlineSecurity')
df = pd.concat([df, df_encoded], axis=1)
df.drop('OnlineSecurity', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['OnlineBackup'], prefix='OnlineBackup')
df = pd.concat([df, df_encoded], axis=1)
df.drop('OnlineBackup', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['DeviceProtection'], prefix='DeviceProtection')
df = pd.concat([df, df_encoded], axis=1)
df.drop('DeviceProtection', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['TechSupport'], prefix='TechSupport')
df = pd.concat([df, df_encoded], axis=1)
df.drop('TechSupport', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['StreamingTV'], prefix='StreamingTV')
df = pd.concat([df, df_encoded], axis=1)
df.drop('StreamingTV', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['StreamingMovies'], prefix='StreamingMovies')
df = pd.concat([df, df_encoded], axis=1)
df.drop('StreamingMovies', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['Contract'], prefix='Contract')
df = pd.concat([df, df_encoded], axis=1)
df.drop('Contract', axis=1, inplace=True)

df_encoded = pd.get_dummies(df['PaymentMethod'], prefix='PaymentMethod')
df = pd.concat([df, df_encoded], axis=1)
df.drop('PaymentMethod', axis=1, inplace=True)

del df['customerID']
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(df['TotalCharges'].dtype)
df.isnull().sum()

columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print(df.head())

selected_featuresQ = ['Contract_Month-to-month','tenure', 'TotalCharges', 'OnlineSecurity_No','TechSupport_No','InternetService_Fiber optic','PaymentMethod_Electronic check','MonthlyCharges','Contract_Two year', 'InternetService_DSL']

selected_features = ['Contract_Month-to-month','tenure', 'TotalCharges', 'OnlineSecurity_No','TechSupport_No','InternetService_Fiber optic','PaymentMethod_Electronic check','MonthlyCharges','Contract_Two year', 'InternetService_DSL', 'Churn']
df2 = df[selected_features].copy()

target_variable = 'Churn'
target = sampled_df[target_variable]

from sklearn.model_selection import train_test_split

sampled_df.drop(target_variable, axis=1, inplace=True)

print("DataFrame without the target variable:")
print(df2.head())

X_train, X_test, y_train, y_test = train_test_split(sampled_df, target, test_size=0.2, random_state=42)


from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)
sample_size = 700

sampled_df = df2.sample(n=sample_size, random_state=42)

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = svc.predict(X_test)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)


from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
start_time = time.time()


end_time = time.time()

execution_time = end_time - start_time

print(f"The training time of the SVC model is {execution_time} seconds.")


train_score_c4 = svc.score(X_train, y_train)
test_score_c4 = svc.score(X_test, y_test)

print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")


from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=10598, seed_transpiler=10598)

num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)

quantum_kernel = QuantumKernel(feature_map=num_features, quantum_instance=quantum_instance)

qsvc = QSVC(quantum_kernel)

qsvc.fit(X_train, y_train)

y_pred = qsvc.predict(X_test)

score = qsvc.score(X_test, y_test)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    
    
y_train_array = y_train.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)
optimizer = COBYLA(maxiter=20)
sampler = Sampler()
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


# Define the grid of hyperparameters to search
ansatz_reps_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust as needed
feature_map_reps_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust as needed

best_accuracy = 0.0
best_ansatz_reps = None
best_feature_map_reps = None

for ansatz_reps in ansatz_reps_values:
    for feature_map_reps in feature_map_reps_values:
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=feature_map_reps)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=ansatz_reps)
        
        
        optimizer = COBYLA(maxiter=20)
        sampler = Sampler()
        
        vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=callback_graph,
        )
        
        vqc.fit(X_train, y_train_array)
        
        y_pred_train = vqc.predict(X_train)
        accuracy = np.mean(y_pred_train == y_train_array)
        
        print(f"Ansatz Reps: {ansatz_reps}, Feature Map Reps: {feature_map_reps}, Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_ansatz_reps = ansatz_reps
            best_feature_map_reps = feature_map_reps

print("Grid search finished.")
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Ansatz Reps: {best_ansatz_reps}")
print(f"Best Feature Map Reps: {best_feature_map_reps}")


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    
    
y_train_array = y_train.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)


ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

y_train_array = y_train.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

y_test_array = y_test.values
train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)
num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=5)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

precisions = []
recalls = []
f1_scores = []

for threshold in np.arange(0.1, 1.0, 0.1):
    y_pred_binary = [1 if prob > threshold else 0 for prob in y_pred]

    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Threshold: {threshold:.1f}")
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("F1 Score:", f1)
    print("---")

import matplotlib.pyplot as plt

thresholds = np.arange(0.1, 1.0, 0.1)
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1_scores, label="F1 Score")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()
****

num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)
optimizer = COBYLA(maxiter=20)
sampler = Sampler()



objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)


num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=2)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)

num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=4)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
# Fit the VQC classifier to the training data
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=6)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")


train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)


num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)


num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=10)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)




num_features = len(selected_featuresQ)
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=3)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=20)
sampler = Sampler()

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

    
y_train_array = y_train.values
y_test_array = y_test.values
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)
objective_func_vals = []
start = time.time()
vqc.fit(X_train, y_train_array)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")



y_test_array = y_test.values
train_score_q4 = vqc.score(X_train, y_train_array)
test_score_q4 = vqc.score(X_test, y_test_array)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")



from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = vqc.predict(X_test)

y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]

precision = precision_score(y_test, y_pred_binary)

recall = recall_score(y_test, y_pred_binary)

f1 = f1_score(y_test, y_pred_binary)

print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)

