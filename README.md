# Experiment-6---Heart-attack-prediction-using-MLP
## Aim:
      To construct a  Multi-Layer Perceptron to predict heart attack using Python
## Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<br>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<br>

## Program:



## Output:

## Result:
     Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     

