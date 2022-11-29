"""
@author: Pierre Gibertini

Machine learning model for interference detection between 3D objects

2. MODEL TRAINING
"""
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from utils import scale_data


def main():
    # LOADING DATA
    data = []

    with open("data/data_10000_5000.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            data.append((list(map(float, row))))

    print(len(data))

    # SEPARATING X AND Y
    inputs = [row[0:-1] for row in data]
    outputs = [row[-1] for row in data]

    # SCALING DATA
    scaled_data, scaler_input = scale_data(data_matrix=inputs)
    dump(scaler_input, "scaler/scaler.joblib")  # saving scaler
    # scaled_data = inputs

    X_train, X_test, Y_train, Y_test = train_test_split(
        scaled_data, outputs, train_size=int(len(outputs) * 0.8), shuffle=True
    )

    # MLP TRAIN
    regressor = MLPRegressor(
        hidden_layer_sizes=(105, 37, 27),
        activation="relu",
        solver="adam",
        alpha=0.00001,
        learning_rate="adaptive",
        verbose=True,
        max_iter=2000,
        tol=1e-6,
    )
    regressor.fit(X_train, Y_train)

    # MLP TEST
    print(f"neural network r²: {regressor.score(X_test, Y_test)}")
    dump(regressor, "model/MLP.joblib")  # saving model

    # RF TRAIN
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, Y_train)

    # RF TEST
    print(f"random forest r²: {regressor.score(X_test, Y_test)}")
    dump(regressor, "model/RF.joblib")  # saving model


if __name__ == "__main__":
    main()
