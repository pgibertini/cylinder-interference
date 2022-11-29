"""
@author: Pierre Gibertini

Machine learning model for interference detection between 3D objects

4. MODEL VALIDATION
"""
import csv
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import scale_data


def main():
    # LOADING DATA
    data = []

    with open("data/data_10000_5000.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|")
        for row in reader:
            data.append((list(map(float, row))))

    # data = [row for row in data if row[-1] != 0]
    print(len(data))

    # SEPARATING X AND Y
    inputs = [row[0:-1] for row in data]
    outputs = [row[-1] for row in data]

    # SCALING DATA
    scaled_data, _ = scale_data(data_matrix=inputs)

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
    # MLP CROSS VALIDATION
    scores = cross_val_score(regressor, scaled_data, outputs, cv=10)
    print(f"neural network cross validation r²: {scores}")

    # RF TRAIN
    regressor = RandomForestRegressor(n_estimators=100)

    # RF CROSS VALIDATION
    scores = cross_val_score(regressor, scaled_data, outputs, cv=10)
    print(f"random forest cross validation r²: {scores}")


if __name__ == "__main__":
    main()
