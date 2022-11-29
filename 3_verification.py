"""
@author: Pierre Gibertini

Machine learning model for interference detection between 3D objects

3. VERIFICATION AND PERFORMANCE
"""
import time
import volmdlr
from volmdlr.core import VolumeModel
from itertools import permutations
from volmdlr.primitives3d import Cylinder
from utils import relative_pos_cyl, random_3d_vector
from joblib import load

# CYLINDERS PARAMS
RADIUS = 0.05
LENGTH = 0.1


cylinders = [
    Cylinder(
        position=volmdlr.O3D,
        axis=volmdlr.X3D,
        radius=RADIUS,
        length=LENGTH,
        color=(1, 0, 0),
    ),
    Cylinder(
        position=volmdlr.O3D,
        axis=volmdlr.X3D,
        radius=RADIUS,
        length=LENGTH,
        color=(1, 0, 0),
    ),
    Cylinder(
        position=volmdlr.Point3D(0.01, 0.05, 0.005),
        axis=volmdlr.Vector3D(1, 1, 1),
        radius=RADIUS,
        length=LENGTH,
        color=(0, 1, 0),
    ),
    Cylinder(
        position=volmdlr.Point3D(0.11, 0.05, 0.005),
        axis=random_3d_vector(),
        radius=RADIUS,
        length=LENGTH,
        color=(0, 0, 1),
    ),
    Cylinder(
        position=volmdlr.Point3D(-0.002, 0.01, -0.0345),
        axis=random_3d_vector(),
        radius=RADIUS,
        length=LENGTH,
        color=(127 / 255, 0, 1),
    ),
]

# 3D REPRESENTATION
vm = VolumeModel(cylinders)
vm.babylonjs()

# LOADING REGRESSORS AND SCALER
regressor_RF = load("model/RF.joblib")
regressor_MLP = load("model/MLP.joblib")
scaler = load("scaler/scaler.joblib")

for (cyl0, color0), (cyl1, color1) in permutations(
    zip(cylinders, ["red", "red (clone)", "green", "blue", "purple"]), 2
):
    # COMPUTING INPUT
    x = scaler.transform([relative_pos_cyl(cyl0, cyl1)])

    VOLUME = cyl0.volume()

    # ESTIMATION AND PREDICTION
    print(f"\n{color0} and {color1}")
    print(
        f"Estimation: {(cyl0.interference_volume_with_other_cylinder(cyl1, n_points=10000) / VOLUME) * 100}%"
    )
    print(f"RF prediction: {regressor_RF.predict(x)[0] * 100}%")
    print(f"MLP prediction: {regressor_MLP.predict(x)[0] * 100}%")

# PERFORMANCE
print("\nPERFORMANCE TEST")
N_TEST = 100

start = time.perf_counter()
for _ in range(N_TEST):
    _ = cyl0.interference_volume_with_other_cylinder(cyl1, n_points=10000) / VOLUME
print(f"Classic, time per calculus: {(time.perf_counter() - start) * 1000 / N_TEST} ms")

start = time.perf_counter()
for _ in range(N_TEST):
    _ = regressor_MLP.predict(x)[0]
print(f"MLP, time per calculus: {(time.perf_counter() - start) * 1000 / N_TEST} ms")

start = time.perf_counter()
for _ in range(N_TEST):
    _ = regressor_RF.predict(x)[0]
print(f"RF, time per calculus: {(time.perf_counter() - start) * 1000 / N_TEST} ms")
