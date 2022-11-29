"""
@author: Pierre Gibertini

Machine learning model for interference detection between 3D objects

1. DATA GENERATION
"""
import math
import multiprocessing
import time
import csv
import numpy
import volmdlr
import volmdlr.core
import matplotlib.pyplot as plt
from typing import List
from volmdlr.primitives3d import Cylinder
from scipy.stats import qmc
from tqdm import tqdm
from utils import relative_pos_cyl, random_3d_vector

# GENERATION PARAMS
SIZE_SAMPLE = 10000
N_POINTS_VOLUME = 5000
GEN_COEFF = 0.65  # this value is used to define the size of the generating space. chose a value to have ~50% of y=0
USE_MULTIPROCESSING = True  # may not work with iPython

# BASE OBJECT
RADIUS = 0.05
LENGTH = 0.1
obj = Cylinder(
    position=volmdlr.O3D,
    axis=volmdlr.X3D,
    radius=RADIUS,
    length=LENGTH,
    color=(1, 0, 0),
)
cyl0 = obj
VOLUME = obj.volume()
bounding_box = obj.bounding_box
R_GEN = (
    GEN_COEFF
    * math.sqrt(3)
    * max(
        abs(bounding_box.xmax - bounding_box.xmin),
        abs(bounding_box.ymax - bounding_box.ymin),
        abs(bounding_box.zmax - bounding_box.zmin),
    )
)


def draw_to_point(draw: List[float]) -> volmdlr.Point3D:
    """
    :param draw: draw done in LHS
    :return: random point in sphere
    """
    u = draw[0]
    v = draw[1]
    r = numpy.cbrt(draw[2]) * R_GEN

    theta = u * 2.0 * math.pi
    phi = math.acos(2.0 * v - 1.0)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return volmdlr.Point3D(x, y, z)


def generate_data_uniform(draw: List[float]) -> List[float]:
    """
    :param draw: draw done in LHS
    :return: data X, Y for training
    """
    cyl1 = Cylinder(
        position=draw_to_point(draw),
        axis=random_3d_vector(),
        radius=RADIUS,
        length=LENGTH,
        color=(0, 1, 0),
    )

    return relative_pos_cyl(cyl0, cyl1) + [
        cyl0.interference_volume_with_other_cylinder(cyl1, n_points=N_POINTS_VOLUME)
        / VOLUME
    ]


def main():
    start = time.perf_counter()
    # GENERATING CYL
    sampler = qmc.LatinHypercube(d=3, seed=0)

    # GENERATION BOUNDARIES FROM BOUNDING BOX
    lb = [0, 0, 0]
    ub = [1, 1, 1]

    sample = qmc.scale(
        sampler.random(n=SIZE_SAMPLE),
        lb,
        ub,
    )

    # GENERATING DATA
    if USE_MULTIPROCESSING:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        data = list(
            tqdm(pool.imap_unordered(generate_data_uniform, sample), total=SIZE_SAMPLE)
        )

    else:
        data = []

        for pos in tqdm(sample):
            data.append(generate_data_uniform(pos))

    # DELETING Y=0
    # data = [row for row in data if row[-1] != 0]

    print(
        f"y=0: {100 * (len([x for x in [row[-1] for row in data] if x == 0]) / SIZE_SAMPLE)}%"
    )

    # SAVING DATA
    with open(f"data/data_{SIZE_SAMPLE}_{N_POINTS_VOLUME}.csv", "w") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(data)

    print(f"file written : data_{SIZE_SAMPLE}_{N_POINTS_VOLUME}.csv")
    print(f"time taken: {time.perf_counter() - start}s")

    # PLOTTING Y HISTOGRAM
    y = [row[-1] for row in data]
    _ = plt.hist(y, bins="auto")
    plt.title("Y repartition")
    plt.show()


if __name__ == "__main__":
    main()
