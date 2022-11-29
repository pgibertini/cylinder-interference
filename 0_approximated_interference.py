"""
@author: Pierre Gibertini

Machine learning model for interference detection between 3D objects

0. APPROXIMATED INTERSECTION
"""
import volmdlr
from volmdlr.core import VolumeModel
from volmdlr.primitives3d import Cylinder

# CYLINDERS PARAMS
RADIUS = 0.05
LENGTH = 0.1


def main():
    cyl1 = Cylinder(
        position=volmdlr.O3D,
        axis=volmdlr.X3D,
        radius=RADIUS,
        length=LENGTH,
        color=(1, 0, 0),
    )
    cyl2 = Cylinder(
        position=volmdlr.Point3D(0.01, 0.05, 0.005),
        axis=volmdlr.Vector3D(1, 1, 1),
        radius=RADIUS,
        length=LENGTH,
        color=(0, 0, 1),
    )

    VolumeModel([cyl1, cyl2]).babylonjs()

    ax = volmdlr.O3D.plot()

    points_cyl1 = [p for p in cyl1.lhs_points_inside(1000) if not cyl2.point_belongs(p)]
    points_cyl2 = [p for p in cyl2.lhs_points_inside(1000) if not cyl1.point_belongs(p)]
    points_inter = [p for p in cyl1.lhs_points_inside(1000) if cyl2.point_belongs(p)]

    for p in points_cyl1:
        p.plot(ax=ax, color="red")

    for p in points_cyl2:
        p.plot(ax=ax, color="blue")

    for p in points_inter:
        p.plot(ax=ax)

    print(f"% of intersection : {(len(points_inter) / len(points_cyl1)) * 100}%")


if __name__ == "__main__":
    main()
