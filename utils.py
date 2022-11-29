"""
@author: Pierre Gibertini
@date: November 2022

Useful functions
"""
from volmdlr.primitives3d import Cylinder
import sklearn.preprocessing as preprocessing
from random import gauss
from typing import List
import volmdlr
import volmdlr.core
import math


def relative_pos_cyl(cylinder_0: Cylinder, cylinder_1: Cylinder) -> List[float]:
    """
    :param cylinder_0: volmdlr Cylinder
    :param cylinder_1: volmdlr Cylinder
    :return: relative rotative and translative position
    """
    # PLACING CYLINDERS AS FOR TRAINING (cylinder_0 at global origin)
    vec_to_origin, rot_axis, rot_angle = movement_to_point_and_axis(
        cylinder_0, volmdlr.O3D, volmdlr.X3D
    )

    cylinder_1 = cylinder_1.translation(vec_to_origin)

    # The local frame may be oriented differently from the global frame
    if rot_axis is not None:
        cylinder_1 = cylinder_1.rotation(volmdlr.O3D, rot_axis, rot_angle)

    frame_1 = volmdlr.Frame3D.from_point_and_vector(
        cylinder_1.position, cylinder_1.axis
    )

    transition_matrix = frame_1.transfer_matrix()
    relative_position = frame_1.origin

    return [
        transition_matrix.M11,
        transition_matrix.M12,
        transition_matrix.M13,
        transition_matrix.M22,
        transition_matrix.M23,
        transition_matrix.M33,
        relative_position[0],
        relative_position[1],
        relative_position[2],
    ]


def movement_to_point_and_axis(obj, point: volmdlr.Point3D, axis: volmdlr.Vector3D):
    translation_vector = point - obj.position
    rotation_angle = None
    rotation_axis = None

    if obj.axis != axis:
        # Rotation angle
        dot = obj.axis.dot(axis)
        rotation_angle = math.acos(dot / (obj.axis.norm() * axis.norm()))

        # Rotation axis
        vector2 = axis - obj.axis
        rotation_axis = obj.axis.cross(vector2)
        rotation_axis.normalize()

    return translation_vector, rotation_axis, rotation_angle


def scale_data(data_matrix: List[List[float]]):
    """
    :param data_matrix: the data to scale
    :return: scaled matrix, scaler
    """
    _scaler = preprocessing.StandardScaler()
    _scaled_matrix = _scaler.fit_transform(data_matrix)
    return _scaled_matrix, _scaler


def make_rand_vector(dims: int) -> List[float]:
    """
    :param dims: dimension of desired vector
    :return: random vector of desired dim
    """
    vec = [gauss(0, 1) for _ in range(dims)]
    mag = sum(x**2 for x in vec) ** 0.5
    return [x / mag for x in vec]


def random_3d_vector():
    """
    :return: random volmdlr 3d vector
    """
    random_vector = make_rand_vector(3)
    return volmdlr.Vector3D(random_vector[0], random_vector[1], random_vector[2])
