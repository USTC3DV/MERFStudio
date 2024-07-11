
import numpy as np
import open3d as o3d


def initialize_points(path, rotation_matrix):
    point_cloud = o3d.io.read_point_cloud(path)

  
    points = np.asarray(point_cloud.points)
    pad = np.ones(shape=(*points.shape[:-1], 1), dtype=np.float32)

    #colmap to nerf
    applied_transform = np.eye(4)
    applied_transform = applied_transform[np.array([1, 0, 2, 3]), :]
    applied_transform[2, :] *= -1
    points = np.dot(points,applied_transform.T)
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points = rotated_points[..., :-1]
    
    rotated_point_cloud = o3d.geometry.PointCloud()
    rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)


    o3d.io.write_point_cloud(path, rotated_point_cloud)
    return rotated_points

def get_rotation_matrix(A, B, t, in_path_ply, out_path_ply):
    n = np.array([-A, -B, 1.]) 
    z = np.array([0, 0, 1.])
    # print(n*z)
    cos_theta = 1. / (np.sqrt(A ** 2 + B ** 2 + 1))
    theta = np.arccos(cos_theta)
    
    rotation_axis = np.cross(n, z)
    rotation_matrix_ = rotation_matrix(rotation_axis, 2 * np.pi - cos_theta, in_path_ply, out_path_ply)

    return rotation_matrix_


def rotation_matrix(axis, angle, in_path_ply, out_path_ply):

    axis = axis / np.linalg.norm(axis) 
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)

    rotation_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c), 0.],
                                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b), 0.],
                                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c, 0.],
                                [0., 0., 0., 1.]])
    print("R ", rotation_matrix)
   
    point_cloud = o3d.io.read_point_cloud(in_path_ply)

    points = np.asarray(point_cloud.points)

    pad = np.ones(shape=(*points.shape[:-1], 1), dtype=float)
    points = np.concatenate([points, pad], axis=-1)
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points = rotated_points[..., :-1]
  
    rotated_point_cloud = o3d.geometry.PointCloud()
    rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)


    o3d.io.write_point_cloud(out_path_ply, rotated_point_cloud)

    return rotation_matrix
