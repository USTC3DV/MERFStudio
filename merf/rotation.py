# 作者 任晨曲
# 作者 任晨曲
import numpy as np
import open3d as o3d


def initialize_points(path, rotation_matrix):
    point_cloud = o3d.io.read_point_cloud(path)

    # # 将点云数据转换为NumPy数组
    points = np.asarray(point_cloud.points)
    pad = np.ones(shape=(*points.shape[:-1], 1), dtype=np.float32)
    points = np.concatenate([points, pad], axis=-1)
    # # 对点云进行旋转
    #colmap to nerf
    applied_transform = np.eye(4)
    applied_transform = applied_transform[np.array([1, 0, 2, 3]), :]
    applied_transform[2, :] *= -1
    points = np.dot(points,applied_transform.T)
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points = rotated_points[..., :-1]
    # # 创建旋转后的点云对象
    rotated_point_cloud = o3d.geometry.PointCloud()
    rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    # # 将旋转后的点云写入PLY文件
    o3d.io.write_point_cloud(path, rotated_point_cloud)
    return rotated_points

def initialize_points_1(points, rotation_matrix):
    # point_cloud = o3d.io.read_point_cloud(path)

    # # 将点云数据转换为NumPy数组
    points = np.asarray(points)
    pad = np.ones(shape=(*points.shape[:-1], 1), dtype=np.float32)
    points = np.concatenate([points, pad], axis=-1)
    # # 对点云进行旋转
    #colmap to nerf
    applied_transform = np.eye(4)
    applied_transform = applied_transform[np.array([1, 0, 2, 3]), :]
    applied_transform[2, :] *= -1
    points = np.dot(points,applied_transform.T)
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points = rotated_points[..., :-1]
    # # # 创建旋转后的点云对象
    # rotated_point_cloud = o3d.geometry.PointCloud()
    # rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    # # 将旋转后的点云写入PLY文件
    # o3d.io.write_point_cloud(path, rotated_point_cloud)
    return rotated_points

def get_rotation_matrix(A, B, t, in_path_ply, out_path_ply):
    n = np.array([-A, -B, 1.])  # 面的法向量
    z = np.array([0, 0, 1.])
    # print(n*z)
    cos_theta = 1. / (np.sqrt(A ** 2 + B ** 2 + 1))
    theta = np.arccos(cos_theta)
    print("旋转角  ", theta)
    rotation_axis = np.cross(n, z)
    rotation_matrix_ = rotation_matrix(rotation_axis, 2 * np.pi - cos_theta, in_path_ply, out_path_ply)

    return rotation_matrix_


def rotation_matrix(axis, angle, in_path_ply, out_path_ply):
    """
    计算绕给定轴旋转指定角度的旋转矩阵
    :param axis: 旋转轴，三维向量
    :param angle: 旋转角度（弧度）
    :return: 旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴向量
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)

    rotation_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c), 0.],
                                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b), 0.],
                                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c, 0.],
                                [0., 0., 0., 1.]])
    print("R ", rotation_matrix)
    # # 读取PLY文件
    point_cloud = o3d.io.read_point_cloud(in_path_ply)

    # # 将点云数据转换为NumPy数组
    points = np.asarray(point_cloud.points)
    # # 对点云进行旋转
    pad = np.ones(shape=(*points.shape[:-1], 1), dtype=float)
    points = np.concatenate([points, pad], axis=-1)
    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points = rotated_points[..., :-1]
    # # 创建旋转后的点云对象
    rotated_point_cloud = o3d.geometry.PointCloud()
    rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

    # # 将旋转后的点云写入PLY文件
    o3d.io.write_point_cloud(out_path_ply, rotated_point_cloud)

    return rotation_matrix

# import open3d as o3d
# # import numpy as np

# # 读取PLY文件
# point_cloud = o3d.io.read_point_cloud('./data/output.ply')

# # 将点云数据转换为NumPy数组
# points = np.asarray(point_cloud.points)


# # 计算旋转矩阵
# rotation_matrix = rotation_matrix(rotation_axis, 2*np.pi - cos_theta)

# print("R ",rotation_matrix)
# # 对点云进行旋转
# rotated_points = np.dot(points, rotation_matrix.T)

# # 创建旋转后的点云对象
# rotated_point_cloud = o3d.geometry.PointCloud()
# rotated_point_cloud.points = o3d.utility.Vector3dVector(rotated_points)

# # 将旋转后的点云写入PLY文件
# o3d.io.write_point_cloud('./data/rotation_output_2.ply', rotated_point_cloud)
