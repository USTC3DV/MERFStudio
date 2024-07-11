
from sklearn.linear_model import RANSACRegressor

import  open3d as o3d
import  numpy as np

def get_normal(path):
   
    ransac = RANSACRegressor()

    point_cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(point_cloud.points)


    ransac.fit(points[:, :2], points[:, 2])

    plane_params = ransac.estimator_.coef_
    plane_params = np.append(plane_params, ransac.estimator_.intercept_)

  
    print("平面模型参数：", plane_params)
    return plane_params[0],plane_params[1],plane_params[2] # A,B,t