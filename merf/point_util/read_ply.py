import open3d as o3d
import numpy as np
def get_center_and_scale(path):
    point_cloud = o3d.io.read_point_cloud(path)
    point = np.asarray(point_cloud.points)
    minx = np.inf
    miny = np.inf
    minz = np.inf
    maxx = -np.inf
    maxy = -np.inf
    maxz = -np.inf
    for i in range(len(point)):
        if(point[i][0] < minx ):
            minx = point[i][0]
        if (point[i][1] < miny):
            miny = point[i][1]
        if (point[i][2] < minz):
            minz = point[i][2]
        if (point[i][0] > maxx):
            maxx = point[i][0]
        if (point[i][1] > maxy):
            maxy = point[i][1]
        if (point[i][2] > maxz):
            maxz = point[i][2]

    length = maxx - minx
    width = maxy - miny
    height = maxz - minz
    print(f"length {length} width {width} height {height}")
    extend = 0.005
    extend_length = length * extend
    extend_width  = width * extend
    extend_height = height * extend

    print(f"original min {minx},{miny},{minz} max {maxx} {maxy} {maxz} ")
    minx = minx - extend_length
    maxx = maxx + extend_length
    miny = miny - extend_width
    maxy = maxy + extend_width
    minz = minz - extend_height
    maxz = maxz + extend_height
    print(f"changed_1 min {minx},{miny},{minz} max {maxx} {maxy} {maxz} ")

    center_x = (minx + maxx)/2
    center_y = (miny + maxy)/2
    center_z = (minz + maxz)/2
    print(f"center  {center_x},{center_y}, {center_z} ")

    minx -= center_x
    maxx -= center_x
    miny -= center_y
    maxy -= center_y
    minz -= center_z
    maxz -= center_z

    print(f"changed_2 min {minx},{miny},{minz} max {maxx} {maxy} {maxz} ")
    scale_x = max(abs(minx),abs(maxx))
    scale_y = max(abs(miny),abs(maxy))
    scale_z = max(abs(minz),abs(maxz))

    scale = max(scale_x,max(scale_y,scale_z))
    print(f" scene_scale_t = {scale}")
    return np.array([center_x,center_y,center_z]), scale, maxz, minz
