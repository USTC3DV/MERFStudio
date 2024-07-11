import plotly.graph_objects as go
import numpy as np
import plotly.offline as pyo

def visualize_pointcloud_and_cameras(points, camera_positions):
    """
    Visualize a point cloud and multiple camera positions in 3D using Plotly.

    Parameters:
    - points (np.ndarray): A (N, 3) array representing the point cloud.
    - camera_positions (np.ndarray): A (M, 3) array representing the positions of M cameras.
    """
    # Create scatter plot for point cloud
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=3, opacity=0.8),
        name="Points"
    )

    # Create camera position markers
    cameras = go.Scatter3d(
        x=camera_positions[:, 0],
        y=camera_positions[:, 1],
        z=camera_positions[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name="Cameras"
    )

    # Plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(nticks=10, range=[-30,30]),
            yaxis=dict(nticks=10, range=[-30,30]),
            zaxis=dict(nticks=10, range=[-30,30]),
            aspectmode='cube'
        )
    )
    fig = go.Figure(data=[scatter, cameras], layout=layout)
    pyo.plot(fig, filename='outputs/camera_and_points.html', auto_open=True)
    fig.write_html("/gpfs/home/juyonggroup/kevin2000/outputs/pointcloud.html")
    #fig.show()
    

def visualize_camera_poses_plotly(poses,scale=0.1):
    """
    Visualizes camera poses in 3D space using pyramids (cones) with Plotly.

    Parameters:
        poses (numpy array): A Nx4x4 numpy array, where N is the number of poses.
        scale (float): Scaling factor for the pyramid size.
    """
    # Define a basic pyramid in camera coordinates
    pyramid_tip = np.array([0, 0, 0, 1])
    pyramid_base = np.array([[0.2, 0.2, -0.5, 1], 
                             [-0.2, 0.2, -0.5, 1], 
                             [-0.2, -0.2, -0.5, 1], 
                             [0.2, -0.2, -0.5, 1]])
    # Scale only the spatial parts of the pyramid base vertices
    pyramid_base[:, :3] *= scale
    
    
    # Create a 3D figure
    fig = go.Figure()

    # Plot each camera pose
    for i in range(len(poses)):
        if i % 3 == 0:
            # Apply the pose transformation to the pyramid vertices
            transformed_tip = np.dot(poses[i], pyramid_tip)[:3]
            transformed_base = np.dot(poses[i], pyramid_base.T).T[:, :3]

            # Add pyramid base and sides to the plot
            fig.add_trace(go.Mesh3d(x=[*transformed_base[:, 0], transformed_tip[0]], 
                                    y=[*transformed_base[:, 1], transformed_tip[1]], 
                                    z=[*transformed_base[:, 2], transformed_tip[2]], 
                                    i=[0, 0, 0, 1, 1, 2], 
                                    j=[1, 3, 2, 2, 3, 3], 
                                    k=[3, 1, 1, 0, 0, 0],
                                    opacity=0.5))

    # Labels and title
    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='cube',
                        xaxis=dict(range=[-5, 5]),
                        yaxis=dict(range=[-5, 5]),
                        zaxis=dict(range=[-5, 5])),
                      title='Camera Poses Visualization')

    # Show the plot
    fig.write_html("/gpfs/home/juyonggroup/kevin2000/outputs/pose.html")
    #fig.show()
