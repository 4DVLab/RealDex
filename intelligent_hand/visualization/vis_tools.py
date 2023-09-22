import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
# from urdfpy import URDF

class RobotViewer():
    def __init__(self, robo_model):
        self.robo_model = robo_model
    
    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_contact_points=False, pose=None):
        """
        Borrowed from https://github.com/PKU-EPIC/DexGraspNet.git
        Author: Jialiang Zhang, Ruicheng Wang
        Class: HandModel
        
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.robo_model.mesh:
            v = self.robo_model.current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.robo_model.global_rotation[i].T + self.robo_model.global_translation[i]
            f = self.mesh[link_name]['faces']
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity))
        
        return data
