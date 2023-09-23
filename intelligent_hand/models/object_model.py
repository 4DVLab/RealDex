import numpy
import trimesh


class ObjectModel():
    def __init__(self, mesh_path = None, verts = None, faces = None):
        if mesh_path is not None:
            self.mesh = trimesh.load_mesh(mesh_path)    
        else:
            try:
                self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            except ValueError:
                print("Object should be represented as a triangle mesh.")

        self.origin_center = self.mesh.center_mass
        self.verts = self.mesh.vertices - self.origin_center
        self.faces = self.mesh.faces

    def transform(self, rot, transl):
        verts = rot @ self.verts + self.origin_center + transl


