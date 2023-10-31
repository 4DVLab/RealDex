import plotly.graph_objects as go
import glob
import trimesh
import io
import PIL.Image

def load_mesh(path):
    mesh = trimesh.load_mesh(path)
    verts = mesh.vertices
    faces = mesh.faces
    mesh_plotly = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1
    )
    return mesh_plotly

def get_frames():
    data_path = "/Users/yumeng/Downloads/arm_hand_mesh_ply_simp/"
    file_list = list(glob.glob(data_path + '/*')) #file path
    file_list = sorted(file_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    file_list = file_list[:60]
    mesh_list = [load_mesh(path) for path in file_list]
    frames = []
    for mesh in mesh_list:
        frame = go.Frame(data=mesh)
        frames.append(frame)
    return frames, mesh_list


def show_interactive(frames, mesh_list):
    fig = go.Figure()

    fig.add_trace(mesh_list[0])
    fig.update(frames=frames)

    animation_duration = 50  #(ms)
    fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': animation_duration, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}])])])

    fig.show()
    return fig
    
def export_gif(fig):
    img_frames = []
    for slider_pos, frame in enumerate(fig.frames):
        fig.update(data=frame.data)
        img_frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png"))))
        
    # Create the gif file.
    img_frames[0].save("./out",
                save_all=True,
                append_images=img_frames[1:],
                optimize=True,
                duration=50,
                loop=0)
    
if __name__ == "__main__":
    frames, meshes = get_frames()
    fig = show_interactive(frames, meshes)
    export_gif(fig)