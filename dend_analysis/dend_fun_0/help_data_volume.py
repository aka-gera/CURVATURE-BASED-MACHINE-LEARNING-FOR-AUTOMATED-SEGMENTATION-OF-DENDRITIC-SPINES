




class vol_cropper:
    def __init__(self,vol=None, margin=4, multiple=16):
        self.margin = margin
        self.multiple = multiple
        self.crop_indices = None  
        self.vol=vol  

    def get_crop_indices(self, vol=None):
        """Return crop indices around non-zero region with a margin."""
        self.vol=vol = vol if vol is not None else self.vol
        coords = np.argwhere(vol != 0)

        if coords.size == 0: 
            self.crop_indices = (0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2])
            return self.crop_indices

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)

        # Expand bounding box
        zmin = max(zmin - self.margin, 0)
        ymin = max(ymin - self.margin, 0)
        xmin = max(xmin - self.margin, 0)

        zmax = min(zmax + self.margin + 1, vol.shape[0])
        ymax = min(ymax + self.margin + 1, vol.shape[1])
        xmax = min(xmax + self.margin + 1, vol.shape[2])

        self.crop_indices = (zmin, zmax, ymin, ymax, xmin, xmax)
        return self.crop_indices

    def pad_to_multiple(self, x):
        """Pad a 3D volume so each dimension is divisible by `multiple`."""
        D, H, W = x.shape

        pad_D = (self.multiple - D % self.multiple) % self.multiple
        pad_H = (self.multiple - H % self.multiple) % self.multiple
        pad_W = (self.multiple - W % self.multiple) % self.multiple

        return np.pad(
            x,
            ((0, pad_D), (0, pad_H), (0, pad_W)),
            mode='constant',
            constant_values=0
        )

    def crop_and_pad(self, vol=None, mask=None):
        """Crop and pad both volume and mask identically.""" 
        self.vol=vol = vol if vol is not None else self.vol
        crop_idx = self.get_crop_indices(vol)
        zmin, zmax, ymin, ymax, xmin, xmax = crop_idx

        # Crop
        vol_crop = vol[zmin:zmax, ymin:ymax, xmin:xmax]
        mask_crop = None
        if mask is not None:
            mask_crop = mask[zmin:zmax, ymin:ymax, xmin:xmax]

        # Pad
        vol_ready = self.pad_to_multiple(vol_crop)
        if mask is not None:
            mask_ready = self.pad_to_multiple(mask_crop)
            return vol_ready, mask_ready, crop_idx

        return vol_ready, crop_idx

    def remap_vertices(self, idx_original):
        """Map original vertex coordinates to cropped volume coordinates."""
        if self.crop_indices is None:
            raise ValueError("crop_indices not set. Run crop_and_pad() first.")

        zmin, zmax, ymin, ymax, xmin, xmax = self.crop_indices
        offset = np.array([zmin, ymin, xmin])
        return idx_original - offset





from scipy.ndimage import zoom

def resize_volume(volume, target_shape, order=1): 
    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    return zoom(volume, zoom=factors, order=order)
 




def reverse_resize_volume(volume_resized, original_shape, order=1):
    factors = (
        original_shape[0] / volume_resized.shape[0],
        original_shape[1] / volume_resized.shape[1],
        original_shape[2] / volume_resized.shape[2],
    )
    return zoom(volume_resized, zoom=factors,  order=order)


 
def bresenham_3d(p0, p1):
    p0 = np.array(p0, dtype=int)
    p1 = np.array(p1, dtype=int)

    points = []
    dx, dy, dz = np.abs(p1 - p0)
    sx = 1 if p0[0] < p1[0] else -1
    sy = 1 if p0[1] < p1[1] else -1
    sz = 1 if p0[2] < p1[2] else -1

    x, y, z = p0
    if dx >= dy and dx >= dz:
        yd = dy - dx // 2
        zd = dz - dx // 2
        while x != p1[0]:
            points.append((x, y, z))
            if yd >= 0:
                y += sy
                yd -= dx
            if zd >= 0:
                z += sz
                zd -= dx
            x += sx
            yd += dy
            zd += dz
    elif dy >= dx and dy >= dz:
        xd = dx - dy // 2
        zd = dz - dy // 2
        while y != p1[1]:
            points.append((x, y, z))
            if xd >= 0:
                x += sx
                xd -= dy
            if zd >= 0:
                z += sz
                zd -= dy
            y += sy
            xd += dx
            zd += dz
    else:
        xd = dx - dz // 2
        yd = dy - dz // 2
        while z != p1[2]:
            points.append((x, y, z))
            if xd >= 0:
                x += sx
                xd -= dz
            if yd >= 0:
                y += sy
                yd -= dz
            z += sz
            xd += dx
            yd += dy

    points.append(tuple(p1))
    return points



import numpy as np
import trimesh

class mesh_to_volume(vol_cropper): 
    def __init__(self, vertices, faces, margin=4, multiple=16):
        super().__init__(vol=None, margin=margin, multiple=multiple)

        self.vertices = vertices
 
        self.mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces)
        ) 
        edges = np.median(self.mesh.edges_unique_length) * 0.25
        print('mesh_to_volume-------edge----------', edges) 
        self.voxelized = self.mesh.voxelized(pitch=edges) 
        self.volume = self.voxelized.matrix.astype(np.uint8) 
 


    def get_volume(self, mesh_shaft_wrap=None,fill_method="orthographic"): 
        solid = self.voxelized.fill(method=fill_method)
        vol = solid.matrix.astype(np.uint8)
 
        if mesh_shaft_wrap is not None:

            # --- NEW: voxelize the shaft mesh instead of using vertex indices ---
            voxelized_shaft = mesh_shaft_wrap.voxelized(pitch=self.voxelized.pitch)
            volume_shaft = voxelized_shaft.fill(method=fill_method).matrix.astype(np.uint8)

            # Get origins from voxel transforms
            origin_main  = self.voxelized.transform[:3, 3]
            origin_shaft = voxelized_shaft.transform[:3, 3]
            pitch        = self.voxelized.pitch

            # Compute offset in voxel units
            offset = ((origin_shaft - origin_main) / pitch).round().astype(int)
            ox, oy, oz = offset

            sx, sy, sz = volume_shaft.shape

            # Clamp to avoid out-of-bounds
            x1 = max(0, ox); y1 = max(0, oy); z1 = max(0, oz)
            x2 = min(vol.shape[0], ox + sx)
            y2 = min(vol.shape[1], oy + sy)
            z2 = min(vol.shape[2], oz + sz)

            if x1 < x2 and y1 < y2 and z1 < z2:
                sx1 = x1 - ox; sy1 = y1 - oy; sz1 = z1 - oz
                sx2 = sx1 + (x2 - x1)
                sy2 = sy1 + (y2 - y1)
                sz2 = sz1 + (z2 - z1)

                # Paste shaft interior into main volume
                vol[x1:x2, y1:y2, z1:z2] = np.where(
                    volume_shaft[sx1:sx2, sy1:sy2, sz1:sz2] == 1,
                    2,
                    vol[x1:x2, y1:y2, z1:z2]
                )

        self.vol_shape = vol.shape
 
        return vol


 

    def get_volume_model(self,mesh_shaft_wrap=None): 

        vol=self.get_volume(mesh_shaft_wrap=mesh_shaft_wrap)
        # import tensorflow as tf
        # vol = tf.cast(vol, tf.float32)

        if len(vol.shape) == 3:
            vol = vol[None, ..., None]    
        elif len(vol.shape) == 4:
            vol = vol[..., None]   

        return vol 
    
 
    def map_vertices_to_original_volume(self):
        return np.array(self.voxelized.points_to_indices(self.mesh.vertices))




    def get_pred_rhs(self, model, vol=None,): 
        if vol is None:
            vol = self.get_volume_model() 
        idx_org = self.map_vertices_to_original_volume()
        if len(vol.shape) == 3:
            vol = vol[None, ..., None]    
        elif len(vol.shape) == 4:
            vol = vol[..., None]   
        predii_np = model(tf.cast(vol, tf.float32) ).numpy()    
        # c1 = predii_np[0,..., 0:1][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        c2 = predii_np[0,..., 1:2][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        c3 = predii_np[0,..., 2:3][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        return np.array([c2,c3]).T 


    def get_volume_crop(self,mesh_shaft_wrap=None,fill_method="orthographic"): 
        vol=self.get_volume(mesh_shaft_wrap=None,fill_method=fill_method )
        if mesh_shaft_wrap is None:
            vol, crop_indices=self.crop_and_pad(vol,) 
            idx_new = self.remap_vertices(self.map_vertices_to_original_volume() ) 
            return vol,idx_new
        else:
            mask=self.get_volume(mesh_shaft_wrap=mesh_shaft_wrap,fill_method=fill_method ) 
            vol,mask, _=self.crop_and_pad(vol,mask=mask) 
            idx_new = self.remap_vertices(self.map_vertices_to_original_volume() ) 
            return vol,mask,idx_new 
  



    def get_volume_model_crop(self,fill_method="orthographic"):   
        vol,idx_org=self.get_volume_crop(mesh_shaft_wrap=None,fill_method=fill_method ) 
        # import tensorflow as tf
        # vol = tf.cast(vol, tf.float32)

        if len(vol.shape) == 3:
            vol = vol[None, ..., None]    
        elif len(vol.shape) == 4:
            vol = vol[..., None]   

        return vol,idx_org
    

    def get_pred_rhs_crop(self, model, vol=None,idx_org=None, fill_method="orthographic",): 
        if vol is None:
            vol,idx_org=self.get_volume_crop(fill_method=fill_method )
        if len(vol.shape) == 3:
            vol = vol[None, ..., None]    
        elif len(vol.shape) == 4:
            vol = vol[..., None]   
        predii_np = model(tf.cast(vol, tf.float32) ).numpy()    
        # c1 = predii_np[0,..., 0:1][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        c2 = predii_np[0,..., 1:2][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        c3 = predii_np[0,..., 2:3][idx_org[:,0],idx_org[:,1],idx_org[:,2]].flatten()
        return np.array([c2,c3]).T 



























def paste_volume(vol, volume_shaft, voxelized_main, voxelized_shaft):
    """
    Paste a voxelized shaft volume into the main dendrite volume,
    correctly aligned using voxel grid transforms and safely clamped.
    """

    # --- 1. Extract origins from voxel transforms ---
    origin_main  = voxelized_main.transform[:3, 3]
    origin_shaft = voxelized_shaft.transform[:3, 3]
    pitch        = voxelized_main.pitch

    # --- 2. Compute offset in voxel units ---
    offset = ((origin_shaft - origin_main) / pitch).round().astype(int)
    ox, oy, oz = offset

    # --- 3. Shaft volume shape ---
    sx, sy, sz = volume_shaft.shape

    # --- 4. Clamp the paste region to avoid out-of-bounds ---
    x1 = max(0, ox)
    y1 = max(0, oy)
    z1 = max(0, oz)

    x2 = min(vol.shape[0], ox + sx)
    y2 = min(vol.shape[1], oy + sy)
    z2 = min(vol.shape[2], oz + sz)

    # If the region is invalid (no overlap), return unchanged
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return vol

    # --- 5. Compute corresponding region inside shaft volume ---
    sx1 = x1 - ox
    sy1 = y1 - oy
    sz1 = z1 - oz

    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    sz2 = sz1 + (z2 - z1)

    # --- 6. Paste shaft voxels (label 2) into main volume ---
    vol[x1:x2, y1:y2, z1:z2] = np.where(
        volume_shaft[sx1:sx2, sy1:sy2, sz1:sz2] == 1,
        2,
        vol[x1:x2, y1:y2, z1:z2]
    )

    return vol


def mesh_without_vertices(mesh, remove_idx):
    remove_idx = set(remove_idx)

    # Keep only faces whose vertices are NOT in remove_idx
    mask = ~np.isin(mesh.faces, list(remove_idx)).any(axis=1)

    new_faces = mesh.faces[mask]
    new_vertices = mesh.vertices.copy()

    # Build a mapping from old vertex index → new vertex index
    keep = np.ones(len(new_vertices), dtype=bool)
    keep[list(remove_idx)] = False

    old_to_new = -np.ones(len(new_vertices), dtype=int)
    old_to_new[keep] = np.arange(keep.sum())

    # Remap faces
    new_faces = old_to_new[new_faces]

    # Remove unused vertices
    new_vertices = new_vertices[keep]

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)



 
import plotly.graph_objects as go

def plot_voxel_volume(vol, tf_fig=False):
    # Remove batch/channel dimensions
    vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(f"Volume must be 3-D after squeeze, got shape {vol.shape}")

    # Extract voxel coordinates
    xs, ys, zs = np.where(vol > 0)
    vals = vol[vol > 0]

    # Color mapping
    color_map = {0: 'gray', 1: "blue", 2: "red"}
    colors = [color_map.get(v, "gray") for v in vals]

    fig = go.Figure() 
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=0.7),
        name='',
    ))

    # --- Compute bounding box ---
    D, H, W = vol.shape

    # Box corners
    corners = np.array([
        [0, 0, 0],
        [D, 0, 0],
        [D, H, 0],
        [0, H, 0],
        [0, 0, W],
        [D, 0, W],
        [D, H, W],
        [0, H, W]
    ])

    # Box edges (pairs of corner indices)
    edges = [
        (0,1), (1,2), (2,3), (3,0),   # bottom rectangle
        (4,5), (5,6), (6,7), (7,4),   # top rectangle
        (0,4), (1,5), (2,6), (3,7)    # vertical edges
    ]

    # --- Add box edges ---
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]],
            y=[corners[i,1], corners[j,1]],
            z=[corners[i,2], corners[j,2]],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))

    # --- Layout ---
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=900,
        height=900
    )

    if tf_fig:
        return fig
    else:
        fig.show()




def add_pca_axes_to_fig(fig, projector, colors=("red", "green", "blue"), width=8):
    """
    Add PCA axes to a Plotly 3D figure using true extremity points.
    PC1 = red, PC2 = green, PC3 = blue
    """
    for axis, color in enumerate(colors):
        p_min, p_max = projector.axis_extremes(axis=axis)

        fig.add_trace(go.Scatter3d(
            x=[p_min[0], p_max[0]],
            y=[p_min[1], p_max[1]],
            z=[p_min[2], p_max[2]],
            mode="lines+markers+text",
            line=dict(color=color, width=width),
            marker=dict(size=4, color=color),
            text=[f"PC{axis+1} min", f"PC{axis+1} max"],
            textposition="top center",
            showlegend=False
        ))
def add_pca_axes_from_volume(fig, projector, vol, 
                             colors=("red", "green", "blue"), width=8):
    """
    Add PCA axes to a Plotly 3D figure using extremity points
    computed from the voxel volume.
    """
    for axis, color in enumerate(colors):
        p_min, p_max = projector.axis_extremes_from_volume(vol, axis=axis)

        fig.add_trace(go.Scatter3d(
            x=[p_min[0], p_max[0]],
            y=[p_min[1], p_max[1]],
            z=[p_min[2], p_max[2]],
            mode="lines+markers+text",
            line=dict(color=color, width=width),
            marker=dict(size=4, color=color),
            text=[f"PC{axis+1} min", f"PC{axis+1} max"],
            textposition="top center",
            showlegend=False
        ))


import plotly.graph_objects as go
import numpy as np

def plot_voxel_volume_fill(vol, 
                           tf_fig=False,
                           color_map = {0: 'gray', 1: "blue", 2: "red"},
                           projector=None,
                           ):
    vol = np.squeeze(vol)

    if vol.ndim != 3:
        raise ValueError(f"Volume must be 3D after squeeze, got shape {vol.shape}")

    xs, ys, zs = np.where(vol > 0)
    vals = vol[vol > 0]

    
    colors = [color_map.get(v, "gray") for v in vals]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=0.7),
        name='',
    ))

    D, H, W = vol.shape

    corners = np.array([
        [0, 0, 0],
        [D, 0, 0],
        [D, H, 0],
        [0, H, 0],
        [0, 0, W],
        [D, 0, W],
        [D, H, W],
        [0, H, W]
    ])

    # --- Correct full cube mesh ---
    fig.add_trace(go.Mesh3d(
        x=corners[:,0],
        y=corners[:,1],
        z=corners[:,2],
        i=[0,0,4,4,0,0,3,3,0,0,1,1],
        j=[1,2,5,6,1,5,2,6,3,7,2,6],
        k=[2,3,6,7,5,4,6,7,7,4,6,5],
        color='lightgray',
        opacity=0.15,
        name='bounding_box',
        showscale=False
    ))

    # Wireframe edges
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]],
            y=[corners[i,1], corners[j,1]],
            z=[corners[i,2], corners[j,2]],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False,
            name='',
        ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=900,
        height=900,
        showlegend=False 
    )
    if projector is not None:
        add_pca_axes_from_volume(fig, projector, vol)


    if tf_fig:
        return fig
    else:
        fig.show()



def volume_from_projected_coords(coords_pca, values, shape=None, pad=5):
    """
    Build a 3D volume from PCA-projected voxel coordinates.
    coords_pca: (M,3)
    values: (M,)
    shape: optional (Dx,Hy,Wz). If None, auto-fit bounding box.
    """
    coords = np.asarray(coords_pca)

    # Compute bounding box in PCA space
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Shift coords to positive space
    shifted = coords - mins

    # Determine volume shape
    if shape is None:
        shape = np.ceil(maxs - mins).astype(int) + pad

    vol = np.zeros(shape, dtype=values.dtype)

    # Round coords to nearest voxel
    idx = np.round(shifted).astype(int)

    # Fill volume
    vol[idx[:,0], idx[:,1], idx[:,2]] = values

    return vol









import plotly.express as px

def plot_slice(vol, axis=0, index=0,tf_fig=False):
    vol = np.squeeze(vol)

    if axis == 0:
        slice_ = vol[index, :, :]
    elif axis == 1:
        slice_ = vol[:, index, :]
    else:
        slice_ = vol[:, :, index]

    fig = px.imshow(slice_, color_continuous_scale="Viridis")
    fig.update_layout(title=f"Slice {index} (axis {axis})")

    return fig if tf_fig else fig.show() 
       

import numpy as np
import plotly.graph_objects as go

def plot_slice_dui(vol, axis=0, index=0, tf_fig=False):
    vol = np.squeeze(vol)
    D, H, W = vol.shape

    # Determine number of slices along chosen axis
    if axis == 0:
        N = D
    elif axis == 1:
        N = H
    else:
        N = W

    fig = go.Figure()

    # --- Add one trace per slice ---
    for i in range(N):
        if axis == 0:
            sl = vol[i, :, :]
        elif axis == 1:
            sl = vol[:, i, :]
        else:
            sl = vol[:, :, i]

        # Add as Heatmap so Viridis works
        fig.add_trace(
            go.Heatmap(
                z=sl,
                colorscale="Viridis",
                visible=False,
                showscale=False
            )
        )

    # Make the initial slice visible
    fig.data[index].visible = True

    # --- Build slider steps ---
    steps = []
    for i in range(N):
        step = dict(
            method="update",
            args=[
                {"visible": [j == i for j in range(N)]},
                {"title": f"Slice {i} (axis {axis})"}
            ]
        )
        steps.append(step)

    # --- Add slider ---
    sliders = [dict(
        active=index,
        currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        dragmode="pan"
    )

    return fig if tf_fig else fig.show() 



from scipy.ndimage import affine_transform
import numpy as np
from sklearn.decomposition import PCA

import numpy as np 

class pca_projector:
    def __init__(self, vertices): 
        self.vertices =V = np.asarray(vertices) 
        self.center = V.mean(axis=0) 
        Vc = V - self.center 
        U, S, Vt = np.linalg.svd(Vc, full_matrices=False) 
        self.R = Vt
 
    def project(self, points):
        """
        Project new points into the PCA space.
        points: (N, 3)
        """
        P = np.asarray(points)
        Pc = P - self.center
        return Pc @ self.R.T
 

    def unproject(self, points_pca):
        """
        Transform PCA-space points back to original coordinates.
        """
        Pp = np.asarray(points_pca)
        return Pp @ self.R + self.center
 

    def project_volume_coords(self, vol):
        """
        Project all voxel coordinates of a volume into PCA space.
        Returns:
            coords_pca : (M, 3) PCA coords of non-zero voxels
            values     : voxel intensities
        """
        vol = np.asarray(vol)
        z, y, x = np.where(vol > 0)

        pts = np.vstack([x, y, z]).T
        pts_pca = self.project(pts)

        values = vol[z, y, x]
        return pts_pca, values

 
    def axis_extremes(self, axis=0, points=None):
        """
        Project points onto PCA axis and return the two extremity points.
        
        axis: 0, 1, or 2 (PC1, PC2, PC3)
        points: optional (N,3) array; if None, uses original vertices
        """
        if points is None:
            points = self.vertices

        # PCA axis direction (unit vector)
        axis_vec = self.R[axis]
        axis_vec = axis_vec / np.linalg.norm(axis_vec)

        # Center points
        P = np.asarray(points)
        Pc = P - self.center

        # Scalar projection of each point onto the axis
        scalars = Pc @ axis_vec   # shape (N,)

        # Find min and max projection values
        s_min = scalars.min()
        s_max = scalars.max()

        # Convert back to 3D points
        p_min = self.center + s_min * axis_vec
        p_max = self.center + s_max * axis_vec

        return p_min, p_max



    def axis_extremes_from_volume(self, vol, axis=0):
        """
        Compute PCA axis extremity points using voxel coordinates from the volume.
        """
        # Extract voxel coordinates
        z, y, x = np.where(vol > 0)
        pts = np.vstack([x, y, z]).T

        # PCA axis direction (unit vector)
        axis_vec = self.R[axis]
        axis_vec = axis_vec / np.linalg.norm(axis_vec)

        # Center points
        Pc = pts - self.center

        # Scalar projection of each voxel onto the axis
        scalars = Pc @ axis_vec

        # Min and max projection values
        s_min = scalars.min()
        s_max = scalars.max()

        # Convert back to 3D points
        p_min = self.center + s_min * axis_vec
        p_max = self.center + s_max * axis_vec

        return p_min, p_max


 

def plot_slice_dui_slider(vol, axis=0, index=0, tf_fig=False, base_width=900):
    vol = np.squeeze(vol)
    D, H, W = vol.shape

    # Determine number of slices and slice getter
    if axis == 0:
        N = D
        get_slice = lambda i: vol[i, :, :]
    elif axis == 1:
        N = H
        get_slice = lambda i: vol[:, i, :]
    else:
        N = W
        get_slice = lambda i: vol[:, :, i]

    fig = go.Figure()

    # --- Add one trace per slice ---
    for i in range(N):
        sl = get_slice(i)

        # Auto-transpose if height > width
        h, w = sl.shape
        if h > w:
            sl = sl.T
            h, w = sl.shape

        fig.add_trace(
            go.Heatmap(
                z=sl,
                colorscale="Viridis",
                visible=False,
                showscale=False
            )
        )

    # Make the initial slice visible
    fig.data[index].visible = True

    # --- Build slider steps ---
    steps = []
    for i in range(N):
        steps.append(dict(
            method="update",
            args=[
                {"visible": [j == i for j in range(N)]},
                {"title": f"Slice {i} (axis {axis})"}
            ]
        ))

    sliders = [dict(
        active=index,
        currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=steps
    )]

    # --- Determine final slice shape for layout ---
    sl0 = get_slice(0)
    h0, w0 = sl0.shape
    if h0 > w0:
        sl0 = sl0.T
        h0, w0 = sl0.shape

    # Compute dynamic height
    aspect = h / w if w != 0 else 1
    fig_height = int(base_width * aspect+1)
    print(aspect,fig_height,base_width)

    fig.update_layout(
        sliders=sliders,
        dragmode="pan",
        xaxis=dict(range=[0, w], constrain="domain"),
        yaxis=dict(range=[h, 0], scaleanchor="x"),
        # width=base_width,
        # height=fig_height
    )

    return fig if tf_fig else fig.show()


import numpy as np
import plotly.graph_objects as go

def plot_slice_dui(vol, axis=0, index=0, tf_fig=False, base_width=900):
    vol = np.squeeze(vol)
    D, H, W = vol.shape

    # Determine slice getter
    if axis == 0:
        get_slice = lambda i: vol[i, :, :]
    elif axis == 1:
        get_slice = lambda i: vol[:, i, :]
    else:
        get_slice = lambda i: vol[:, :, i]

    # --- Extract the slice ---
    sl = get_slice(index)

    # Auto‑transpose if height > width
    h, w = sl.shape
    if h > w:
        sl = sl.T
        h, w = sl.shape

    # Compute dynamic height
    aspect = h / w if w != 0 else 1
    fig_height = int(base_width * aspect)

    # --- Plot the slice ---
    fig = go.Figure(
        data=go.Heatmap(
            z=sl,
            colorscale="Viridis",
            showscale=False
        )
    )

    fig.update_layout(
        dragmode="pan",
        xaxis=dict(range=[0, w], constrain="domain"),
        yaxis=dict(range=[h, 0], scaleanchor="x"),
        # width=base_width,
        # height=fig_height,
        title=f"Slice {index} (axis {axis})"
    )

    return fig if tf_fig else fig.show() 

 


import tensorflow as tf
def get_report(mask):
    flat = tf.reshape(mask, [-1])    
    vals, idx = tf.unique(flat)
    mask=np.squeeze(mask)
    print("Total voxels:", mask.size)
    # print("Background (0):", np.sum(mask == 0))
    # print("Dendrite interior (1):", np.sum(mask == 1))
    # print("Shaft interior (2):", np.sum(mask == 2))
    vals=np.unique(np.round(1e2*vals.numpy())*1e-2)
    for ii in np.sort(vals):
        if ii>10:
            return
        po= 'Shaft interior (2):' if ii==2 else "Background (0):" if ii==0 else "Spines (1):" if ii==1 else f'vals {ii}'
        print(f" {po} ({ii}):", np.sum(mask == ii))
    tf.print("unique vals:", vals)


 
