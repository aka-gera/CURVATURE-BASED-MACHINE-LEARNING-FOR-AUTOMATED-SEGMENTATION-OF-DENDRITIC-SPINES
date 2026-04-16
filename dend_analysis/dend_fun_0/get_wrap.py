
import os
import numpy as np

 
import trimesh 
from dend_fun_0.obj_get import  Obj_to_coord




def get_wrap(self,
                    dend_path_entry,
                    dend_path_exit,
                    dend_first_name=None,
                    spine_path=None,
                    shaft_path=None,
                    file_path=None, 
                    radius_threshold=None, 
                    disp_infos=None,
                    size_threshold=None,
                    alpha_fraction: float =0.001, 
                    offset_fraction: float =0.0001,
                    dend_name=None,
                    new_data=True,
                    ):
    import  pymeshlab 
    disp_infos=disp_infos or self.disp_infos
    file_path  = file_path or self.file_path
    spine_path = spine_path or self.spine_path
    shaft_path = shaft_path or self.shaft_path   
    radius_threshold = radius_threshold or self.radius_threshold
    size_threshold=size_threshold or self.size_threshold
    dend_first_name=dend_first_name or self.dend_first_name  
    if disp_infos: 
        print(f"Wrap path dend_path_original_m: {dend_path_entry}")  
    vertices_00,faces=Obj_to_coord(file_path_original=dend_path_entry,)  
 

    ms = pymeshlab.MeshSet()  
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices_00, face_matrix=faces) ) 
    
    ms.apply_filter(
        'generate_alpha_wrap', 
    alpha=pymeshlab.PercentageValue(alpha_fraction),
    offset=pymeshlab.PercentageValue(offset_fraction)
    )

    # ms.apply_filter('generate_alpha_wrap' )   
    mesh_shaft_wrap = ms.current_mesh()
    vertices_shaft_wrap = mesh_shaft_wrap.vertex_matrix()
    faces_shaft_wrap= mesh_shaft_wrap.face_matrix()
    mesh_shaft_wrap = trimesh.Trimesh(vertices=vertices_shaft_wrap,faces=faces_shaft_wrap)
    mesh_shaft_wrap.export(os.path.join(dend_path_exit )) 

def get_wrap_full(mesh_path_name_entry=None,
                  mesh_path_name_exit=None,
                  mesh=None,
                  alpha_fraction: float =0.001, 
                  offset_fraction: float =0.0001,): 
    if mesh_path_name_entry is not None:
        vertices_00,faces=Obj_to_coord(file_path_original=mesh_path_name_entry,) 
    elif mesh is not None:
        vertices_00,faces=mesh.vertices,mesh.faces
    else:
        raise('No mesh detected')

    import  pymeshlab 
 

    ms = pymeshlab.MeshSet()  
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices_00, face_matrix=faces) ) 

    ms.apply_filter(
        'generate_alpha_wrap', 
    alpha=pymeshlab.PercentageValue(alpha_fraction),
    offset=pymeshlab.PercentageValue(offset_fraction)
    )

    # ms.apply_filter('generate_alpha_wrap' )   
    mesh_shaft_wrap = ms.current_mesh()
    vertices_shaft_wrap = mesh_shaft_wrap.vertex_matrix()
    faces_shaft_wrap= mesh_shaft_wrap.face_matrix()
    mesh_shaft_wrap = trimesh.Trimesh(vertices=vertices_shaft_wrap,faces=faces_shaft_wrap)
    if mesh_path_name_exit is not None:
        mesh_shaft_wrap.export(mesh_path_name_exit)
    else:
        return mesh_path_name_exit
    

  
def get_wrap_o3d(vertices, faces, number_of_points=8000, radius=0.9, max_nn=30):
    # Build trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    import open3d as o3d
    # Convert to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Make sure we don't sample more points than vertices (optional but safer)
    number_of_points = int(min(number_of_points, len(mesh.vertices)))

    # Sample points using Poisson disk sampling
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=number_of_points)

    # Estimate and orient normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    # Poisson reconstruction (returns mesh and densities)
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=12,
        width=0,
        scale=1.1,
        linear_fit=False,
        n_threads=1,
    )
 
    return np.asarray(mesh_poisson.vertices), np.asarray(mesh_poisson.triangles)




import numpy as np
import trimesh
import open3d as o3d 

def get_alpha_wrap(vertices, faces, 
                    voxel_resolution=128, 
                    smooth_iterations=5, 
                    number_of_points=2000, 
                    radius=0.9, 
                    max_nn=30, 
                    sdf_trunc=3.0, 
                    alpha=0.02, 
                    target_triangles=8000, 
                    alpha_fraction=1.02,
                    offset_fraction=1.00):
    from dend_fun_2.help_pinn_data_fun import mesh_resize
    import  pymeshlab 

    # Build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
 

    target_number_of_triangles=min(2*vertices.shape[0], 2*target_triangles) 
    skl=mesh_resize(vertices=vertices,
                            faces=faces,
                            target_number_of_triangles=target_number_of_triangles ,)
    vertices=skl.mesh.vertices
    faces=skl.mesh.faces
    print('[[[[[[[[[[[[mesh_resize]]]]]]]]]]]]',vertices.shape,target_triangles,target_number_of_triangles)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces) )

    ms.apply_filter(
        'generate_alpha_wrap',
        alpha=pymeshlab.PercentageValue(alpha_fraction),
        offset=pymeshlab.PercentageValue(offset_fraction)
    )

    mesh_wrap = ms.current_mesh()
    vertices_wrap = mesh_wrap.vertex_matrix()
    faces_wrap = mesh_wrap.face_matrix()

    return vertices_wrap, faces_wrap



 

class build_mesh:
    def __init__(self, points=None):
        self.points = np.asarray(points)
        self.points = self.sort_points_into_loop(self.points)
        self.centroid = self.points.mean(axis=0)
        self.u, self.v = self.compute_pca_axes(self.points)
 
    def compute_pca_axes(self, pts):
        X = pts - pts.mean(axis=0)
        C = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)
        u = eigvecs[:, idx[-1]]
        v = eigvecs[:, idx[-2]]
        return u, v
 
    def sort_points_into_loop(self, pts):
        c = pts.mean(axis=0)
        X = pts - c
        u, v = self.compute_pca_axes(pts)
        x = np.dot(X, u)
        y = np.dot(X, v)
        angles = np.arctan2(y, x)
        order = np.argsort(angles)
        return pts[order]
 
    def project_to_plane(self, pts):
        X = pts - self.centroid
        x = np.dot(X, self.u)
        y = np.dot(X, self.v)
        return np.column_stack([x, y])

    def lift_from_plane(self, pts2d):
        x = pts2d[:, 0]
        y = pts2d[:, 1]
        return self.centroid + np.outer(x, self.u) + np.outer(y, self.v)
 
    def translate_loop(self, pts, dt):
        dirs = pts - self.centroid
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs_norm = dirs / norms
        return pts + dirs_norm * dt
 
    def build_loft_faces(self, num_loops, pts_per_loop):
        faces = []
        for k in range(num_loops - 1):
            base0 = k * pts_per_loop
            base1 = (k + 1) * pts_per_loop

            for i in range(pts_per_loop):
                i0 = base0 + i
                i1 = base0 + (i + 1) % pts_per_loop
                j0 = base1 + i
                j1 = base1 + (i + 1) % pts_per_loop

                faces.append([i0, i1, j1])
                faces.append([i0, j1, j0])

        return np.array(faces)
 
    def build_cap_from_loop(self, loop, offset):
        loop = np.asarray(loop)
        n = len(loop)
 
        normal = np.zeros(3)
        for i in range(n):
            p0 = loop[i]
            p1 = loop[(i + 1) % n]
            normal += np.cross(p0, p1)
        normal /= np.linalg.norm(normal)

        # 2. Offset centroid
        centroid = loop.mean(axis=0)
        centroid_offset = centroid + offset * normal

        # 3. Build vertices
        vertices = np.vstack([loop, centroid_offset])
        c_idx = len(vertices) - 1

        # 4. Triangle fan
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([c_idx, i, j])

        return vertices, np.array(faces) 
    
    def smooth_taubin(self, mesh, iterations=10, lamb=0.5, nu=-0.53):
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=lamb,
            nu=nu,
            iterations=iterations
        )
        return mesh
    def fan_faces(self, n):
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        return np.array(faces)

    def ring_faces(self, n):
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j])  # degenerate but acceptable for smoothing
        return np.array(faces)
    def nondegenerate_faces(self, mesh, eps=1e-12):
        verts = mesh.vertices
        faces = mesh.faces

        mask = []
        for (a, b, c) in faces:
            p0, p1, p2 = verts[a], verts[b], verts[c]
            area = np.linalg.norm(np.cross(p1 - p0, p2 - p0)) * 0.5
            mask.append(area > eps)

        return np.array(mask)

 

    def build_loft_from_loops(self, loops): 
        vertices = np.vstack(loops) 
        n_points = len(loops[0])
        n_loops = len(loops)
 
        faces = self.build_loft_faces(n_loops, n_points) 
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # clean mesh
        mesh.update_faces(mesh.unique_faces())
        mask = self.nondegenerate_faces(mesh)
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        return mesh

 
    def build_loft(self,steps=None, num_steps=5, offset=0.1,iterations=20,chunk_size=1):
        loop_r = np.linalg.norm(self.points - self.centroid, axis=1)
        R = loop_r.mean()
        if steps is None:
            steps = -np.linspace(0, R, num_steps)
        loops = [self.translate_loop(self.points, dt) for dt in steps[:-1]]
 
        vertices = np.vstack(loops)
        faces = self.build_loft_faces(len(loops), len(self.points))
 
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces) 

        mesh.update_faces(mesh.unique_faces())
        mask = self.nondegenerate_faces(mesh)
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        mesh = self.smooth_taubin(mesh, iterations=iterations)
        return mesh

 


    def build_loft_all(self, num_steps=5, offset=0.0001, iterations=20, chunk_size=1):

        current_loop = self.points.copy()
        loops = [current_loop] 
        loop_r = np.linalg.norm(current_loop - self.centroid, axis=1)
        R = loop_r.mean()
 
        steps = -np.linspace(0, R, num_steps) 
        step_chunks = np.array_split(steps, max(1, num_steps // chunk_size))

        for chunk in step_chunks: 
            partial_mesh = self.build_loft(
                steps=chunk,
                offset=offset,
                iterations=iterations,
                chunk_size=chunk_size
            )
 
            smoothed_vertices = partial_mesh.vertices
            n = len(self.points)
            current_loop = smoothed_vertices[-n:].copy() 
            self.centroid = current_loop.mean(axis=0)
 
            loops.append(current_loop)
 
        final_offset_mesh = mesh_from_points_with_offset_centroid(current_loop, offset=offset)
        last_loop = self.sort_points_into_loop(np.asarray(final_offset_mesh.vertices))
        loops.append(last_loop)
 
        final_mesh = self.build_loft_from_loops(loops) 

        return final_mesh
 

def mesh_from_points_with_offset_centroid(points, offset=0.1):
    points = np.asarray(points)
    n = len(points) 
    centroid = points.mean(axis=0)
 
    normal = np.zeros(3)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        normal += np.cross(p0, p1)
    normal = normal / np.linalg.norm(normal)
 
    centroid_offset = centroid + offset * normal
 
    vertices = np.vstack([points, centroid_offset])
    c_idx = len(vertices) - 1
 
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([c_idx, i, j])

    faces = np.array(faces, dtype=int)

    # 6. Build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh






