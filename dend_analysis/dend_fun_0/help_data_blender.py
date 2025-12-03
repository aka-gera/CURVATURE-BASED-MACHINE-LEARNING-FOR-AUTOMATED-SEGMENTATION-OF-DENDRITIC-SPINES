import bpy
import os
import logging

 
import trimesh
import numpy as np 
from dend_fun_0.get_path import get_param 


def export_vertex_group_to_obj(obj_name, vgroup_name, export_folder):
    try:
        # Get the object
        obj = bpy.data.objects.get(obj_name)
        if not obj or obj.type != 'MESH':
            logging.error(f"Object '{obj_name}' not found or is not a mesh.")
            return False
         
        vgroup = obj.vertex_groups.get(vgroup_name)
        if not vgroup:
            logging.error(f"Vertex group '{vgroup_name}' not found.")
            return False
 
        mesh = obj.data
        selected_verts = {v.index for v in mesh.vertices if any(g.group == vgroup.index for g in v.groups)}
 
        new_mesh = mesh.copy()
        new_mesh.name = f"{obj.name}_{vgroup_name}_mesh"
         
        new_mesh.polygons.foreach_set("hide", [not all(v in selected_verts for v in poly.vertices) for poly in new_mesh.polygons])
        new_mesh.update()  # Ensure mesh data is updated to reflect the changes
 
        new_obj = bpy.data.objects.new(f"{obj.name}_{vgroup_name}", new_mesh)
        bpy.context.collection.objects.link(new_obj) 
        bpy.context.view_layer.objects.active = new_obj
        new_obj.select_set(True)
 
        bpy.ops.object.mode_set(mode='OBJECT')
 
        obj_export_path = os.path.join(export_folder, f"{obj.name}_{vgroup_name}_quad.obj")
        logging.info(f"Exporting OBJ to {obj_export_path}")
        bpy.ops.export_scene.obj(filepath=obj_export_path, use_selection=True)
 
        stl_export_path = os.path.join(export_folder, f"{obj.name}_{vgroup_name}_quad.stl")
        logging.info(f"Exporting STL to {stl_export_path}")
        bpy.ops.export_mesh.stl(filepath=stl_export_path, use_selection=True)

        return True
    
    except Exception as e:
        logging.error(f"Error processing {obj_name} vertex group '{vgroup_name}': {e}")
        return False

def export_vertex_group_to_obj_trimesh(obj_name, vgroup_name, export_path):
    # Get the object
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        print(f"Object '{obj_name}' not found or is not a mesh.")
        return False

    # Get the vertex group
    vgroup = obj.vertex_groups.get(vgroup_name)
    if not vgroup:
        print(f"Vertex group '{vgroup_name}' not found.")
        return False

    # Get mesh data
    mesh = obj.data
    vertices = np.array([v.co for v in mesh.vertices]) 
    selected_verts = {v.index for v in mesh.vertices if any(g.group == vgroup.index for g in v.groups)} 
    faces = [tuple(p.vertices) for p in mesh.polygons if all(v in selected_verts for v in p.vertices)]

    # If no faces are found, return
    if not faces:
        print(f"No faces found in vertex group '{vgroup_name}'.")
        return False

    # Create a trimesh object and export
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    new_mesh.export(export_path)

    print(f"Successfully exported {obj_name} vertex group '{vgroup_name}' to {export_path}")
    return True



  
def get_mesh_obj(spine_name, save_directory=None):
    obj = bpy.data.objects.get(spine_name)  

    if obj and obj.type == 'MESH':
        mesh = obj.data  # Access mesh data
 
        vertices = np.array([v.co[:] for v in mesh.vertices]) 
        faces = np.array([p.vertices[:] for p in mesh.polygons]) 
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces) 
        if save_directory is None:
            save_directory = bpy.path.abspath("//")   
        os.makedirs(save_directory, exist_ok=True) 
        save_path = os.path.join(save_directory, f"{spine_name}.obj") 
        tmesh.export(save_path)
        print(f"Mesh saved as OBJ: {save_path}")

        return tmesh  
    else:
        print(f"Object '{spine_name}' not found or is not a mesh.")
        return None
    

def get_all_mesh_obj(spine_group_name, save_directory=None):
    # Set default save directory if none is provided
    if save_directory is None:
        save_directory = os.path.join(bpy.path.abspath("//"), spine_group_name)  

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Filter objects based on name starting with `spine_group_name`
    filtered_objects = [obj.name for obj in bpy.data.objects if obj.name.startswith(spine_group_name)]
    
    if not filtered_objects:
        print(f"No objects found with prefix '{spine_group_name}'.")
        return
    
    # Process each matching object
    for spine_name in filtered_objects:
        get_mesh_obj(spine_name, save_directory=save_directory)

    print(f"All meshes saved in: {save_directory}")

    
import glob 



def get_all_mesh_obj_union(spine_group_name, save_directory=None): 
    if save_directory is None:
        save_directory = os.path.join(bpy.path.abspath("//"), spine_group_name)    
 
    if not os.path.exists(save_directory):
        print(f"Directory '{save_directory}' does not exist.")
        return None
 
    file_list = sorted(glob.glob(os.path.join(save_directory, f"{spine_group_name}*.obj")))

    if not file_list:
        print(f"No OBJ files found for '{spine_group_name}' in '{save_directory}'.")
        return None
 
    meshes = [trimesh.load(file) for file in file_list]

    if not meshes:
        print("No valid meshes were loaded.")
        return None

    union_mesh = meshes[0]
    for mesh in meshes[1:]:
        union_mesh = union_mesh+mesh 
 
    save_path = os.path.join(save_directory, f"{spine_group_name}_union.obj")
 
    union_mesh.export(save_path)
    print(f"Union mesh saved at: {save_path}")


'''
import bpy

# Get all object names
object_names = [obj.name for obj in bpy.data.objects]

# Print the names
print(object_names)

'''



# run with blender python
'''
import sys,os
import numpy as np

file_supp=r'G:\tom_dendrites\suppfiles'
file_supp=r'G:\dend_analysis\myles'
file_path_org=r'G:\dend_analysis'
# file_path=os.getcwd()   
sys.path.append(os.path.abspath(fr'{file_path_org}\dend_fun\\' ))
from obj_get import Obj_to_vertices, Objs_to_obj
from get_path import get_param 
from help_data_blender import blender_obj


obj_list=np.arange(152)
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list]
dend_path_inits=['BTLNG-d004-segmented' for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}sp'] for de in dend_names]
blender_obj(file_path_org,dend_path_inits,dend_names,dend_namess)

obj_list=np.arange(152)
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list]
dend_path_inits=['tom_dendrites' for _ in range(len(dend_names))]
dend_namess = [[f'{de}_',de,f'{de}p_sps'] for de in dend_names]
blender_obj(file_path_org,dend_path_inits,dend_names,dend_namess)

'''


def blender_obj(file_path_org,dend_path_inits,dend_names,dend_namess):
    
    dend_cla=get_param(file_path_org=file_path_org,  
                    dend_names=dend_names,
                    dend_namess=dend_namess, 
                    dend_path_inits=dend_path_inits, 
                    ) 

    for index,spine_group_name in enumerate(dend_names): 
        dend_cla.get_dend_name(index=index)  
        save_directory= dend_cla.dend_path_original_m
        get_all_mesh_obj(spine_group_name,save_directory=save_directory)
        get_all_mesh_obj_union(spine_group_name,save_directory=save_directory)



'''

obj_list = np.arange(152)  
namm=[]
dend_names = [f'd{str(i).zfill(3)}' for i in obj_list]
for name in dend_names:
    save_directory = os.path.join('G:\\dend_analysis', 'tom_dendrites', name, 'data_org', 'dataaa') 
    files = list_files(save_directory=save_directory, spine_group_name=f'{name}p_sps')

    if files:
        # print(f"Total files found: {name} {len(files)}") 
        namm.append(name)

'''


def list_files(save_directory, spine_group_name,matching_files_thre=5): 
    matching_files = sorted(glob.glob(os.path.join(save_directory, f"{spine_group_name}*.obj")))

    if not matching_files: 
        return None 

    if len(matching_files) > matching_files_thre: 
        return matching_files  # Return the list of matching files

    return None


