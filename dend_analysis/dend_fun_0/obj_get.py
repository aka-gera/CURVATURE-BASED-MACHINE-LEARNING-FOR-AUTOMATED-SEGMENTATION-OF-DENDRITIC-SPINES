
 
'''
file_path_original = r'E:\kasthuri15_rawobj\\'
mesh_name = 'Kasthuri15__3181_Objects.Neurons.Dendrites.Spiny_Dendrites.RC_Oblique_Dendrites.Oblique_Dendrite_1_AS' 
[sys.path.append(i) for i in ['..']]


''' 
  
import numpy as np  
import glob
import os
import re 
import trimesh
import base64 

def Objs_to_obj(file_path_original, mesh_name, file_destination=None, file_type='obj'): 
    input_file_path = os.path.join( file_path_original, f'{mesh_name}.{file_type}' )
    if not os.path.exists(input_file_path):
        print(f"File : {input_file_path}")
        return 
    if file_destination and not os.path.exists(file_destination):
        os.makedirs(file_destination)
    
    if file_type == 'obj':
        print(f"File not found: {input_file_path}")
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
        print(f"File not found: {input_file_path}")
        print(f"File not found: {lines}")
        print(f"File not found: {input_file_path}")
        current_object = None
        object_data = []
        mtl_data = []

        for line in lines:
            if line.startswith('mtllib'):
                mtl_data.append(line)
            
            if line.startswith('o '):
                if current_object:
                    output_path = os.path.join(file_destination, f"{current_object}.obj")
                    with open(output_path, 'w') as out_file:
                        out_file.writelines(mtl_data + object_data)
                    object_data = []
                
                current_object = line.split()[1]
            
            object_data.append(line)

        if current_object:
            output_path = os.path.join(file_destination, f"{current_object}.obj")
            with open(output_path, 'w') as out_file:
                out_file.writelines(mtl_data + object_data)
    
    elif file_type == 'ply':
        mesh = trimesh.load(input_file_path, file_type='ply')
        output_path = os.path.join(file_path_original, f"{mesh_name}.obj")
        mesh.export(output_path, file_type='obj') 
        Objs_to_obj(file_path_original, mesh_name, file_destination=file_destination, file_type='obj')
        print(f"Converted {input_file_path} to {output_path}")
    else:
        print("Unsupported file type. Only 'obj' and 'ply' are supported.")



def Obj_to_coord(file_path_original, faces_new_mesh=True,save=True): 
    if not os.path.exists(file_path_original):
        print(f"File not found: {file_path_original}")
        return
    else:  

        vertices = []
        faces = []
 
        with open(file_path_original, 'r') as file:
            lines = file.readlines()


        for line in lines: 
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            # Extract faces
            elif line.startswith('f '): 
                face = [int(f.split('/')[0]) for f in line.split()[1:]]
                faces.append(face[:3]) 
        vertices = np.array(vertices)
        faces = np.array(faces )
        if faces_new_mesh:
            faces-=faces.flatten().min()
 
        return vertices,faces
   




def Obj_to_vertices(file_path_original, mesh_name, file_destination=None, nbm=0,faces_new_mesh=True,save=True):
    full_path = os.path.join(file_path_original,f'{mesh_name}.obj')
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return
    else:  

        vertices = []
        faces = []

        # Read the OBJ file
        with open(full_path, 'r') as file:
            lines = file.readlines()


        for line in lines:
            # Extract vertices
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            # Extract faces
            elif line.startswith('f '): 
                face = [int(f.split('/')[0]) for f in line.split()[1:]]
                faces.append(face[:3]) 
        vertices = np.array(vertices)
        faces = np.array(faces )
        if faces_new_mesh:
            faces-=faces.flatten().min() 
        if save:
            txt_save = True
            if file_destination is None:
                txt_save = False
                file_destination = file_path_original

            if txt_save:
                file_path_destination=fr'{file_destination}'
                os.makedirs(file_path_destination, exist_ok=True)

            np.savetxt(os.path.join(file_path_destination,f'vertices_{nbm}.txt'),  vertices, fmt='%f')  
            np.savetxt(os.path.join(file_path_destination,f'faces_{nbm}.txt'), faces, fmt='%d') 
            print('-------------Obj_to_vertices____________-- i saved in ',file_path_destination,)
        else:
            return vertices,faces
        
def get_obj_files(directory): 
    obj_files = sorted(glob.glob(os.path.join(directory, "*.obj")))
    return obj_files
 

def get_obj_filenames(directory,startwith=None):     
    obj_files = sorted(glob.glob(os.path.join(directory, "*.obj"))) 
    obj_filenames = [os.path.splitext(os.path.basename(file))[0] for file in obj_files]
    if startwith is None:
        return obj_filenames
    else:
        return [ob for ob in obj_filenames if ob.startswith(startwith)]
    



def get_obj_filenames_with_indices(directory, startwith=None,ext="*.obj"):     
    obj_files = sorted(glob.glob(os.path.join(directory,ext ))) 
    obj_filenames = [os.path.splitext(os.path.basename(file))[0] for file in obj_files]

    if startwith:
        obj_filenames = [ob for ob in obj_filenames if ob.startswith(startwith)]
    
    print('==========-----',obj_filenames)
    obj_indices = {}
    for filename in obj_filenames:
        match = re.search(r'\d+$', filename)  
        if match:
            obj_indices[filename] = int(match.group())

    return  obj_indices 


def get_obj_filenames_with_indices_2(directory, startwith=None,ext="*.obj"):
    obj_files = sorted(glob.glob(os.path.join(directory,ext )))   
    if not obj_files:
        return [] 
    
    obj_filenames = [os.path.splitext(os.path.basename(file))[0] for file in obj_files]

    if startwith:
        obj_filenames = [ob for ob in obj_filenames if ob.startswith(startwith)]
     

    return  obj_filenames 

 

def get_file_with_indices(directory, startwith=None,ext="*.obj"):     
    obj_files = sorted(glob.glob(os.path.join(directory,ext ))) 
    obj_filenames = [os.path.splitext(os.path.basename(file))[0] for file in obj_files]

    if startwith:
        obj_filenames = [ob for ob in obj_filenames if ob.startswith(startwith)]
 
    obj_indices = {}
    for filename in obj_filenames:
        match = re.search(r'\d+$', filename)  
        if match:
            obj_indices[filename] = int(match.group())

    return  obj_indices 

 


def parse_obj_upload(contents, filename, export_dir=None ):  
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    obj_text = decoded.decode("utf-8")

    vertices = []
    faces = []

    # Parse OBJ lines
    for line in obj_text.splitlines():
        if line.startswith('v '):  # vertex line
            vertex = list(map(float, line.split()[1:]))
            vertices.append(vertex)
        elif line.startswith('f '):  # face line 
            face = [int(f.split('/')[0]) - 1 for f in line.split()[1:]]
            if len(face) >= 3:
                faces.append(face[:3]) 
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
 
    if export_dir and filename:
        export_path = os.path.join(export_dir, f"{filename}.obj")
        mesh.export(export_path) 
    return mesh



