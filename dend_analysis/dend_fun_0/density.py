import numpy as np 
 

#################### Dendrite Density Plot ############################################# 
 
def flatten_cylinder(points, a, b, eps_angle=0.05, eps_height=0.005, n_replicates_theta=1, n_replicates_height=1): 
    # Ensure the axis vector is float
    axis_vector = b.astype(float) - a.astype(float)
    axis_vector /= np.linalg.norm(axis_vector)  # Normalize
    
    points_on_axis = points - a 
    height_component = np.dot(points_on_axis, axis_vector)
     
    projection = points_on_axis - np.outer(height_component, axis_vector)
     
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
     
    hmin, hmax = np.min(height_component), np.max(height_component)
    height_range = hmax - hmin
     
    flattened_points = np.vstack((angles, height_component)).T
     
    replicate_points = []
    grid_size_th = n_replicates_theta #int(np.sqrt(n_replicates))  # e.g. 3 for 9 replicates
    grid_size_h = n_replicates_height #int(np.sqrt(n_replicates))  # e.g. 3 for 9 replicates
    
    angle_shifts = [i * 2*np.pi for i in range(-(grid_size_th//2), grid_size_th//2 + 1)]
    height_shifts = [j * height_range for j in range(-(grid_size_h//2), grid_size_h//2 + 1)]
    
    for dtheta in angle_shifts:
        for dh in height_shifts:
            if dtheta == 0 and dh == 0:
                continue  # skip original
            replicate_points.append(flattened_points + np.array([dtheta, dh]))
     
    if replicate_points:
        flattened_points = np.vstack([flattened_points] + replicate_points)
    
    return flattened_points,np.vstack((angles, height_component)).T



def Find_points_radius(r, a, b): 
    ab = a - b
    ab_lengths = np.linalg.norm(ab, axis=1, keepdims=True)   
    ab_unit = ab / ab_lengths 
    return a + ab_unit * r

 
def Closest_point_on_line(line_start, line_end, point): 
    
    line_vec = line_end - line_start 
    point_vec = point - line_start 

    line_vec_norm = line_vec / np.linalg.norm(line_vec)#np.repmat(,point.shape[0],point[1])  
    return line_start  + np.outer( np.dot(point_vec, line_vec_norm),line_vec_norm)

#################### Dendrite Density Plot #############################################


def calculate_intensity(cylder, heatmap, xedges, yedges): 
    # xedges,yedges=np.meshgrid(xedges,yedges)
    # xedges,yedges=xedges.flatten(),yedges.flatten()
    # Reshape the cylindrical coordinates 
    x_cylinder = cylder[:, 0]   # Cylindrical x-coordinates
    y_cylinder = cylder[:, 1]   # Cylindrical y-coordinates 
    intensity = np.zeros((len(x_cylinder), len(y_cylinder))) 
    for i in range(intensity.shape[0]):
        for j in range(intensity.shape[1]): 
            x_value = x_cylinder[i]
            y_value = y_cylinder[j]
            for ii in range(xedges.size-1):
                for jj in range(yedges.size-1):
                    if (x_value >=xedges[ii]) and (x_value<xedges[ii+1]) and (y_value >=yedges[jj]) and (y_value<yedges[jj+1]) : 
                        intensity[i, j] = heatmap[ii,jj]

    return intensity













 
import plotly.graph_objects as go 
from scipy.ndimage import gaussian_filter 

def cylinder_coordinates(angles, heights, a, b,r): 
    v = b - a
    height = np.linalg.norm(v)
    v /= height  
    not_v = np.array([1, 0, 0]) if not np.allclose(v, [1, 0, 0]) else np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
 
    theta, z = np.meshgrid(angles, heights)
    theta=theta.reshape(-1,1)
    z=z.reshape(-1,1)
    
    # Cylinder coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
     
    X, Y, Z = [a[i] + v[i] * z + r * (n1[i] * x + n2[i] * y) for i in range(3)]
    return np.hstack((X,Y,Z))






class  get_cylinder:
    def __init__(self,  
                shaft_points, 
                neck_points,
                radius,
                point_a,
                point_b,  
                sigma_gaussian_param=10.5,
                epss=100.5,
                flat_offset_height= 100.4,
                flat_offset=100.4,  
                colorbar_param=None,
                colorbar=None,
                bins = (200, 270),
                colorscale='Jet',
                showscale=False,
                zsmooth='best', 
                opacity=1.,
                scale_factor=.1 ,
                n_replicates_theta=2,
                n_replicates_height=2,
                eps_theta=0.0,
                eps_height=.0005,
                width=800,
                height=700,
                 ):
        pass
        self.width=width
        self.height=height
        centroid_proj_cylder=Find_points_radius(r=radius,a=shaft_points,b=neck_points) 

 
        flattened_points,flattened_points_org = flatten_cylinder(centroid_proj_cylder, 
                                            a=point_a, 
                                            b=point_b, 
                                            eps_angle=flat_offset,
                                            eps_height=flat_offset_height,
                                            n_replicates_theta=n_replicates_theta, 
                                            n_replicates_height=n_replicates_height,) 
        heatmap, xedges, yedges = np.histogram2d(flattened_points[:,0], flattened_points[:,1], bins=bins)
 
        smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma_gaussian_param)
 
        angle_mask = (xedges[:-1] >= -np.pi - eps_theta) & (xedges[:-1] <= np.pi*(1+0.05) + eps_theta)

        hmin, hmax = np.min(flattened_points_org[:,1]), np.max(flattened_points_org[:,1])
        height_mask = (yedges[:-1] >= hmin - eps_height) & (yedges[:-1] <= hmax + eps_height)
 
        filtered_heatmap_cyl = smoothed_heatmap[np.ix_(angle_mask, height_mask)]


        # **Filter only the rows of the heatmap corresponding to -π ≤ angles ≤ π**
        angle_maskk = (xedges[:-1] > -np.pi-epss) & (xedges[:-1] <  np.pi+epss)
        filtered_heatmap = smoothed_heatmap # [angle_mask, :]
        minv,maxv=np.min(np.min(heatmap,axis=0)),np.max(np.max(heatmap,axis=0))

        yedges_=yedges- (np.max(yedges) + np.min(yedges)) / 2   
        if colorbar_param is None:
            colorbar_param=dict(
                title='Spines Density',     
                # tickvals=[0,.01,.02,.03,.04,.05,.06],      # Set tick values from 0 to 2
                # ticktext=['0', '','1','','2','','', ], # Set tick labels from 0 to 2
                tickvals=[0,.01,.02,.03,.04,.05,.06],      # Set tick values from 0 to 2
                ticktext=[f'{minv}', '','',f'{(maxv+minv)/2}','','',  f'{maxv}'], 
                # ticks="" ,       # Remove tick marks
                yanchor="middle",  # Position the colorbar
                len=0.75,        # Adjust height of the colorbar
                x=1.0 ,         # Position it outside the plot
                thickness=10, 
                title_side='right',        
                titleside='right'            
            )  
        xx= (flattened_points[:,0] >= -np.pi-epss) & (flattened_points[:,0] <= np.pi+epss) 

        yy=flattened_points[:,1]-(np.max(flattened_points[:,1]) + np.min(flattened_points[:,1])) / 2  
        yyy=flattened_points_org[:,1]-(np.max(flattened_points_org[:,1]) + np.min(flattened_points_org[:,1])) / 2 
 
        self.density_heatmap_points = go.Scatter(
            x=flattened_points[:,0],
            y=yy,
            mode="markers",
            marker=dict(color="green", size=15), 
            name="Replicas Points"
        )
        self.density_heatmap_points_org = go.Scatter(
            x=flattened_points_org[:,0],
            y=yyy,
            mode="markers",
            marker=dict(color="black", size=20), 
            name="Original Points"
        )



        self.density_heatmap = go.Heatmap(
            x=xedges[:-1],  
            y=yedges_[:-1],  
            z=filtered_heatmap.T,  
            colorscale=colorscale,
            showscale=showscale,
            zsmooth=zsmooth, 
            opacity=opacity,
            # colorbar=colorbar_param,[angle_maskk]
        )
 
        filtered_heatmap_cyl = smoothed_heatmap[np.ix_(angle_mask, height_mask)] 
        xyz = cylinder_coordinates(
            angles=xedges[:-1][angle_mask],  
            heights=yedges[:-1][height_mask],
            a=point_a, 
            b=point_b,
            r=radius,
        )
 
        heatmap_shape = filtered_heatmap_cyl.T.shape  
        y_scaled = xyz[:, 1]

        self.density_heatmap_surface = go.Surface(
            x= xyz[:, 0].reshape(heatmap_shape),
            y= y_scaled.reshape(heatmap_shape),
            z= xyz[:, 2].reshape(heatmap_shape),
            surfacecolor=filtered_heatmap_cyl.T,  
            colorscale=colorscale,
            showscale=showscale,
            opacity=opacity,
            colorbar=colorbar
        )

 