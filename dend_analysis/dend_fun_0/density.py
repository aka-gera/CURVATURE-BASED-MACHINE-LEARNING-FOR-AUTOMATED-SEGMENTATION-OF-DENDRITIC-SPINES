import numpy as np 
 

def projecton_ab(points,a,b): 
    d = b - a   
    t = np.dot(points-a, d) / np.dot(d, d)
    return a + np.outer(t, d)

  
def proj_extremities(points,a,b): 
    points=np.vstack((a,b,points)) 
    dists = np.linalg.norm(points-np.mean(points,axis=0), axis=1) 
    pmax = points[np.argmax(dists)] 
    pmin = points[np.argmax(np.linalg.norm(points-pmax,axis=1))]
 
    return pmin,pmax

 
def flatten_cylinder(points, a, b, 
                                     n_replicates_theta=1,
                                     n_replicates_height=1): 

    axis_vector = b.astype(float) - a.astype(float)
    L= np.linalg.norm(axis_vector)   
    axis_vector /=L
    points=np.vstack((a, points,b))
 
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis_vector)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    n1 = np.cross(axis_vector, tmp)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis_vector, n1)
    frame = (axis_vector, n1, n2)
 
    points_on_axis = points - a 

 
    height_component = np.dot(points_on_axis, axis_vector)

    hmin, hmax = np.min(height_component), np.max(height_component)
    height_range = hmax - hmin
 
    x = np.dot(points_on_axis, n1)
    y = np.dot(points_on_axis, n2)
 
    angles = np.arctan2(y, x)
 
    flattened_all = np.vstack((angles, height_component)).T
      
    flattened_points = flattened_all[1:-1]

    replicate_points = []
    grid_size_th = n_replicates_theta #int(np.sqrt(n_replicates))  # e.g. 3 for 9 replicates
    grid_size_h = n_replicates_height #int(np.sqrt(n_replicates))  # e.g. 3 for 9 replicates
    
    angle_shifts = [i * 2*np.pi for i in range(-(grid_size_th//2), grid_size_th//2 + 1)]
    height_shifts = [j * height_range for j in range(-(grid_size_h//2), grid_size_h//2 + 1)]
    
    for dtheta in angle_shifts:
        for dh in height_shifts:
            if dtheta == 0 and dh == 0:
                continue  
            replicate_points.append(flattened_points + np.array([dtheta, dh]))
     
    if replicate_points:
        flattened_points = np.vstack([flattened_points] + replicate_points)
    

    return flattened_points,flattened_all[1:-1], frame,[hmin,hmax]




def Find_points_radius(r, a, b): 
    ab = a - b
    ab_lengths = np.linalg.norm(ab, axis=1, keepdims=True)   
    ab_unit = -ab / ab_lengths 
    return a + ab_unit * r
 
  

  




from dend_fun_0.help_fun import plotly_scatter,Lines_plot,plotly_lines
 
import plotly.graph_objects as go 
from scipy.ndimage import gaussian_filter 
 


def cylinder_coordinates(theta, heights, a, frame, r):
    v, n1, n2 = frame 
    x = r * np.cos(theta)
    y = r * np.sin(theta) 
    pts = a + np.outer(heights, v) + np.outer(x, n1) + np.outer(y, n2)
    return pts


from sklearn.neighbors import KDTree 



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
                n_replicates_theta=2,#2,
                n_replicates_height=2,#2,
                eps_theta=0.0,
                eps_height=.0005,
                width=800,
                height=700, 
                 ):
        pass
        self.width=width
        self.height=height
 
        # neck_pointss=Find_points_radius(r=10*radius,a=shaft_points,b=neck_points)  
        point_center=np.linspace(point_a, point_b, num=200) 
        shaft_pointss=point_center[KDTree(point_center).query(shaft_points)[1].flatten()]
        neck_pointss =neck_points +shaft_pointss - shaft_points
        neck_pointsss=projecton_ab(neck_pointss,point_a,point_b)
        centroid_proj_cylder=Find_points_radius(r=radius,a=neck_pointsss,b=neck_pointss)  
        point_a,point_b=proj_extremities(neck_pointsss, point_a, point_b)
        flattened_points,flattened_points_org,frame,(hmin, hmax) = flatten_cylinder(centroid_proj_cylder, 
                                            a=point_a, 
                                            b=point_b, 
                                            # eps_angle=flat_offset,
                                            # eps_height=flat_offset_height,
                                            n_replicates_theta=n_replicates_theta, 
                                            n_replicates_height=n_replicates_height,
                                             )  
        heatmap, xedges, yedges = np.histogram2d(flattened_points[:,0], flattened_points[:,1], bins=bins)
        smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma_gaussian_param) 
        angle_mask = (xedges[:-1] >= -np.pi) & (xedges[:-1] <= np.pi*(1+0.2) ) 
        height_mask = (yedges[:-1] >= hmin  ) & (yedges[:-1] <= hmax )
  
  

        yedges_=yedges- (np.max(yedges) + np.min(yedges)) / 2    

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

        self.density_org_points=[]
        self.density_org_points.append(plotly_scatter(centroid_proj_cylder, marker=None, mode='markers', color='black', symbol=None, size=12, opacity=0.9, name=f'spine neck proj.', showlegend=True))
        self.density_org_points.extend(plotly_lines(neck_pointsss,centroid_proj_cylder,color='yellow',width=10,mode='lines',showlegend=False)) 
        self.density_org_points.extend(plotly_lines(neck_points,centroid_proj_cylder,color='green',width=10,mode='lines',showlegend=False)) 
        self.density_org_points.append(plotly_scatter(neck_points, marker=None, mode='markers', color='red', symbol='x', size=5, opacity=0.8, name='neck org.', showlegend=True))
        self.density_org_points.append(plotly_scatter(np.linspace(point_a,point_b, num=100) , marker=None, mode='markers', color='purple', symbol=None, size=5, opacity=0.70, name='central axis', showlegend=True)) 
        self.density_org_points.append(plotly_scatter(np.array([point_a,point_b]), marker=None, mode='markers', color='purple', symbol=None, size=15, opacity=0.99, name='extremities', showlegend=True))
        '''
        self.density_org_points.append(plotly_scatter(neck_points, marker=None, mode='markers', color='black', symbol=None, size=5, opacity=0.8, name='neck org.', showlegend=True))
        self.density_org_points.append(plotly_scatter(neck_pointss, marker=None, mode='markers', color='black', symbol='x', size=5, opacity=0.99, name='neck proj.', showlegend=True))
        self.density_org_points.append(plotly_scatter(neck_pointsss, marker=None, mode='markers', color='black', symbol='square', size=5, opacity=0.99, name='neck proj. proj.', showlegend=True))
        self.density_org_points.append(plotly_scatter(shaft_points, marker=None, mode='markers', color='blue', symbol=None, size=5, opacity=0.8, name='shaft org.', showlegend=True))
        self.density_org_points.append(plotly_scatter(shaft_pointss, marker=None, mode='markers', color='blue', symbol='x', size=5, opacity=0.99, name='shaft proj.', showlegend=True))
        self.density_org_points.extend(plotly_lines(neck_points,neck_pointss,color='black',width=5,mode='lines',showlegend=False))
        self.density_org_points.extend(plotly_lines(shaft_points,shaft_pointss,color='blue',width=5,mode='lines',showlegend=False))

        '''

        # self.density_org_points.extend(plotly_lines(shaft_pointss,centroid_proj_cylder,color='yellow',width=10,mode='lines',showlegend=False))
        # self.density_org_points.append(plotly_scatter(projection, marker=None, mode='markers', color='red', symbol=None, size=10, opacity=0.30, name='spine proj. cylinder', showlegend=True))
        # self.density_org_points.append(plotly_scatter(shaft_points, marker=None, mode='markers', color='blue', symbol=None, size=10, opacity=0.30, name='shaft cylinder', showlegend=True))

        filtered_heatmap_cyl = smoothed_heatmap[np.ix_(angle_mask, height_mask)] 
        z2d = (smoothed_heatmap.T) 
        z3d = (filtered_heatmap_cyl.T) 

        zmin = min(z2d.min(), z3d.min())
        zmax = max(z2d.max(), z3d.max())





        self.density_heatmap = go.Heatmap(
            x=xedges[:-1],  
            y=yedges_[:-1],  
            z=z2d,  
            colorscale=colorscale,
            showscale=showscale,
            zsmooth=zsmooth, 
            opacity=opacity,
            zmin=zmin,
            zmax=zmax
            # colorbar=colorbar_param,[angle_maskk]
        )
  

        theta_vals = xedges[:-1][angle_mask]
        height_vals = yedges[:-1][height_mask]

        Theta, heightt= np.meshgrid(theta_vals, height_vals)

        xyz = cylinder_coordinates(
            theta=Theta.ravel(),
            heights=heightt.ravel(),
            a=point_a,
            frame=frame,
            r=radius
        )
 
        heatmap_shape = filtered_heatmap_cyl.T.shape   

        self.density_heatmap_surface = go.Surface(
            x= xyz[:, 0].reshape(heatmap_shape),
            y= xyz[:, 1].reshape(heatmap_shape),
            z= xyz[:, 2].reshape(heatmap_shape),
            # surfacecolor=filtered_heatmap_cyl.T,  
            surfacecolor=z3d,
            colorscale=colorscale,
            showscale=showscale,
            opacity=opacity,
            colorbar=colorbar,
            cmin=zmin,
            cmax=zmax
        ) 


 
 