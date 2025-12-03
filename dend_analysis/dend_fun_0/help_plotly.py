import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np   




import plotly.graph_objs as go
import plotly.io as pio

def hide_button(fig, enable=True):
    if enable:
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Hide All",
                         method="update",
                         args=[{"visible": ['legendonly']*len(fig.data)}]),
                    dict(label="Show All",
                         method="update",
                         args=[{"visible": [True]*len(fig.data)}])
                ],
                font=dict(color='black'),
                x=1.08,
                y=1.0,
                xanchor='left',
                yanchor='bottom',
            )]
        )
    else:
        fig.update_layout(updatemenus=[])

        
def get_mesh3d(dend, intensity, colorscale='jet'):
    return go.Mesh3d(
        x=dend.vertices[:, 0],
        y=dend.vertices[:, 1],
        z=dend.vertices[:, 2],
        i=dend.faces[:, 0],
        j=dend.faces[:, 1],
        k=dend.faces[:, 2],
        intensity=intensity,
        colorscale=colorscale,
        showscale=False,
        colorbar=dict(
            title=None,     # No title for the colorbar
            tickvals=[],    # No tick values
            ticktext=[],    # No tick labels
            ticks=""        # Remove tick marks
        )
    )    

def scatter_plot(vertices,intensity=None,width=800,height=600):
    scatter1 = go.Scatter3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],

        mode='markers',
        marker=dict(
            size=2,
            color=intensity,  # Assign intensity values
            colorscale='Viridis',  # Choose a colorscale
            colorbar=dict(title='Intensity')  # Add a colorbar to indicate intensity
        )
    ) 


    # scatter2 = go.Scatter3d(
    #     x=Vert_init[:,0],
    #     y=Vert_init[:,1],
    #     z=Vert_init[:,2],

    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color='black'
    #     ),
    #     showlegend=False
    # ) 
    layout = go.Layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ), 
        width=width,  
        height=height  
    )
    # Create the figure and plot it
    fig = go.Figure(data=[scatter1 ], layout=layout)
    pio.show(fig)



def plotly_mesh(vertices,
                faces,
                intensity=None,
                colorscale='balance', 
                opacity=1.,
                fig_name='figure', 
                width=900,
                height=700,
                color_template = 'plotly_dark',
                paper_bgcolor = 'grey',
                color_font = 'white',
                size_font = 20,   
                title = ' ' ,   
                showscale=True,  
                 colorbar=None,
                colorbar_len=None,
                colorbar_title=None,     # No title for the colorbar
                colorbar_tickvals=None,    # No tick values
                colorbar_ticktext=None,    # No tick labels
                colorbar_ticks=""        # Remove tick marks
                ): 
    if colorbar is None:
        colorbar=dict(
            len=colorbar_len,
            title=colorbar_title,     # No title for the colorbar
            tickvals=colorbar_tickvals,    # No tick values
            ticktext=colorbar_ticktext,    # No tick labels
            ticks=colorbar_ticks,        # Remove tick marks
        )
    mesh3d = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=intensity,  
        colorscale=colorscale, 
        opacity=opacity,  
        showscale=showscale,  
        name=fig_name,
        colorbar=colorbar,
    ) 

    fig = go.Figure(data=[mesh3d])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        width=width,
        height=height
    )

    fig.update_layout(
            barmode='overlay',
            paper_bgcolor=paper_bgcolor,  
            font=dict(color=color_font,size=size_font), 
            template=color_template, 
            hovermode='closest', 
            title = title,
    )
    return fig







def plotly_analysis(energy,  
                    volume,
                    areas,
                    dt=1, 
                    width=1300, 
                    height=400, 
                    color_template='plotly', 
                    paper_bgcolor=None, 
                    color_font=None, 
                    size_font=20, 
                    title=' '):
     
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Energy", "Volume","Area"))
 
    colors = [f'rgba({i*30 % 256}, {(i*60) % 256}, {(i*90) % 256}, 0.5)' for i in range(energy.shape[1])]
    
    # Add Energy traces
    for i in range(energy.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dt*np.arange(len(energy[:, i])),
                y=energy[:, i], 
                legendgroup="enegy",    
                legendgrouptitle_text="Energy: dt", 
                name=f'{2**i*dt}',
                mode='lines', 
                line=dict(color=colors[i]),
                showlegend=True
            ), 
            row=1, col=1
        )
        
    fig.update_xaxes(title_text="time", row=1, col=1, title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="Energy", 
        row=1, col=1, 
        title_font=dict(size=size_font), 
        type="log",
        title_standoff=.0  
    )

    # Add Volume traces
    for i in range(volume.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dt*np.arange(len(volume[:, i])),
                y=np.abs(volume[:, i]),  
                legendgroup="volume",
                legendgrouptitle_text="Volume: dt",
                name=f'{2**i*dt}',
                mode='lines',   
                line=dict(color=colors[i]), 
                showlegend=True
            ), 
            row=1, col=2
        )

    fig.update_xaxes(title_text="Time", row=1, col=2, title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="Volume", 
        row=1, col=2, 
        title_font=dict(size=size_font), 
        # type="log",
        title_standoff=.0   
    )

    # Add Volume traces
    for i in range(areas.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dt*np.arange(len(areas[:, i])),
                y=np.abs(areas[:, i]),  
                legendgroup="area",
                legendgrouptitle_text="Area: dt",
                name=f'{2**i*dt}',
                mode='lines',   
                line=dict(color=colors[i]), 
                showlegend=True
            ), 
            row=1, col=3
        )

    fig.update_xaxes(title_text="Time", row=1, col=3, title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="Area", 
        row=1, col=3, 
        title_font=dict(size=size_font), 
        # type="log",
        title_standoff=.0   
    )
     
     
    fig.update_layout(
        width=width,
        height=height,
        title_text=title,
        title_font=dict(size=size_font),
        font=dict(size=size_font, color=color_font),
        template=color_template,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=80, r=20, t=50, b=50)
    )
    # fig.update_layout(legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01
    # ))
 
    min_energy = np.min(energy, axis=0)
    min_volume = np.min(volume, axis=0)  

    print("Min Energy for each step size:", min_energy)
    print("Min Volume for each step size:", min_volume)

    return fig 
 





def plotly_analysis_error(error,    
                    dt=1, 
                    width=1000, 
                    height=400, 
                    color_template='plotly', 
                    paper_bgcolor=None, 
                    color_font=None, 
                    size_font=20, 
                    title=' '):
     
    fig = make_subplots(rows=1, cols=2, subplot_titles=("q_2 norm", "Convergence Rate"))
 
    colors = [f'rgba({i*30 % 256}, {(i*60) % 256}, {(i*90) % 256}, 0.5)' for i in range(error.shape[1])]
    
    # Add error traces
    for i in range(error.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=dt*np.arange(len(error[:, i])),
                y=error[:, i], 
                legendgroup="q_2 norm",    
                legendgrouptitle_text="q_2 norm", 
                name=f'dt= {2**i*dt}',
                mode='lines', 
                line=dict(color=colors[i]),
                showlegend=True
            ), 
            row=1, col=1
        )
        
    fig.update_xaxes(title_text="Time", row=1, col=1, title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="q_2 norm", 
        row=1, col=1, 
        title_font=dict(size=size_font), 
        type="log",
        title_standoff=.0  
    )

    conv=convergence_rate(error) 
    x=dt*np.arange(len(conv[:, 0]))
    for i in range(conv.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=conv[:, i],  
                legendgroup="Conv: dt",
                legendgrouptitle_text="Conv",
                name=f'{2**i*dt}',
                mode='lines',   
                line=dict(color=colors[i]), 
                showlegend=True
            ) , 
            row=1, col=2
        )
        
    # fig.update_xaxes(title_text="Step",
    #                  title_font=dict(size=size_font))
    # fig.update_yaxes(
    #     title_text="Convergence",  
    #     title_font=dict(size=size_font),  
    #     title_standoff=.0  
    # )
    # # Add Volume traces
    # for i in range(volume.shape[1]):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=np.arange(len(volume[:, i])),
    #             y=volume[:, i],  
    #             legendgroup="volume",
    #             legendgrouptitle_text="Volume",
    #             name=f'dt= {2**i*dt}',
    #             mode='lines',   
    #             line=dict(color=colors[i]), 
    #             showlegend=True
    #         ), 
    #         row=1, col=2
    #     )

    fig.update_xaxes(title_text="Time", row=1, col=2, title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="Conv", 
        row=1, col=2, 
        title_font=dict(size=size_font), 
        # type="log",
        title_standoff=.0   
    )
     
    fig.update_layout(
        width=width,
        height=height,
        title_text=title,
        title_font=dict(size=size_font),
        font=dict(size=size_font, color=color_font),
        template=color_template,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=80, r=20, t=50, b=50)
    )
    # fig.update_layout(legend=dict(
    #     yanchor="top",
    #     y=0.99,
    #     xanchor="left",
    #     x=0.01
    # ))
 
    min_error = np.mean(error, axis=0)
    min_mean = np.mean(conv, axis=0)  

    print("Min L_2 norm for each step size:", min_error)
    print("Min convergence for each step size:", min_mean)

    return fig 
 



def plotly_convergence_rate(error,    
                    dt=1, 
                    width=600, 
                    height=400, 
                    color_template='plotly', 
                    paper_bgcolor=None, 
                    color_font=None, 
                    size_font=20, 
                    title=' '):
     
    conv=convergence_rate(error)
    fig = go.Figure()
    x=dt*np.arange(len(conv[:, 0]))
    for i in range(conv.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=conv[:, i],  
                name=f'dt= {2**i*dt}',
                mode='lines',  
                showlegend=True
            ) 
        )
        
    fig.update_xaxes(title_text="Time",
                     title_font=dict(size=size_font))
    fig.update_yaxes(
        title_text="Convergence",  
        title_font=dict(size=size_font),  
        title_standoff=.0  
    )
 
    fig.update_layout(
        width=width,
        height=height,
        title_text=title,
        title_font=dict(size=size_font),
        font=dict(size=size_font, color=color_font),
        template=color_template,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=80, r=20, t=50, b=50)
    )
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.80
    ))
 
    min_conv = np.min(conv, axis=0) 

    print("Convergence rate for each time:", min_conv) 

    return fig 
 



def convergence_rate(error):
    sav = np.zeros((error.shape[0],error.shape[1]-1))
    for i in range(error.shape[1]-1):
        sav[:,i] = -(np.log(error[:,i])-np.log(error[:,i+1]))/np.log(2) 

    return sav



def Plotly_histogram_return(data,nbinsx=30,title=None,xtitle=None,name=None,ytitle='Count',xrange=None,yrange=None,color='blue',opacity=0.75,width=600, height=400):
    histogram = go.Histogram(
        x=data,
        nbinsx=nbinsx,  
        marker=dict(color=color),
        opacity=opacity,
        name=name ,
    )
    
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xtitle, range=xrange,),
        yaxis=dict(title=ytitle, range=yrange,),
        width=width, 
        height=height,
    )
    return histogram,layout
 
  
def Plotly_histogram(data,nbinsx=20,title=None,xtitle=None,xrange=None,yrange=None,ytitle='Count',width=600, height=400):
    histogram = go.Histogram(
        x=data,
        nbinsx=nbinsx,  
        marker=dict(color='blue')
    )
    
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xtitle, range=xrange,),
        yaxis=dict(title=ytitle, range=yrange,),
        width=width, 
        height=height,
    )
    
    fig = go.Figure(data=[histogram], layout=layout) 
    pio.show(fig)



def plotly_metric(headd,neckk,lengthh,height=800,width=800,title_size=20,colorscale='Blues',xtitle=None,ytitle=None,ztitle=None,marginal=None):
    fig = go.Figure()

    # Scatter plot overlay with colorbar at the bottom
    fig.add_trace(go.Scatter(
        x=headd,
        y=neckk,  
        mode='markers',
        marker=dict(
            color=lengthh,  # Use lengthh for color intensity
            colorscale='hot',  # Set your desired colorscale
            size=15,
            colorbar=dict(
                title=dict(
                    text=ztitle,
                    font=dict(size=title_size)
                ),
                orientation='h',  # Horizontal orientation
                x=0.5,  # Position in the middle (adjust as needed)
                y=1.,  # Push it below the plot
                xanchor='center',  # Center the colorbar
                # thickness=15,  # Control the thickness (height) of the colorbar
                len=0.5 , # Adjust the length of the colorbar, 
            ),
            cmax=max(lengthh),  # Set max for color scaling
            cmin=min(lengthh)   # Set min for color scaling
        ),
        xaxis='x',
        yaxis='y'
    ))

                    #     m, b = np.polyfit(headd, neckk, 1)

                #     # Create regression line
                #     reg_x = np.linspace(headd.min(), neckk.max(), 100)
                #     reg_y = m * reg_x + b  
                #     figure.add_trace(go.Scatter(x=reg_x, y=reg_y, mode="lines", name="Linear fit"))
                #     equation_text = f"y = {m:.3f}x + {b:.3f}"
                #     figure.add_annotation(
                #         x=headd.mean(),   
                #         y=neckk.max(),   
                #         text=equation_text,
                #         showarrow=False,
                #         font=dict(size=20, color="black"),
                #         bgcolor="white",
                #         bordercolor="black"
                #     )
    fig.add_trace(go.Histogram2dContour(
        x=headd, 
        y=neckk,
        colorscale=colorscale,
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))


    fig.add_trace(go.Histogram(
        y=neckk,
        xaxis='x2',
        marker=dict(
            color='rgba(0, 0, 0, 1)'
        )
    ))
    
    fig.add_trace(go.Histogram(
        x=headd,
        yaxis='y2',
        marker=dict(
            color='rgba(0, 0, 0, 1)'
        )
    ))

    # Update layout
    fig.update_layout(
        autosize=False,
        height=height,
        width=width,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        
        # Main plot settings
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False,
            # ticklen=20 ,
                title=dict(
                    text=xtitle,
                    font=dict(size=title_size)
                ),
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False, 
            # ticklen=20 ,
                title=dict(
                    text=ytitle,
                    font=dict(size=title_size)
                ),
        ),
        
        xaxis2=dict(
            zeroline=False,
            # ticklen=20 ,
            domain=[0.85, 1],
            showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            ticklen=20 ,
            domain=[0.85, 1],
            showgrid=False
        )
    )

    return fig



def Confusion_matrix(y_true,y_predicted):
    TP = np.sum((y_predicted == 1) & (y_true == 1))   
    TN = np.sum((y_predicted == 0) & (y_true == 0)) 
    FP = np.sum((y_predicted == 1) & (y_true == 0))  
    FN = np.sum((y_predicted == 0) & (y_true == 1))  

    return np.array([[TN, FP],  # True Negatives, False Positives
                      [FN, TP]])  # False Negatives, True Positives


def Compute_metrics(cm):
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]

    # Accuracy
    accuracy = (TP + TN) / np.sum(cm)

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall (Sensitivity)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

# cm = np.array([[50, 10],  # True Negatives, False Positives
#                [5, 35]])  # False Negatives, True Positives

from sklearn.metrics import classification_report
import plotly.figure_factory as ff   


class aka_plot : 
    def __init__(self, tcouleur='plotly_dark', bcouleur='navy', fcouleur='white', fsize=20):
        self.tcouleur = tcouleur
        self.bcouleur = bcouleur
        self.fcouleur = fcouleur
        self.fsize = fsize 
        self.update_layout_parameter = dict(        
                                        # barmode='overlay',  
                                        font=dict(color=fcouleur,size=fsize),  
                                        title_x=0.015,
                                        title_y=0.97,
                                        template=self.tcouleur
                                        )
        self.update_axes = dict(  
                            title_font = {"size": 14},
                            # title_standoff = 25
                            )

    def Plotly_confusion_matrix(self,cm,
                                width=600,
                                height=600,
                                labels = ['False', 'True'],
                                title="Confusion Matrix",
                                xaxis_title="Predicted Label",
                                yaxis_title="Actual Label",): 
        
        
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,  # Predicted labels
                y=labels,  # Actual labels
                colorscale='Blues',
                text=cm,  # Show numbers in cells
                texttemplate="%{text}",  # Format as numbers
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            width=width,
            height=height
        )
        fig.update_layout(**self.update_layout_parameter)
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes)
        return fig


    def Plotly_classification_report(self,y, y_predict, lab=1,
                                width=600,
                                height=600,
                                labels = ['False', 'True'],
                                title='Classification Report',
                                xaxis_title="Predicted Label",
                                yaxis_title="Actual Label",): 
        # Get the classification report as a string
        report_str = classification_report(y, y_predict, zero_division=0)

        # Process the report to extract rows
        report_lines = [line.strip() for line in report_str.split('\n') if line.strip()]
        data = [line.split() for line in report_lines[1:]]

        # Extract numerical values and convert them to a numpy array
        cm = np.array([[float(x) if x.replace('.', '', 1).isdigit() else 0 for x in row[1:]] for row in data[:-3]])
        cm[:,-1]=cm[:,-1].astype(int)
        colss1 = ['precision', 'recall', 'f1-score', 'support']

        # Create the heatmap
        if lab == 1:
            fig = ff.create_annotated_heatmap(
                cm,
                x=colss1,
                y=labels,
                colorscale='Viridis'
            )
        else:
            cmm = cm[:, :-1]
            annotation_text = [['' for _ in range(cmm.shape[1])] for _ in range(cmm.shape[0])]
            fig = ff.create_annotated_heatmap(
                cmm,
                x=colss1[:-1],
                colorscale='Viridis',
                showscale=True,
                annotation_text=annotation_text
            )
            # fig.update_yaxes(
            #     title_text='y',
            #     showticklabels=False
            # )

        # Add layout
        # fig.update_layout(title='Classification Report')

        fig.update_layout(
            title=title,
            # xaxis_title=xaxis_title,
            # yaxis_title=yaxis_title,
            # xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            # yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
            width=width,
            height=height
        )
        fig.update_layout(**self.update_layout_parameter)
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes)
        return fig



    def Plotly_Figure(self, data,layout  ): 
        fig =go.Figure(data=data, layout=layout) 
        fig.update_layout(**self.update_layout_parameter)
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes)
        return fig


    def Plotly_Figure_Sub(self ,subplot_titles,rows=2, cols=1,
    shared_xaxes=True,
    shared_yaxes=False, ):  
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes
        )

        fig.update_layout(**self.update_layout_parameter)
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes)
        return fig



def get_mesh3d(vertices,faces, intensity=None, colorscale='jet'):
    return go.Mesh3d(
        x= vertices[:, 0],
        y= vertices[:, 1],
        z= vertices[:, 2],
        i= faces[:, 0],
        j= faces[:, 1],
        k= faces[:, 2],
        intensity=intensity,
        colorscale=colorscale,
        showscale=False,
        colorbar=dict(
            title=None,     
            tickvals=[],    
            ticktext=[],    
            ticks=""        
        )
    )    