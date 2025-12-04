# CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES

## Installation (Mac)

Open **Terminal** and paste the commands below:
```bash
cd ~/Desktop
git clone https://github.com/aka-gera/CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES.git
cd CURVATURE-BASED-MACHINE-LEARNING-FOR-AUTOMATED-SEGMENTATION-OF-DENDRITIC-SPINES
chmod +x setup.sh
./setup.sh
 
cd dend_analysis
python3.11 -m venv ../dsa_venv
source ../dsa_venv/bin/activate
python -m  gunicorn -w 4 -b 0.0.0.0:8050 wsgi:server  --timeout 1200 -c gunicorn.conf.py

```
This will:
- Clone the repository  
- Install Python 3.9 and required libraries  
- Create a virtual environment named **`dsa_venv`**  
- Launch the application in your browser  
 

Once the browser window opens:

- Click on **DSA** in the top‑left corner.  
- Drag one or multiple `.obj` meshes into the segmentation box.  
  - If you don’t provide meshes, two demo meshes included in the repository will be used.  
  - The parent directory of meshes is shown in the **“Add Destination”** box. Update this path if needed.  

---

## Parameters & Options

- **Model Architecture**  
  Use the dropdowns to select the segmentation model.  
  - `DNN-3` generally performs better.  

- **Smoothing (Smooth box)**  
  - Enable/disable smoothing for new meshes.  
  - Check the **“Smooth”** box to activate.  
  - Clicking the label opens parameter options (e.g., step size, total steps).  

- **Resizing (Resize box)**  
  - Enable/disable resizing for large meshes (>600,000 vertices).  
  - Clicking the label opens parameter options for resizing.  

- **Spine–Shaft Segm**  
  - Click **“spine-shaft segm”** to open a dropdown.  
  - Adjust weights or thresholds to classify spines.  
  - Enable the checkbox to run segmentation.  

- **Morphologic Parameters (Morphologic Param)**  
  - Enable/disable to perform head–neck segmentation.  
  - Computes head/neck diameter and length.  

- **Clean Path Directory**  
  - Enable the **“clean_path_dir”** box to clear previous output directories before running new segmentation.  

- **Run Analysis**  
  -Double Click to run the segmentation after all parameters are set up.
  
- **Restart**  
  -Click the **Restart** button to kill and restart the process.
 









# Checking Results After Segmentation  

Once segmentation has terminated, you can check the results by following these steps:  

---

## Restart the Application  

1. Kill the current running code in the terminal using **`Ctrl + C`**.  
2. Reopen the application by running:  

```bash
python -m  gunicorn -w 4 -b 0.0.0.0:8050 wsgi:server -c gunicorn.conf.py
```
3. Alternatively, use the **Restart button** in the interface and then refresh your browser page.

## Navigating the Interface

- On the right of **DSA** (top corner), click on **“pinn”** if the **pinn** method was used.  
- Select the **architecture (DNN-nth)** that was used.  
- Click to choose the **file parent name**.  
- Click on the **mesh name**.  
- A new page will appear with visualization options.  

---

## Visualization Options

Use the dropdown buttons to explore different features:

- **Mesh Selection**  
  - Switch between multiple meshes if more than one was dropped.  

- **Method Selection**  
  - Choose the method used (e.g., **PNN**, **cML**) if available.  

- **Architecture Selection**  
  - Select the **DNN-nth** architecture.  

- **Skeleton Visualization**  
  - Choose skeleton view.  
  - **model_sp_iou** → visualize Jaccard index.  
  - **model_sp_loss** → visualize training loss.  

- **SHAP Coefficients**  
  - **model_shap** → visualize SHAP coefficients.  

- **Morphologic Parameters**  
  - Plot morphologic parameters against each other.  
  - Visualize **heatmap** and **cylindrical heatmap** showing spine distribution on dendrite segments.  

---

## Feature Visualization

Features used in training/testing can be visualized:

- Distance shaft skeleton vertices  
- Regionalization:  
  - With smoothing (**kmean_n**)  
  - Without smoothing (**kmean_mean_n**)  
- Gaussian curvature and mean curvature (and their squares):  
  - With smoothing (**mean**, **gauss**)  
  - Without smoothing (**imean**, **igauss**)  

---

## Additional Dropdowns

- Switch view between **dendrite segment**, **shaft**, and **spines segmented**.  
- Under **image/skeleton**, visualize dendrite segment parts.  
- Toggle between **Smoothed** and **Initial (non-smoothed)** meshes.  
- Change page theme: **seaborn**, **plotly_dark**, etc.  

---

## Graph Controls

- Adjust histogram bin count using the **radio bar**.  
- Modify **graph width** and **height** for better visualization.  
