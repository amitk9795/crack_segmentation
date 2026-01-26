import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy import ndimage
import os
import gdown
import pandas as pd
import io

# =============================================================================
# 1. PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Geotechnical Crack Analysis | Amit Kumar",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
    }
    .stTable {
        font-size: 1.1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. SYSTEM CONFIGURATION & MODEL LOADING
# =============================================================================

# Google Drive File ID for best.pt
MODEL_ID = '1bQUE7ZL8luHRPWPX9P2u3zdtyAtWbQNP'
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_ID}'
MODEL_FILENAME = 'best.pt'

@st.cache_resource
def download_and_load_model():
    """
    Downloads the YOLO model from Google Drive if not present, then loads it.
    Uses st.cache_resource to ensure this only happens once.
    """
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading Model from Drive... (This may take a moment)"):
            try:
                gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    try:
        model = YOLO(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# =============================================================================
# 3. ANALYSIS LOGIC
# =============================================================================

def process_image(image_file, model, px_per_mm, thickness_mm, method):
    """
    Executes the image analysis pipeline.
    - AI MODE: Reverted to the robust previous version (Thick Dilation).
    - MANUAL MODE: Uses the lightweight version (Direct Detection).
    """
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Pre-processing setup
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    final_binary_map = None
    
    # ---------------------------------------------------------
    # METHOD 1: AI DETECTION (YOLO) - RESTORED TO PREVIOUS VERSION
    # ---------------------------------------------------------
    if method == "AI Detection (YOLO)":
        img_input = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 1. YOLO Prediction
        results = model.predict(img_input, conf=0.05, save=False, verbose=False)
        
        if results[0].masks is None:
            structure_map = np.zeros(gray.shape, dtype=np.uint8)
        else:
            masks = results[0].masks.data.cpu().numpy()
            structure_map = np.zeros(gray.shape, dtype=np.uint8)
            for m in masks:
                m_resized = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                structure_map = np.maximum(structure_map, m_resized)

        # 2. Adaptive Thresholding (Optimized for clay textures)
        connectivity_map = cv2.adaptiveThreshold(
            gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 85, 15
        )
        connectivity_clean = remove_small_objects(connectivity_map.astype(bool), min_size=250).astype(np.uint8)

        # 3. Fusion & Cleaning (Specific to AI logic)
        combined_map = cv2.bitwise_or(structure_map.astype(np.uint8), connectivity_clean)
        
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Note: Iterations=2 was the setting that worked well for AI
        closed_map = cv2.morphologyEx(combined_map, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)
        
        # Stricter cleaning for AI to reduce noise
        clean_map = remove_small_objects(closed_map.astype(bool), min_size=200).astype(np.uint8)
        clean_map = remove_small_holes(clean_map.astype(bool), area_threshold=200).astype(np.uint8)
        
        # 4. Reconstruction via Dilation (The key step for AI visibility)
        skeleton_base = skeletonize(clean_map)
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        final_binary_map = cv2.dilate(skeleton_base.astype(np.uint8), kernel_thick, iterations=1)

    # ---------------------------------------------------------
    # METHOD 2: MANUAL BLUE FILL (As confirmed working)
    # ---------------------------------------------------------
    else:
        # Convert BGR to HSV
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Define range for Blue
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([150, 255, 255])
        
        # Create mask
        raw_binary_map = cv2.inRange(hsv_img, lower_blue, upper_blue)
        
        # Minimal refinement for manual (trust the user's paint)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        raw_binary_map = cv2.morphologyEx(raw_binary_map, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Standard cleaning
        _, binary_base = cv2.threshold(raw_binary_map, 127, 255, cv2.THRESH_BINARY)
        clean_map = remove_small_objects(binary_base.astype(bool), min_size=50).astype(np.uint8)
        clean_map = remove_small_holes(clean_map.astype(bool), area_threshold=50).astype(np.uint8)
        
        # Direct assignment (No artificial dilation for manual)
        final_binary_map = clean_map

    # ---------------------------------------------------------
    # D. Final Skeletonization & Metrics (Common)
    # ---------------------------------------------------------
    # Generate Skeleton for calculations (re-skeletonize the final map)
    # We bridge small gaps first to ensure skeleton continuity
    skel_prep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skel_prep = cv2.morphologyEx(final_binary_map, cv2.MORPH_CLOSE, skel_prep_kernel, iterations=2)
    final_skeleton = skeletonize(skel_prep.astype(bool)).astype(np.uint8)

    h, w = final_binary_map.shape
    total_area_cm2 = (h * w) / (px_per_mm ** 2) / 100

    # 1. Surface Crack Ratio (R_sc)
    crack_pixels = np.sum(final_binary_map)
    surface_crack_ratio = (crack_pixels / (h * w)) * 100

    # 2. Clod Analysis (N_c, A_av)
    clod_mask = 1 - final_binary_map
    clod_mask = remove_small_objects(clod_mask.astype(bool), min_size=20).astype(np.uint8)
    num_clods, _ = cv2.connectedComponents(clod_mask)
    num_clods -= 1  # remove background
    avg_clod_area = (np.sum(clod_mask) / num_clods) / (px_per_mm ** 2) / 100 if num_clods > 0 else 0

    # 3. Node Analysis (N_n)
    skel_int = final_skeleton.astype(int)
    conv = np.array([[1,1,1],[1,1,1],[1,1,1]])
    neighbor_count = ndimage.convolve(skel_int, conv, mode='constant', cval=0)
    
    # Intersections have > 3 neighbors (center + >2 arms)
    raw_nodes = (skel_int == 1) & (neighbor_count > 3)
    
    num_node_clusters, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_nodes.astype(np.uint8))
    real_node_count = num_node_clusters - 1
    node_density = real_node_count / total_area_cm2
    node_coords = centroids[1:]

    # 4. Segment Analysis (N_seg, L_av, D_c)
    skel_segments = skel_int.copy()
    
    # Dilate nodes to break skeleton
    node_mask = cv2.dilate(raw_nodes.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)
    skel_segments[node_mask == 1] = 0 
    
    # Use different segment threshold based on method
    min_seg_size = 5 if method == "Manual Blue Fill" else 8
    valid_segments = remove_small_objects(skel_segments.astype(bool), min_size=min_seg_size)
    
    num_segments, _ = cv2.connectedComponents(valid_segments.astype(np.uint8))
    num_segments -= 1
    
    segment_density = num_segments / total_area_cm2
    total_len_cm = np.sum(final_skeleton) / px_per_mm / 10
    avg_crack_length = total_len_cm / num_segments if num_segments > 0 else 0
    crack_density = total_len_cm / total_area_cm2

    # 5. Width & Volume (W_av, Volume)
    dist_map = cv2.distanceTransform(final_binary_map, cv2.DIST_L2, 5)
    width_samples = dist_map[final_skeleton == 1] * 2
    avg_width = (np.mean(width_samples) / px_per_mm) / 10 if len(width_samples) > 0 else 0
    
    # Volume = Crack Area * Thickness
    est_volume_cm3 = (crack_pixels / (px_per_mm ** 2) / 100) * (thickness_mm / 10)

    metrics = {
        "R_sc": surface_crack_ratio,
        "N_c": num_clods,
        "A_av": avg_clod_area,
        "N_n": node_density,
        "N_seg": segment_density,
        "L_av": avg_crack_length,
        "D_c": crack_density,
        "W_av": avg_width,
        "Volume": est_volume_cm3
    }
    
    # Prepare red overlay for visualization
    red_mask = np.zeros_like(img_rgb)
    red_mask[:, :, 0] = 255
    overlay = cv2.addWeighted(img_rgb, 1.0, 
                             cv2.bitwise_and(red_mask, red_mask, mask=final_binary_map), 0.6, 0)

    images = {
        "Original": img_rgb,
        "Binary Map": final_binary_map,
        "Skeleton": final_skeleton,
        "Overlay": overlay,
        "Nodes": node_coords
    }
    
    return metrics, images

# =============================================================================
# 4. MAIN APPLICATION UI
# =============================================================================

def main():
    st.title("🏗️ Geotechnical Desiccation Crack Analysis")
    st.markdown("""
    **Developed by:** Amit Kumar  
    **Context:** Automated quantification of desiccation crack patterns in clayey soils using image processing techniques 
    as described by *Tang et al. (2012)*.
    """)
    st.divider()

    # --- SIDEBAR: INPUTS ---
    with st.sidebar:
        st.header("1. Configuration")
        
        # Analysis Mode Selection
        method = st.radio(
            "Select Analysis Method:",
            ("AI Detection (YOLO)", "Manual Blue Fill"),
            help="Choose 'AI Detection' for raw soil images. Choose 'Manual Blue Fill' if you have manually painted cracks with Blue (0,0,255)."
        )
        
        # Load Model Automatically only if AI mode is needed
        model = None
        if method == "AI Detection (YOLO)":
            model = download_and_load_model()
            if model:
                st.success(f"✅ Model Loaded: {MODEL_FILENAME}")
            else:
                st.error("❌ Model Failed to Load")
                st.stop()
        else:
            st.info("ℹ️ Using Manual Color Extraction Mode")

        # Image Uploader
        image_file = st.file_uploader("Upload Soil Image", type=['jpg', 'jpeg', 'png'])
        
        st.header("2. Calibration")
        px_per_mm = st.number_input("Pixels per mm", min_value=1.0, value=4.4333, format="%.4f",
                                   help="Calibration factor to convert pixels to metric units.")
        thickness_mm = st.number_input("Layer Thickness (mm)", min_value=1.0, value=8.0, format="%.1f",
                                     help="Thickness of the soil layer for volume estimation.")
        
        run_btn = st.button("🚀 Run Analysis")

    # --- MAIN EXECUTION ---
    if run_btn:
        if not image_file:
            st.error("Please upload an Image file to proceed.")
        else:
            # Check model availability only for AI mode
            if method == "AI Detection (YOLO)" and model is None:
                st.error("Model is required for AI Detection.")
            else:
                with st.spinner(f"Processing Crack Network ({method})..."):
                    # Pass the selected method to the processing function
                    metrics, images = process_image(image_file, model, px_per_mm, thickness_mm, method)
                
                # --- RESULTS SECTION ---
                st.success("Analysis Complete")
                
                # TAB 1: VISUALIZATION
                tab1, tab2, tab3 = st.tabs(["📊 2x2 Visual Grid", "📋 Geometric Metrics", "📑 Definitions"])
                
                with tab1:
                    # Creating a 2x2 Matplotlib Figure
                    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                    
                    # 1. Original Image
                    axes[0, 0].imshow(images["Original"])
                    axes[0, 0].set_title("1. Original Image", fontsize=8)
                    axes[0, 0].axis('off')
                    
                    # 2. Binary Map (Black & White)
                    axes[0, 1].imshow(images["Binary Map"], cmap='gray')
                    axes[0, 1].set_title("2. Binary Crack Map", fontsize=8)
                    axes[0, 1].axis('off')
                    
                    # 3. Skeleton & Nodes
                    # Visualize thickened skeleton so it is visible in the plot
                    vis_skeleton = cv2.dilate(images["Skeleton"], np.ones((2,2), np.uint8), iterations=1)
                    axes[1, 0].imshow(vis_skeleton, cmap='gray_r')
                    
                    node_coords = images["Nodes"]
                    if len(node_coords) > 0:
                        axes[1, 0].scatter(node_coords[:, 0], node_coords[:, 1], c='red', s=10)
                    axes[1, 0].set_title("3. Skeleton & Nodes", fontsize=8)
                    axes[1, 0].axis('off')
                    
                    # 4. Overlay (Segmentation)
                    axes[1, 1].imshow(images["Overlay"])
                    axes[1, 1].set_title(f"4. Overlay (R_sc={metrics['R_sc']:.1f}%)", fontsize=8)
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    
                    # Center the figure in Streamlit
                    col_spacer1, col_fig, col_spacer2 = st.columns([1, 2, 1])
                    with col_fig:
                        st.pyplot(fig)

                # TAB 2: METRICS
                with tab2:
                    st.markdown("#### Complete Geometric Parameters")
                    
                    # Prepare data for DataFrame
                    data = {
                        "Parameter": [
                            "Surface crack ratio (R_sc)", 
                            "Number of clods (N_c)", 
                            "Average area of clods (A_av)", 
                            "Number of nodes per unit area (N_n)", 
                            "Crack segments per unit area (N_seg)", 
                            "Average length of cracks (L_av)", 
                            "Crack density (D_c)", 
                            "Average width of cracks (W_av)",
                            "Estimated Crack Volume (V_cr)"
                        ],
                        "Value": [
                            metrics['R_sc'],
                            metrics['N_c'],
                            metrics['A_av'],
                            metrics['N_n'],
                            metrics['N_seg'],
                            metrics['L_av'],
                            metrics['D_c'],
                            metrics['W_av'],
                            metrics['Volume']
                        ],
                        "Unit": [
                            "%",
                            "-",
                            "cm²",
                            "cm⁻²",
                            "cm⁻²",
                            "cm",
                            "cm⁻¹",
                            "cm",
                            "cm³"
                        ]
                    }
                    
                    # Create DataFrame
                    df_metrics = pd.DataFrame(data)
                    
                    # Display as interactive Table (allows copying)
                    st.dataframe(
                        df_metrics.style.format({"Value": "{:.4f}"}), 
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    # Create Excel file in memory
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_metrics.to_excel(writer, index=False, sheet_name='Geometric Parameters')
                        
                        # Adjust column width for better readability in the downloaded file
                        workbook = writer.book
                        worksheet = writer.sheets['Geometric Parameters']
                        format_header = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#D3D3D3'})
                        
                        for col_num, value in enumerate(df_metrics.columns.values):
                            worksheet.write(0, col_num, value, format_header)
                            worksheet.set_column(col_num, col_num, 25)

                    # Download Button
                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=buffer.getvalue(),
                        file_name="Geometric_Parameters.xlsx",
                        mime="application/vnd.ms-excel"
                    )

                # TAB 3: DEFINITIONS
                with tab3:
                    st.markdown("### 📚 Terminology & Definitions")
                    st.info("The following definitions are based on **Tang et al. (2012)**.")
                    
                    st.markdown("""
                    * **Surface Crack Ratio ($R_{sc}$):** Defined as the ratio of the crack area to the total surface area of the soil specimen.
                    * **Number of Clods ($N_c$):** The clod is defined as the independent closed area that is split by cracks.
                    * **Average Area of Clods ($A_{av}$):** The mean surface area of the identified soil clods.
                    * **Number of Nodes ($N_n$):** The number of intersection nodes (where crack segments meet) or end nodes.
                    * **Number of Crack Segments ($N_{seg}$):** The count of distinct crack segments defining the outline of the soil crack pattern.
                    * **Average Length of Cracks ($L_{av}$):** The average trace length of the medial axis of crack segments.
                    * **Crack Density ($D_c$):** Calculated as the total crack length per unit area.
                    * **Average Width of Cracks ($W_{av}$):** Determined by calculating the shortest distance from a randomly chosen point on one boundary to the opposite boundary.
                    * **Estimated Crack Volume ($V_{cr}$):** A derived volumetric estimation calculated as the Crack Area multiplied by the specimen thickness.
                    """)

if __name__ == "__main__":
    main()
