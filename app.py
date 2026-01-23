import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy import ndimage
import os
import gdown

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
    Executes the full image analysis pipeline (Tang et al., 2012 logic).
    Handles both AI Detection and Manual Blue Fill methods with mode-specific cleaning.
    """
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # ---------------------------------------------------------
    # A. Pre-processing
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Initialize the binary map (combined_map)
    combined_map = None
    min_clean_size = 200 # Default for AI mode

    # ---------------------------------------------------------
    # B. Detection Strategy (AI vs Manual)
    # ---------------------------------------------------------
    
    if method == "AI Detection (YOLO)":
        # 1. YOLO Prediction
        img_input = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)
        results = model.predict(img_input, conf=0.05, save=False, verbose=False)
        
        if results[0].masks is None:
            structure_map = np.zeros(gray.shape, dtype=np.uint8)
        else:
            masks = results[0].masks.data.cpu().numpy()
            structure_map = np.zeros(gray.shape, dtype=np.uint8)
            for m in masks:
                m_resized = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                structure_map = np.maximum(structure_map, m_resized)

        # 2. Adaptive Thresholding
        connectivity_map = cv2.adaptiveThreshold(
            gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 85, 15
        )
        # High noise filter for AI mode to remove soil texture noise
        connectivity_clean = remove_small_objects(connectivity_map.astype(bool), min_size=250).astype(np.uint8)

        # Fusion
        combined_map = cv2.bitwise_or(structure_map.astype(np.uint8), connectivity_clean)
        
        # Set cleaning threshold high for AI to avoid noise
        min_clean_size = 200

    else: # method == "Manual Blue Fill"
        # Create mask for Perfect Blue [R=0, G=0, B=255]
        # Allowing slight tolerance for anti-aliasing
        lower_blue = np.array([0, 0, 240]) 
        upper_blue = np.array([10, 10, 255])
        
        mask = cv2.inRange(img_rgb, lower_blue, upper_blue)
        combined_map = mask
        
        # Set cleaning threshold VERY LOW for manual mode 
        # (We trust your manual paint, so we keep even small dots/lines)
        min_clean_size = 10

    # ---------------------------------------------------------
    # C. Cleaning & Refinement
    # ---------------------------------------------------------
    # Ensure binary format (0 or 255)
    _, combined_map = cv2.threshold(combined_map, 127, 255, cv2.THRESH_BINARY)
    
    # Use smaller kernel for manual mode to preserve fine details
    kernel_size = (3, 3) if method == "Manual Blue Fill" else (5, 5)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    closed_map = cv2.morphologyEx(combined_map, cv2.MORPH_CLOSE, kernel_bridge, iterations=1)
    
    # DYNAMIC CLEANING: Uses 200 for AI (removes noise) but 10 for Manual (keeps small cracks)
    clean_map = remove_small_objects(closed_map.astype(bool), min_size=min_clean_size).astype(np.uint8)
    
    # Remove small holes inside the cracks
    clean_map = remove_small_holes(clean_map.astype(bool), area_threshold=min_clean_size).astype(np.uint8)

    # ---------------------------------------------------------
    # D. Skeletonization
    # ---------------------------------------------------------
    skeleton_base = skeletonize(clean_map)
    # Dilate to ensure connectivity before final skeletonization
    # Use smaller dilation for manual mode to prevent merging close cracks
    dilation_kernel = (3, 3) if method == "Manual Blue Fill" else (11, 11)
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    
    final_binary_map = cv2.dilate(skeleton_base.astype(np.uint8), kernel_thick, iterations=1)
    final_skeleton = skeletonize(final_binary_map)

    # ---------------------------------------------------------
    # E. Metric Calculations
    # ---------------------------------------------------------
    h, w = final_binary_map.shape
    total_area_cm2 = (h * w) / (px_per_mm ** 2) / 100

    # 1. Clod Analysis (N_c, A_av)
    clod_mask = 1 - final_binary_map
    clod_mask = remove_small_objects(clod_mask.astype(bool), min_size=20).astype(np.uint8)
    num_clods, _ = cv2.connectedComponents(clod_mask)
    num_clods -= 1  # remove background
    avg_clod_area = (np.sum(clod_mask) / num_clods) / (px_per_mm ** 2) / 100 if num_clods > 0 else 0

    # 2. Surface Crack Ratio (R_sc)
    crack_pixels = np.sum(final_binary_map)
    surface_crack_ratio = (crack_pixels / (h * w)) * 100

    # 3. Node Analysis (N_n)
    skel_int = final_skeleton.astype(int)
    conv = np.array([[1,1,1],[1,1,1],[1,1,1]])
    neighbor_count = ndimage.convolve(skel_int, conv, mode='constant', cval=0)
    raw_nodes = (skel_int == 1) & (neighbor_count > 3)
    num_node_clusters, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_nodes.astype(np.uint8))
    real_node_count = num_node_clusters - 1
    node_density = real_node_count / total_area_cm2
    node_coords = centroids[1:]

    # 4. Segment Analysis (N_seg, L_av, D_c)
    skel_segments = skel_int.copy()
    skel_segments[raw_nodes] = 0 
    
    # Allow smaller segments in manual mode
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
    width_samples = dist_map[final_skeleton] * 2
    avg_width = (np.mean(width_samples) / px_per_mm) / 10 if len(width_samples) > 0 else 0
    
    # Volume = Crack Area * Thickness
    est_volume_cm3 = (crack_pixels / (px_per_mm ** 2) / 100) * (thickness_mm / 10)

    metrics
