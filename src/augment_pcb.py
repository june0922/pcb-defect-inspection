import os
import cv2
import numpy as np
import random
from pathlib import Path

def apply_color_mapping(binary_img):
    # binary_img is grayscale (0 or 255)
    h, w = binary_img.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Random Background Base
    bg_choice = random.choice(['green', 'blue'])
    if bg_choice == 'green':
        bg_color = (random.randint(0, 20), random.randint(50, 100), random.randint(0, 20)) # B, G, R
    else:
        bg_color = (random.randint(50, 100), random.randint(0, 20), random.randint(0, 20)) # B, G, R

    # 2. Random Circuit Base
    circuit_choice = random.choice(['copper', 'silver', 'gold'])
    if circuit_choice == 'copper':
        c_color = (51, 115, 184)
    elif circuit_choice == 'silver':
        c_color = (192, 192, 192)
    else:
        c_color = (0, 215, 255)

    colored[binary_img == 0] = c_color
    colored[binary_img == 255] = bg_color
    return colored

def add_noise_and_texture(img):
    h, w, c = img.shape
    # Add Gaussian noise
    noise = np.random.normal(0, 15, (h, w, c)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img_noisy

def apply_embossing(colored_img, binary_img):
    # Create a 3D effect based on the binary mask (edges)
    # Edge detection
    edges = cv2.Canny(binary_img, 50, 150)
    
    # Slight shift for shadow and highlight
    h, w = binary_img.shape
    highlight = np.zeros_like(colored_img)
    shadow = np.zeros_like(colored_img)
    
    # Shift edges
    shift_h, shift_v = 1, 1
    M_h = np.float32([[1, 0, shift_h], [0, 1, shift_v]])
    M_s = np.float32([[1, 0, -shift_h], [0, 1, -shift_v]])
    
    edges_h = cv2.warpAffine(edges, M_h, (w, h))
    edges_s = cv2.warpAffine(edges, M_s, (w, h))
    
    colored_img = colored_img.astype(np.float32)
    colored_img[edges_h == 255] += 50
    colored_img[edges_s == 255] -= 50
    return np.clip(colored_img, 0, 255).astype(np.uint8)

def apply_blur_and_antialias(img):
    # Blur to remove aliasing
    return cv2.GaussianBlur(img, (3, 3), 0.5)

def apply_camera_degradation(img):
    # Defocus blur randomly
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 1.0)
    
    # JPEG compression artifact simulation
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def process_images(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # find all image files
    img_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"Found {len(img_files)} images in {input_dir}")
    
    for img_file in img_files:
        # Load as grayscale since it's binary
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Ensure it's binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 1. Color mapping
        colored = apply_color_mapping(binary)
        
        # 2. Add texture & noise
        textured = add_noise_and_texture(colored)
        
        # 3. Embossing effect
        embossed = apply_embossing(textured, binary)
        
        # 4. Blur & anti-aliasing
        blurred = apply_blur_and_antialias(embossed)
        
        # 5. Camera degradation
        final_img = apply_camera_degradation(blurred)
        
        # Save output
        out_file = output_path / img_file.name
        cv2.imwrite(str(out_file), final_img)
        print(f"Processed and saved: {out_file}")

if __name__ == '__main__':
    # Define directories
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_DIR = PROJECT_ROOT / "merged_data"
    OUTPUT_DIR = PROJECT_ROOT / "merged_colored_data"
    
    process_images(INPUT_DIR, OUTPUT_DIR)
