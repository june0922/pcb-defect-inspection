import os
import glob
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse
import sys
import time
import datetime
import cv2

def solve_group(group_dir, output_prefix, args, out_dir):
    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"No temp files found in {group_dir}")
        return

    print(f"[{output_prefix}] Reconstructing original massive PCB...")
    
    img_width, img_height = 640, 640
    # DeepPCB uses a 25x25 grid for the 16000x16000 original image
    grid_cols = 25
    grid_rows = 25
    
    canvas_w = grid_cols * img_width
    canvas_h = grid_rows * img_height
    
    canvas_temp = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    canvas_test = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    
    min_x, max_x = canvas_w, 0
    min_y, max_y = canvas_h, 0

    for temp_file in temp_files:
        basename = os.path.basename(temp_file)
        # e.g., 00041011_temp.jpg -> 00041011
        name_no_ext = basename.replace('_temp.jpg', '')
        
        # The last 3 digits represent the absolute index in the 25x25 grid
        idx_str = name_no_ext[-3:]
        idx = int(idx_str)
        
        col = idx % grid_cols
        row = idx // grid_cols
        
        x = col * img_width
        y = row * img_height
        
        min_x, max_x = min(min_x, x), max(max_x, x + img_width)
        min_y, max_y = min(min_y, y), max(max_y, y + img_height)

        # Paste Temp
        im_temp = Image.open(temp_file)
        canvas_temp.paste(im_temp, (x, y))
        
        # Paste Test
        test_file = temp_file.replace('_temp.jpg', '_test.jpg')
        if os.path.exists(test_file):
            im_test = Image.open(test_file)
            canvas_test.paste(im_test, (x, y))
        else:
            # If test is missing, fill with white to avoid black gaps
            im_test = Image.new('RGB', (img_width, img_height), (255, 255, 255))
            canvas_test.paste(im_test, (x, y))

    # Crop to the actual bounding box of available pieces to save disk space
    if min_x < max_x and min_y < max_y:
        canvas_temp = canvas_temp.crop((min_x, min_y, max_x, max_y))
        canvas_test = canvas_test.crop((min_x, min_y, max_x, max_y))
        
    out_temp_path = os.path.join(out_dir, f"{output_prefix}_reconstructed_normal.jpg")
    out_test_path = os.path.join(out_dir, f"{output_prefix}_reconstructed_defect.jpg")
    
    canvas_temp.save(out_temp_path, quality=95)
    canvas_test.save(out_test_path, quality=95)
    
    print(f"[{output_prefix}] Saved Reconstructed Image: {canvas_temp.size[0]}x{canvas_temp.size[1]} px")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Absolute Coordinate Reconstructor")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process all groups sequentially")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data/merged_clusters', help="Output directory")
    
    # We keep the old parameters in argparse so scripts/run_jigsaw.bat doesn't crash
    parser.add_argument('--threshold', type=float, default=15.0, help="Ignored: Absolute positioning used")
    parser.add_argument('--edge_depth', type=int, default=2, help="Ignored: Absolute positioning used")
    parser.add_argument('--color_mode', type=str, default='rgb', help="Ignored: Absolute positioning used")
    parser.add_argument('--mutual_best', type=bool, default=True, help="Ignored: Absolute positioning used")
    parser.add_argument('--pattern_bonus', type=float, default=30.0, help="Ignored: Absolute positioning used")
    
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(project_root, args.dataset_dir)
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()
    
    if args.group == 'all':
        group_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('group') and os.path.isdir(os.path.join(dataset_path, d))])
        
        if not group_dirs:
            print(f"No group directories found in {dataset_path}")
            sys.exit(1)
            
        print(f"Found {len(group_dirs)} groups. Processing sequentially (Complete Isolation)...")
        print("=" * 60)
        
        for g in group_dirs:
            g_dir = os.path.join(dataset_path, g, g.replace('group', ''))
            solve_group(g_dir, g, args, out_dir)
            
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"\n---> [Total Time Elapsed] {format_time(elapsed)}")
                
    else:
        group_name = args.group
        group_dir = os.path.join(dataset_path, group_name, group_name.replace('group', ''))
        solve_group(group_dir, group_name, args, out_dir)
        elapsed = time.time() - start_time
        print(f"\n---> [Time Elapsed] {format_time(elapsed)}")
        
    print("\nAll operations completed successfully!")

if __name__ == '__main__':
    main()
