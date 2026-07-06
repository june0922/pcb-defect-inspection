import os
import glob
import numpy as np
import argparse
import sys
import time
import datetime
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

def solve_group(group_dir, output_prefix, args, out_dir):
    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"No temp files found in {group_dir}")
        return

    print(f"[{output_prefix}] Reconstructing original massive PCB...")
    
    N = len(temp_files)
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in temp_files]
    
    edges = []
    # Compare each image with up to 20 neighbors to build the translation graph
    for i in range(N):
        for j in range(i + 1, min(N, i + 21)):
            shift, response = cv2.phaseCorrelate(np.float32(imgs[i]), np.float32(imgs[j]))
            if response > 0.03:
                edges.append((i, j, shift[0], shift[1], response))
                
    if len(edges) == 0:
        print(f"[{output_prefix}] WARNING: No overlapping edges found! Placing diagonally.")
        X = np.arange(N) * 640.0
        Y = np.arange(N) * 640.0
    else:
        A = sp.lil_matrix((len(edges), N))
        bx = np.zeros(len(edges))
        by = np.zeros(len(edges))

        for e_idx, (i, j, dx, dy, resp) in enumerate(edges):
            w = resp # weight by confidence
            A[e_idx, i] = w
            A[e_idx, j] = -w
            bx[e_idx] = w * dx
            by[e_idx] = w * dy

        # Anchor node 0
        A_anchor = sp.lil_matrix((1, N))
        A_anchor[0, 0] = 1.0
        A = sp.vstack([A, A_anchor])
        bx = np.append(bx, 0)
        by = np.append(by, 0)

        # Solve sparse linear system for global absolute coordinates
        X = lsqr(A, bx)[0]
        Y = lsqr(A, by)[0]

    # Normalize coordinates so the top-left most image is at (0, 0)
    X -= X.min()
    Y -= Y.min()
    
    max_x = int(np.ceil(X.max() + 640))
    max_y = int(np.ceil(Y.max() + 640))
    
    # Initialize canvases with WHITE (255)
    canvas_temp = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    canvas_test = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
    
    for i in range(N):
        x = int(round(X[i]))
        y = int(round(Y[i]))
        
        # Load color images
        img_temp = cv2.imread(temp_files[i])
        test_file = temp_files[i].replace('_temp.jpg', '_test.jpg')
        if os.path.exists(test_file):
            img_test = cv2.imread(test_file)
        else:
            img_test = np.ones((640, 640, 3), dtype=np.uint8) * 255
            
        # Paste using minimum blending (darkest pixel wins, preserving circuit lines seamlessly)
        roi_temp = canvas_temp[y:y+640, x:x+640]
        canvas_temp[y:y+640, x:x+640] = np.minimum(roi_temp, img_temp)
        
        roi_test = canvas_test[y:y+640, x:x+640]
        canvas_test[y:y+640, x:x+640] = np.minimum(roi_test, img_test)

    # Note: No need to crop because X and Y start exactly at 0.
    out_temp_path = os.path.join(out_dir, f"{output_prefix}_reconstructed_normal.jpg")
    out_test_path = os.path.join(out_dir, f"{output_prefix}_reconstructed_defect.jpg")
    
    cv2.imwrite(out_temp_path, canvas_temp)
    cv2.imwrite(out_test_path, canvas_test)
    
    print(f"[{output_prefix}] Saved Reconstructed Image: {canvas_temp.shape[1]}x{canvas_temp.shape[0]} px")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Global Phase Correlation Stitcher")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process all groups sequentially")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data/merged_clusters', help="Output directory")
    
    # We keep the old parameters in argparse so scripts/run_jigsaw.bat doesn't crash
    parser.add_argument('--threshold', type=float, default=15.0, help="Ignored: Graph SLAM used")
    parser.add_argument('--edge_depth', type=int, default=2, help="Ignored: Graph SLAM used")
    parser.add_argument('--color_mode', type=str, default='rgb', help="Ignored: Graph SLAM used")
    parser.add_argument('--mutual_best', type=bool, default=True, help="Ignored: Graph SLAM used")
    parser.add_argument('--pattern_bonus', type=float, default=30.0, help="Ignored: Graph SLAM used")
    
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
