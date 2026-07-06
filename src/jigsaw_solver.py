# Jigsaw solver for DeepPCB dataset
import os
import glob
import numpy as np
from PIL import Image
import json
import argparse
import sys

def get_edges(img_array):
    """Extract top, right, bottom, left edges from a grayscale image array."""
    # edges are 1D arrays
    top = img_array[0, :].astype(float)
    right = img_array[:, -1].astype(float)
    bottom = img_array[-1, :].astype(float)
    left = img_array[:, 0].astype(float)
    return top, right, bottom, left

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Jigsaw Solver")
    parser.add_argument('--group', type=str, required=True, help="Group name, e.g., group00041")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data', help="Output directory")
    parser.add_argument('--threshold', type=float, default=5.0, help="MAE threshold for a valid match")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    group_dir = os.path.join(project_root, args.dataset_dir, args.group, args.group.replace('group', ''))
    
    if not os.path.exists(group_dir):
        print(f"Directory not found: {group_dir}")
        sys.exit(1)

    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"No _temp.jpg files found in {group_dir}")
        sys.exit(1)

    print(f"Found {len(temp_files)} images for {args.group}.")

    # Load images and extract edges
    print("Loading images and extracting edges...")
    edges_list = []
    for f in temp_files:
        img = Image.open(f).convert('L')
        arr = np.array(img)
        edges_list.append(get_edges(arr))

    n = len(temp_files)
    
    # Precompute pairwise distances
    print("Computing pairwise edge distances...")
    # dist_RL[i, j] = error of i's Right edge attached to j's Left edge
    dist_RL = np.full((n, n), np.inf)
    # dist_TB[i, j] = error of i's Bottom edge attached to j's Top edge
    dist_TB = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(n):
            if i == j: continue
            # RL match
            err_RL = np.mean(np.abs(edges_list[i][1] - edges_list[j][3]))
            dist_RL[i, j] = err_RL
            
            # TB match
            err_TB = np.mean(np.abs(edges_list[i][2] - edges_list[j][0]))
            dist_TB[i, j] = err_TB

    print("Building clusters (Greedy Grid Assembly)...")
    unplaced = set(range(n))
    clusters = []

    while unplaced:
        # Start a new cluster
        # Find the absolute best match among unplaced nodes to start
        best_err = np.inf
        start_pair = None
        start_type = None # 'RL' or 'TB'
        
        for i in unplaced:
            for j in unplaced:
                if i == j: continue
                if dist_RL[i, j] < best_err:
                    best_err = dist_RL[i, j]
                    start_pair = (i, j)
                    start_type = 'RL'
                if dist_TB[i, j] < best_err:
                    best_err = dist_TB[i, j]
                    start_pair = (i, j)
                    start_type = 'TB'

        if start_pair is None or best_err > args.threshold:
            # No valid matches left, remaining nodes are isolated
            for node in list(unplaced):
                clusters.append({(0, 0): node})
                unplaced.remove(node)
            break

        # Initialize cluster with the best pair
        i, j = start_pair
        cluster_grid = {} # (x, y) -> node_idx
        cluster_grid[(0, 0)] = i
        
        if start_type == 'RL':
            cluster_grid[(1, 0)] = j
        else: # TB
            cluster_grid[(0, 1)] = j
            
        unplaced.remove(i)
        unplaced.remove(j)
        
        placed_nodes = {i: (0, 0), j: (1, 0) if start_type == 'RL' else (0, 1)}

        # Iteratively grow the cluster
        while True:
            best_grow_err = np.inf
            best_grow_info = None # (placed_node, unplaced_node, dx, dy)
            
            for p_node, (px, py) in placed_nodes.items():
                for u_node in unplaced:
                    # Check RL: p_node's Right to u_node's Left (u_node is at px+1, py)
                    if (px + 1, py) not in cluster_grid and dist_RL[p_node, u_node] < best_grow_err:
                        best_grow_err = dist_RL[p_node, u_node]
                        best_grow_info = (p_node, u_node, 1, 0)
                    
                    # Check LR: u_node's Right to p_node's Left (u_node is at px-1, py)
                    if (px - 1, py) not in cluster_grid and dist_RL[u_node, p_node] < best_grow_err:
                        best_grow_err = dist_RL[u_node, p_node]
                        best_grow_info = (p_node, u_node, -1, 0)
                        
                    # Check TB: p_node's Bottom to u_node's Top (u_node is at px, py+1)
                    if (px, py + 1) not in cluster_grid and dist_TB[p_node, u_node] < best_grow_err:
                        best_grow_err = dist_TB[p_node, u_node]
                        best_grow_info = (p_node, u_node, 0, 1)
                        
                    # Check BT: u_node's Bottom to p_node's Top (u_node is at px, py-1)
                    if (px, py - 1) not in cluster_grid and dist_TB[u_node, p_node] < best_grow_err:
                        best_grow_err = dist_TB[u_node, p_node]
                        best_grow_info = (p_node, u_node, 0, -1)
                        
            if best_grow_info is None or best_grow_err > args.threshold:
                break # Cluster cannot grow further
                
            p_node, u_node, dx, dy = best_grow_info
            px, py = placed_nodes[p_node]
            nx, ny = px + dx, py + dy
            
            # Place the node
            cluster_grid[(nx, ny)] = u_node
            placed_nodes[u_node] = (nx, ny)
            unplaced.remove(u_node)

        clusters.append(cluster_grid)

    print(f"Formed {len(clusters)} clusters.")
    
    img_width, img_height = 640, 640
    
    for c_idx, cluster in enumerate(clusters):
        # Normalize coordinates so min_x = 0, min_y = 0
        min_x = min(x for x, y in cluster.keys())
        min_y = min(y for x, y in cluster.keys())
        
        normalized_cluster = {}
        for (x, y), node in cluster.items():
            normalized_cluster[(x - min_x, y - min_y)] = node
            
        max_x = max(x for x, y in normalized_cluster.keys())
        max_y = max(y for x, y in normalized_cluster.keys())
        
        cols = max_x + 1
        rows = max_y + 1
        
        print(f"Cluster {c_idx}: {len(normalized_cluster)} images, Grid: {cols}x{rows}")
        
        # 1. Save layout JSON
        layout_data = []
        for (x, y), node in normalized_cluster.items():
            temp_file = temp_files[node]
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            layout_data.append({
                'grid_x': x,
                'grid_y': y,
                'temp_file': os.path.basename(temp_file),
                'test_file': os.path.basename(test_file)
            })
            
        layout_path = os.path.join(out_dir, f"{args.group}_cluster_{c_idx:02d}_layout.json")
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=4)
            
        # 2. Reconstruct temp image (normal)
        canvas_temp = Image.new('RGB', (cols * img_width, rows * img_height))
        # 3. Reconstruct test image (defect)
        canvas_test = Image.new('RGB', (cols * img_width, rows * img_height))
        
        for (x, y), node in normalized_cluster.items():
            temp_file = temp_files[node]
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            
            im_temp = Image.open(temp_file)
            im_test = Image.open(test_file)
            
            paste_x = x * img_width
            paste_y = y * img_height
            
            canvas_temp.paste(im_temp, (paste_x, paste_y))
            canvas_test.paste(im_test, (paste_x, paste_y))
            
        out_temp_path = os.path.join(out_dir, f"{args.group}_cluster_{c_idx:02d}_temp_normal.jpg")
        out_test_path = os.path.join(out_dir, f"{args.group}_cluster_{c_idx:02d}_test_defect.jpg")
        
        canvas_temp.save(out_temp_path, quality=95)
        canvas_test.save(out_test_path, quality=95)
        print(f"  Saved images to {os.path.basename(out_temp_path)} and {os.path.basename(out_test_path)}")

if __name__ == '__main__':
    main()
