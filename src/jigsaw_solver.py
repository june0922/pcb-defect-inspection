# Jigsaw solver for DeepPCB dataset
import os
import glob
import numpy as np
from PIL import Image
import json
import argparse
import sys
import time
import datetime

def get_edges(img_array):
    """Extract top, right, bottom, left edges from a grayscale image array."""
    top = img_array[0, :].astype(float)
    right = img_array[:, -1].astype(float)
    bottom = img_array[-1, :].astype(float)
    left = img_array[:, 0].astype(float)
    return top, right, bottom, left

def solve_files(temp_files, output_prefix, args, project_root):
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not temp_files:
        print("No files to process.")
        return 0

    n = len(temp_files)
    print(f"\n{'='*50}")
    print(f"Processing {output_prefix} ({n} images)...")
    print(f"{'='*50}")

    print("Loading images and extracting edges...")
    edges_list = []
    
    # Track progress during loading
    load_start = time.time()
    for i, f in enumerate(temp_files):
        img = Image.open(f).convert('L')
        arr = np.array(img)
        edges_list.append(get_edges(arr))
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{n} images...")

    print("Computing pairwise edge distances...")
    dist_RL = np.full((n, n), np.inf)
    dist_TB = np.full((n, n), np.inf)

    # Compute distances efficiently
    # edges_list has shape (n, 4, 640)
    edges_arr = np.array(edges_list) # shape: (n, 4, 640)
    top_edges = edges_arr[:, 0, :]
    right_edges = edges_arr[:, 1, :]
    bottom_edges = edges_arr[:, 2, :]
    left_edges = edges_arr[:, 3, :]

    for i in range(n):
        # Broadcasting to compute distances from image i to all j
        diff_RL = np.abs(right_edges[i] - left_edges)
        err_RL = np.mean(diff_RL, axis=1)
        dist_RL[i, :] = err_RL
        
        diff_TB = np.abs(bottom_edges[i] - top_edges)
        err_TB = np.mean(diff_TB, axis=1)
        dist_TB[i, :] = err_TB
        
    np.fill_diagonal(dist_RL, np.inf)
    np.fill_diagonal(dist_TB, np.inf)

    print("Building clusters (Greedy Grid Assembly)...")
    unplaced = set(range(n))
    clusters = []
    
    # We maintain masked distance matrices to quickly find minimums
    dist_RL_masked = dist_RL.copy()
    dist_TB_masked = dist_TB.copy()

    solve_start = time.time()

    while unplaced:
        if len(unplaced) % 100 == 0:
            print(f"  {len(unplaced)} images remaining...")
            
        # Find the absolute best pair among all unplaced
        min_RL_idx = np.argmin(dist_RL_masked)
        min_RL_val = dist_RL_masked.flat[min_RL_idx]
        
        min_TB_idx = np.argmin(dist_TB_masked)
        min_TB_val = dist_TB_masked.flat[min_TB_idx]
        
        if min_RL_val < min_TB_val:
            best_err = min_RL_val
            start_pair = (min_RL_idx // n, min_RL_idx % n)
            start_type = 'RL'
        else:
            best_err = min_TB_val
            start_pair = (min_TB_idx // n, min_TB_idx % n)
            start_type = 'TB'

        if best_err > args.threshold:
            # No valid connections left, place remaining as singletons
            for node in list(unplaced):
                clusters.append({(0, 0): node})
                unplaced.remove(node)
                dist_RL_masked[node, :] = np.inf
                dist_RL_masked[:, node] = np.inf
                dist_TB_masked[node, :] = np.inf
                dist_TB_masked[:, node] = np.inf
            break

        i, j = start_pair
        cluster_grid = {}
        cluster_grid[(0, 0)] = i
        
        if start_type == 'RL':
            cluster_grid[(1, 0)] = j
        else:
            cluster_grid[(0, 1)] = j
            
        unplaced.remove(i)
        unplaced.remove(j)
        
        # Mask out i and j so they are not picked as new start pairs or by other unplaced nodes
        dist_RL_masked[i, :] = np.inf
        dist_RL_masked[:, i] = np.inf
        dist_TB_masked[i, :] = np.inf
        dist_TB_masked[:, i] = np.inf
        
        dist_RL_masked[j, :] = np.inf
        dist_RL_masked[:, j] = np.inf
        dist_TB_masked[j, :] = np.inf
        dist_TB_masked[:, j] = np.inf
        
        placed_nodes = {i: (0, 0), j: (1, 0) if start_type == 'RL' else (0, 1)}

        while True:
            best_grow_err = np.inf
            best_grow_info = None
            
            # Fast search for next adjacent node
            # We look at all placed_nodes and find the best unplaced neighbor
            for p_node, (px, py) in placed_nodes.items():
                # Right neighbor
                if (px + 1, py) not in cluster_grid:
                    u_node = np.argmin(dist_RL[p_node, :])
                    val = dist_RL[p_node, u_node]
                    if u_node in unplaced and val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, 1, 0)
                
                # Left neighbor
                if (px - 1, py) not in cluster_grid:
                    u_node = np.argmin(dist_RL[:, p_node])
                    val = dist_RL[u_node, p_node]
                    if u_node in unplaced and val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, -1, 0)
                
                # Bottom neighbor
                if (px, py + 1) not in cluster_grid:
                    u_node = np.argmin(dist_TB[p_node, :])
                    val = dist_TB[p_node, u_node]
                    if u_node in unplaced and val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, 0, 1)
                        
                # Top neighbor
                if (px, py - 1) not in cluster_grid:
                    u_node = np.argmin(dist_TB[:, p_node])
                    val = dist_TB[u_node, p_node]
                    if u_node in unplaced and val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, 0, -1)
                        
            if best_grow_info is None or best_grow_err > args.threshold:
                break
                
            p_node, u_node, dx, dy = best_grow_info
            px, py = placed_nodes[p_node]
            nx, ny = px + dx, py + dy
            
            cluster_grid[(nx, ny)] = u_node
            placed_nodes[u_node] = (nx, ny)
            unplaced.remove(u_node)
            
            dist_RL_masked[u_node, :] = np.inf
            dist_RL_masked[:, u_node] = np.inf
            dist_TB_masked[u_node, :] = np.inf
            dist_TB_masked[:, u_node] = np.inf

        clusters.append(cluster_grid)

    print(f"Formed {len(clusters)} clusters for {output_prefix}.")
    
    img_width, img_height = 640, 640
    
    for c_idx, cluster in enumerate(clusters):
        min_x = min(x for x, y in cluster.keys())
        min_y = min(y for x, y in cluster.keys())
        
        normalized_cluster = {}
        for (x, y), node in cluster.items():
            normalized_cluster[(x - min_x, y - min_y)] = node
            
        max_x = max(x for x, y in normalized_cluster.keys())
        max_y = max(y for x, y in normalized_cluster.keys())
        
        cols = max_x + 1
        rows = max_y + 1
        
        if len(normalized_cluster) > 1:
            print(f"Cluster {c_idx}: {len(normalized_cluster)} images, Grid: {cols}x{rows}")
        
        layout_data = []
        for (x, y), node in normalized_cluster.items():
            temp_file = temp_files[node]
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            
            if not os.path.exists(test_file):
                test_file_name = None
            else:
                test_file_name = os.path.basename(test_file)
                
            layout_data.append({
                'grid_x': int(x),
                'grid_y': int(y),
                'temp_file': os.path.basename(temp_file),
                'test_file': test_file_name,
                'original_group': os.path.basename(os.path.dirname(os.path.dirname(temp_file)))
            })
            
        layout_path = os.path.join(out_dir, f"{output_prefix}_cluster_{c_idx:03d}_layout.json")
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=4)
            
        # Ensure BOTH temp and test canvases are initialized with WHITE (255, 255, 255)
        canvas_temp = Image.new('RGB', (cols * img_width, rows * img_height), (255, 255, 255))
        canvas_test = Image.new('RGB', (cols * img_width, rows * img_height), (255, 255, 255))
        
        for (x, y), node in normalized_cluster.items():
            temp_file = temp_files[node]
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            
            im_temp = Image.open(temp_file)
            
            paste_x = x * img_width
            paste_y = y * img_height
            
            canvas_temp.paste(im_temp, (paste_x, paste_y))
            
            if os.path.exists(test_file):
                im_test = Image.open(test_file)
                canvas_test.paste(im_test, (paste_x, paste_y))
            
        out_temp_path = os.path.join(out_dir, f"{output_prefix}_cluster_{c_idx:03d}_temp_normal.jpg")
        out_test_path = os.path.join(out_dir, f"{output_prefix}_cluster_{c_idx:03d}_test_defect.jpg")
        
        canvas_temp.save(out_temp_path, quality=95)
        canvas_test.save(out_test_path, quality=95)
        
    return n

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Jigsaw Solver")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process every image together")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data', help="Output directory")
    parser.add_argument('--threshold', type=float, default=5.0, help="MAE threshold for a valid match")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(project_root, args.dataset_dir)

    start_time = time.time()
    
    if args.group == 'all':
        group_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('group') and os.path.isdir(os.path.join(dataset_path, d))])
        
        if not group_dirs:
            print(f"No group directories found in {dataset_path}")
            sys.exit(1)
            
        all_temp_files = []
        for g in group_dirs:
            g_dir = os.path.join(dataset_path, g, g.replace('group', ''))
            all_temp_files.extend(sorted(glob.glob(os.path.join(g_dir, '*_temp.jpg'))))
            
        print(f"Found {len(all_temp_files)} images across {len(group_dirs)} groups. Mixing all together...")
        
        solve_files(all_temp_files, "mixed_all_groups", args, project_root)
        
        elapsed = time.time() - start_time
        print(f"\n---> [Time Elapsed] {format_time(elapsed)}")
                
    else:
        group_name = args.group
        group_dir = os.path.join(dataset_path, group_name, group_name.replace('group', ''))
        temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
        solve_files(temp_files, group_name, args, project_root)
        elapsed = time.time() - start_time
        print(f"\n---> [Time Elapsed] {format_time(elapsed)}")
        
    print("\nAll operations completed successfully!")

if __name__ == '__main__':
    main()
