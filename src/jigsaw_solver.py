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

def get_edges(img_array, depth):
    """
    Extract top, right, bottom, left edges from an image array.
    img_array shape: (H, W, C)
    """
    # top matches bottom. We flip top vertically so index 0 is the exact boundary
    top = img_array[:depth, :, :].astype(float)
    top = top[::-1, :, :]
    
    right = img_array[:, -depth:, :].astype(float)
    
    bottom = img_array[-depth:, :, :].astype(float)
    
    # left matches right. We flip left horizontally so index 0 is the exact boundary
    left = img_array[:, :depth, :].astype(float)
    left = left[:, ::-1, :]
    
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
    print(f"Params: depth={args.edge_depth}, color={args.color_mode}, mutual_best={args.mutual_best}, pattern_bonus={args.pattern_bonus}")
    print(f"{'='*50}")

    print("Loading images and extracting edges...")
    edges_list = []
    
    # Track progress during loading
    load_start = time.time()
    for i, f in enumerate(temp_files):
        if args.color_mode == 'rgb':
            img = Image.open(f).convert('RGB')
        else:
            img = Image.open(f).convert('L')
            
        arr = np.array(img)
        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis] # Ensure 3D shape (H, W, 1)
            
        edges_list.append(get_edges(arr, args.edge_depth))
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{n} images...")

    print("Computing pairwise edge distances...")
    dist_RL = np.full((n, n), np.inf)
    dist_TB = np.full((n, n), np.inf)

    # Compute distances efficiently
    # edges_list has shape (n, 4) where each is an array
    top_edges = np.array([e[0] for e in edges_list])
    right_edges = np.array([e[1] for e in edges_list])
    bottom_edges = np.array([e[2] for e in edges_list])
    left_edges = np.array([e[3] for e in edges_list])

    for i in range(n):
        # Broadcasting to compute distances from image i to all j
        diff_RL = np.abs(right_edges[i] - left_edges)
        err_RL = np.mean(diff_RL, axis=(1, 2, 3))
        
        if args.pattern_bonus > 0:
            intersect_RL = (right_edges[i] / 255.0) * (left_edges / 255.0)
            bonus_RL = np.mean(intersect_RL, axis=(1, 2, 3)) * args.pattern_bonus
            err_RL -= bonus_RL
            
        dist_RL[i, :] = err_RL
        
        diff_TB = np.abs(bottom_edges[i] - top_edges)
        err_TB = np.mean(diff_TB, axis=(1, 2, 3))
        
        if args.pattern_bonus > 0:
            intersect_TB = (bottom_edges[i] / 255.0) * (top_edges / 255.0)
            bonus_TB = np.mean(intersect_TB, axis=(1, 2, 3)) * args.pattern_bonus
            err_TB -= bonus_TB
            
        dist_TB[i, :] = err_TB
        
    np.fill_diagonal(dist_RL, np.inf)
    np.fill_diagonal(dist_TB, np.inf)

    print("Building clusters (Greedy Grid Assembly)...")
    unplaced = set(range(n))
    clusters = []
    
    # We maintain masked distance matrices to quickly find minimums among UNPLACED pairs
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
            
            u_nodes = list(unplaced)
            if not u_nodes:
                break
            
            # Fast search for next adjacent node
            # We look at all placed_nodes and find the best unplaced neighbor
            for p_node, (px, py) in placed_nodes.items():
                # Right neighbor (p_node -> u_node)
                if (px + 1, py) not in cluster_grid:
                    vals = dist_RL[p_node, u_nodes]
                    min_idx = np.argmin(vals)
                    u_node = u_nodes[min_idx]
                    val = vals[min_idx]
                    
                    if args.mutual_best:
                        if np.argmin(dist_RL[:, u_node]) != p_node:
                            val = np.inf
                            
                    if val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, 1, 0)
                
                # Left neighbor (u_node -> p_node)
                if (px - 1, py) not in cluster_grid:
                    vals = dist_RL[u_nodes, p_node]
                    min_idx = np.argmin(vals)
                    u_node = u_nodes[min_idx]
                    val = vals[min_idx]
                    
                    if args.mutual_best:
                        if np.argmin(dist_RL[u_node, :]) != p_node:
                            val = np.inf
                            
                    if val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, -1, 0)
                
                # Bottom neighbor (p_node -> u_node)
                if (px, py + 1) not in cluster_grid:
                    vals = dist_TB[p_node, u_nodes]
                    min_idx = np.argmin(vals)
                    u_node = u_nodes[min_idx]
                    val = vals[min_idx]
                    
                    if args.mutual_best:
                        if np.argmin(dist_TB[:, u_node]) != p_node:
                            val = np.inf
                            
                    if val < best_grow_err:
                        best_grow_err = val
                        best_grow_info = (p_node, u_node, 0, 1)
                        
                # Top neighbor (u_node -> p_node)
                if (px, py - 1) not in cluster_grid:
                    vals = dist_TB[u_nodes, p_node]
                    min_idx = np.argmin(vals)
                    u_node = u_nodes[min_idx]
                    val = vals[min_idx]
                    
                    if args.mutual_best:
                        if np.argmin(dist_TB[u_node, :]) != p_node:
                            val = np.inf
                            
                    if val < best_grow_err:
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

    print(f"Formed {len(clusters)} total clusters for {output_prefix}.")
    
    img_width, img_height = 640, 640
    
    saved_clusters = 0
    for c_idx, cluster in enumerate(clusters):
        if len(cluster) <= 1:
            # Do not save singleton clusters
            continue
            
        min_x = min(x for x, y in cluster.keys())
        min_y = min(y for x, y in cluster.keys())
        
        normalized_cluster = {}
        for (x, y), node in cluster.items():
            normalized_cluster[(x - min_x, y - min_y)] = node
            
        max_x = max(x for x, y in normalized_cluster.keys())
        max_y = max(y for x, y in normalized_cluster.keys())
        
        cols = max_x + 1
        rows = max_y + 1
        
        print(f"Saving Cluster {saved_clusters}: {len(normalized_cluster)} images, Grid: {cols}x{rows}")
        
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
            
        layout_path = os.path.join(out_dir, f"{output_prefix}_cluster_{saved_clusters:03d}_layout.json")
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
            
        out_temp_path = os.path.join(out_dir, f"{output_prefix}_cluster_{saved_clusters:03d}_temp_normal.jpg")
        out_test_path = os.path.join(out_dir, f"{output_prefix}_cluster_{saved_clusters:03d}_test_defect.jpg")
        
        canvas_temp.save(out_temp_path, quality=95)
        canvas_test.save(out_test_path, quality=95)
        
        saved_clusters += 1
        
    print(f"Total merged clusters saved: {saved_clusters}")
    return n

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Jigsaw Solver")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process every image together")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data/merged_clusters', help="Output directory")
    parser.add_argument('--threshold', type=float, default=15.0, help="MAE threshold for a valid match")
    parser.add_argument('--edge_depth', type=int, default=5, help="Thickness of the edge compared (pixels)")
    parser.add_argument('--color_mode', type=str, default='rgb', choices=['gray', 'rgb'], help="Color mode for comparison")
    parser.add_argument('--mutual_best', type=str2bool, default=True, help="Force mutual best match condition")
    parser.add_argument('--pattern_bonus', type=float, default=30.0, help="Bonus score given to matches that connect white structural patterns")
    
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
