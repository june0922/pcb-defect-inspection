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

def solve_group(group_name, args, project_root):
    group_dir = os.path.join(project_root, args.dataset_dir, group_name, group_name.replace('group', ''))
    
    if not os.path.exists(group_dir):
        print(f"Directory not found: {group_dir}")
        return 0

    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"No _temp.jpg files found in {group_dir}")
        return 0

    print(f"\n{'='*50}")
    print(f"Processing {group_name} ({len(temp_files)} images)...")
    print(f"{'='*50}")

    print("Loading images and extracting edges...")
    edges_list = []
    for f in temp_files:
        img = Image.open(f).convert('L')
        arr = np.array(img)
        edges_list.append(get_edges(arr))

    n = len(temp_files)
    
    print("Computing pairwise edge distances...")
    dist_RL = np.full((n, n), np.inf)
    dist_TB = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(n):
            if i == j: continue
            err_RL = np.mean(np.abs(edges_list[i][1] - edges_list[j][3]))
            dist_RL[i, j] = err_RL
            
            err_TB = np.mean(np.abs(edges_list[i][2] - edges_list[j][0]))
            dist_TB[i, j] = err_TB

    print("Building clusters (Greedy Grid Assembly)...")
    unplaced = set(range(n))
    clusters = []

    while unplaced:
        best_err = np.inf
        start_pair = None
        start_type = None 
        
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
            for node in list(unplaced):
                clusters.append({(0, 0): node})
                unplaced.remove(node)
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
        
        placed_nodes = {i: (0, 0), j: (1, 0) if start_type == 'RL' else (0, 1)}

        while True:
            best_grow_err = np.inf
            best_grow_info = None
            
            for p_node, (px, py) in placed_nodes.items():
                for u_node in unplaced:
                    if (px + 1, py) not in cluster_grid and dist_RL[p_node, u_node] < best_grow_err:
                        best_grow_err = dist_RL[p_node, u_node]
                        best_grow_info = (p_node, u_node, 1, 0)
                    
                    if (px - 1, py) not in cluster_grid and dist_RL[u_node, p_node] < best_grow_err:
                        best_grow_err = dist_RL[u_node, p_node]
                        best_grow_info = (p_node, u_node, -1, 0)
                        
                    if (px, py + 1) not in cluster_grid and dist_TB[p_node, u_node] < best_grow_err:
                        best_grow_err = dist_TB[p_node, u_node]
                        best_grow_info = (p_node, u_node, 0, 1)
                        
                    if (px, py - 1) not in cluster_grid and dist_TB[u_node, p_node] < best_grow_err:
                        best_grow_err = dist_TB[u_node, p_node]
                        best_grow_info = (p_node, u_node, 0, -1)
                        
            if best_grow_info is None or best_grow_err > args.threshold:
                break
                
            p_node, u_node, dx, dy = best_grow_info
            px, py = placed_nodes[p_node]
            nx, ny = px + dx, py + dy
            
            cluster_grid[(nx, ny)] = u_node
            placed_nodes[u_node] = (nx, ny)
            unplaced.remove(u_node)

        clusters.append(cluster_grid)

    print(f"Formed {len(clusters)} clusters for {group_name}.")
    
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
            
            # Check if test_file really exists, otherwise mark as null in json
            if not os.path.exists(test_file):
                test_file_name = None
            else:
                test_file_name = os.path.basename(test_file)
                
            layout_data.append({
                'grid_x': x,
                'grid_y': y,
                'temp_file': os.path.basename(temp_file),
                'test_file': test_file_name
            })
            
        layout_path = os.path.join(out_dir, f"{group_name}_cluster_{c_idx:02d}_layout.json")
        with open(layout_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=4)
            
        canvas_temp = Image.new('RGB', (cols * img_width, rows * img_height))
        # Defect image canvas is initialized black. We do not mix with temp images.
        canvas_test = Image.new('RGB', (cols * img_width, rows * img_height), (0, 0, 0))
        
        for (x, y), node in normalized_cluster.items():
            temp_file = temp_files[node]
            test_file = temp_file.replace('_temp.jpg', '_test.jpg')
            
            im_temp = Image.open(temp_file)
            
            paste_x = x * img_width
            paste_y = y * img_height
            
            canvas_temp.paste(im_temp, (paste_x, paste_y))
            
            # Paste test image ONLY if it exists to strictly prevent mixing with temp image
            if os.path.exists(test_file):
                im_test = Image.open(test_file)
                canvas_test.paste(im_test, (paste_x, paste_y))
            
        out_temp_path = os.path.join(out_dir, f"{group_name}_cluster_{c_idx:02d}_temp_normal.jpg")
        out_test_path = os.path.join(out_dir, f"{group_name}_cluster_{c_idx:02d}_test_defect.jpg")
        
        canvas_temp.save(out_temp_path, quality=95)
        canvas_test.save(out_test_path, quality=95)
        
    return n

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Jigsaw Solver")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process every group")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data', help="Output directory")
    parser.add_argument('--threshold', type=float, default=5.0, help="MAE threshold for a valid match")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    start_time = time.time()
    
    if args.group == 'all':
        dataset_path = os.path.join(project_root, args.dataset_dir)
        group_dirs = sorted([d for d in os.listdir(dataset_path) if d.startswith('group') and os.path.isdir(os.path.join(dataset_path, d))])
        
        if not group_dirs:
            print(f"No group directories found in {dataset_path}")
            sys.exit(1)
            
        # Calculate total images for ETA prediction
        total_images = 0
        for g in group_dirs:
            g_dir = os.path.join(dataset_path, g, g.replace('group', ''))
            total_images += len(glob.glob(os.path.join(g_dir, '*_temp.jpg')))
            
        print(f"Found {len(group_dirs)} groups with a total of {total_images} images to process.")
        
        processed_images = 0
        for g in group_dirs:
            count = solve_group(g, args, project_root)
            processed_images += count
            
            elapsed = time.time() - start_time
            if processed_images > 0 and processed_images < total_images:
                time_per_image = elapsed / processed_images
                remaining_images = total_images - processed_images
                eta_seconds = time_per_image * remaining_images
                print(f"\n---> [Progress] {processed_images}/{total_images} images processed ({(processed_images/total_images)*100:.1f}%)")
                print(f"---> [Time Elapsed] {format_time(elapsed)}")
                print(f"---> [ETA Remaining] {format_time(eta_seconds)}")
                
    else:
        solve_group(args.group, args, project_root)
        elapsed = time.time() - start_time
        print(f"\n---> [Time Elapsed] {format_time(elapsed)}")
        
    print("\nAll operations completed successfully!")

if __name__ == '__main__':
    main()
