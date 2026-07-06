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
import networkx as nx

def get_match(img1, img2, S=150, P=50):
    matches = []
    
    # 1. Bottom of img1
    template = img1[640-S:640, P:640-P]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    crop = img2[loc[1]:loc[1]+S, loc[0]:loc[0]+(640-2*P)]
    mad = np.mean(cv2.absdiff(template, crop))
    matches.append((score, mad, P-loc[0], (640-S)-loc[1]))
    
    # 2. Top of img1
    template = img1[0:S, P:640-P]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    crop = img2[loc[1]:loc[1]+S, loc[0]:loc[0]+(640-2*P)]
    mad = np.mean(cv2.absdiff(template, crop))
    matches.append((score, mad, P-loc[0], -loc[1]))
    
    # 3. Right of img1
    template = img1[P:640-P, 640-S:640]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    crop = img2[loc[1]:loc[1]+(640-2*P), loc[0]:loc[0]+S]
    mad = np.mean(cv2.absdiff(template, crop))
    matches.append((score, mad, (640-S)-loc[0], P-loc[1]))
    
    # 4. Left of img1
    template = img1[P:640-P, 0:S]
    res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    crop = img2[loc[1]:loc[1]+(640-2*P), loc[0]:loc[0]+S]
    mad = np.mean(cv2.absdiff(template, crop))
    matches.append((score, mad, -loc[0], P-loc[1]))
    
    best = None
    for m in matches:
        # STRICT THRESHOLD to eliminate false positive "sliding" matches on PCB traces
        if m[0] > 0.8 and m[1] < 20:
            if best is None or m[0] > best[0]:
                best = m
    return best

def solve_group(group_dir, output_prefix, args, out_dir):
    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"No temp files found in {group_dir}")
        return

    print(f"[{output_prefix}] Reconstructing original massive PCB...")
    
    N = len(temp_files)
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in temp_files]
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    edges_info = {}
    
    print(f"[{output_prefix}] Computing strict spatial template matches...")
    for i in range(N):
        for j in range(i + 1, min(N, i + 16)):
            match = get_match(imgs[i], imgs[j])
            if match:
                score, mad, dx, dy = match
                G.add_edge(i, j)
                edges_info[(i, j)] = (dx, dy, score)
                
    cc = list(nx.connected_components(G))
    print(f"[{output_prefix}] Found {len(cc)} disjoint clusters.")
    
    clusters_coords = []
    
    for cluster_nodes in cc:
        cluster_nodes = list(cluster_nodes)
        n_nodes = len(cluster_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(cluster_nodes)}
        
        # Build local edges
        local_edges = []
        for i in cluster_nodes:
            for j in G.neighbors(i):
                if i < j:
                    dx, dy, w = edges_info[(i, j)]
                    local_edges.append((node_to_idx[i], node_to_idx[j], dx, dy, w))
                    
        if n_nodes == 1:
            X = np.array([0.0])
            Y = np.array([0.0])
        else:
            A = sp.lil_matrix((len(local_edges), n_nodes))
            bx = np.zeros(len(local_edges))
            by = np.zeros(len(local_edges))

            for e_idx, (u, v, dx, dy, w) in enumerate(local_edges):
                A[e_idx, u] = w
                A[e_idx, v] = -w
                bx[e_idx] = w * dx
                by[e_idx] = w * dy

            # Anchor node 0 of this cluster
            A_anchor = sp.lil_matrix((1, n_nodes))
            A_anchor[0, 0] = 1.0
            A = sp.vstack([A, A_anchor])
            bx = np.append(bx, 0)
            by = np.append(by, 0)

            X = lsqr(A, bx)[0]
            Y = lsqr(A, by)[0]

        X -= X.min()
        Y -= Y.min()
        
        clusters_coords.append({
            'nodes': cluster_nodes,
            'X': X,
            'Y': Y,
            'W': int(np.ceil(X.max() + 640)),
            'H': int(np.ceil(Y.max() + 640))
        })
    
    # Save each cluster as a separate image file!
    color_imgs_temp = [cv2.imread(f) for f in temp_files]
    
    for c_idx, c in enumerate(clusters_coords):
        final_w = c['W']
        final_h = c['H']
        
        print(f"[{output_prefix}] Cluster {c_idx}: Allocating {final_w}x{final_h} canvases...")
        
        canvas_temp = np.ones((final_h, final_w, 3), dtype=np.uint8) * 255
        canvas_test = np.ones((final_h, final_w, 3), dtype=np.uint8) * 255
        
        for idx, node in enumerate(c['nodes']):
            x = int(round(c['X'][idx]))
            y = int(round(c['Y'][idx]))
            
            img_temp = color_imgs_temp[node]
            test_file = temp_files[node].replace('_temp.jpg', '_test.jpg')
            if os.path.exists(test_file):
                img_test = cv2.imread(test_file)
            else:
                img_test = np.ones((640, 640, 3), dtype=np.uint8) * 255
                
            roi_temp = canvas_temp[y:y+640, x:x+640]
            canvas_temp[y:y+640, x:x+640] = np.minimum(roi_temp, img_temp)
            
            roi_test = canvas_test[y:y+640, x:x+640]
            canvas_test[y:y+640, x:x+640] = np.minimum(roi_test, img_test)

        out_temp_path = os.path.join(out_dir, f"{output_prefix}_c{c_idx}_reconstructed_normal.jpg")
        out_test_path = os.path.join(out_dir, f"{output_prefix}_c{c_idx}_reconstructed_defect.jpg")
        
        cv2.imwrite(out_temp_path, canvas_temp)
        cv2.imwrite(out_test_path, canvas_test)
        
    print(f"[{output_prefix}] Saved {len(clusters_coords)} separate rectangular images!")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="DeepPCB Strict Template Matching Stitcher with CC Packing")
    parser.add_argument('--group', type=str, default='all', help="Group name (e.g., group00041) or 'all' to process all groups sequentially")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData', help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data/merged_clusters', help="Output directory")
    
    # We keep the old parameters in argparse so scripts/run_jigsaw.bat doesn't crash
    parser.add_argument('--threshold', type=float, default=15.0, help="Ignored")
    parser.add_argument('--edge_depth', type=int, default=2, help="Ignored")
    parser.add_argument('--color_mode', type=str, default='rgb', help="Ignored")
    parser.add_argument('--mutual_best', type=bool, default=True, help="Ignored")
    parser.add_argument('--pattern_bonus', type=float, default=30.0, help="Ignored")
    
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
            
        print(f"Found {len(group_dirs)} groups. Processing sequentially...")
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
