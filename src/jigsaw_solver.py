# PCB 이미지 조각을 정밀 매칭하여 원본 크기로 복원하는 Jigsaw Solver

import os
import glob
import json
import numpy as np
import argparse
import sys
import time
import datetime
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import networkx as nx

IMG_SIZE = 640
SEARCH_WINDOW = 50
MATCH_SCORE_THRESHOLD = 0.80
OVERLAP_MISMATCH_THRESHOLD = 0.03
MIN_OVERLAP_PX = 30
TEMPLATE_STRIP = 150
TEMPLATE_PAD = 40


def validate_overlap(img_a, img_b, dx, dy):
    """겹침 영역의 이진 XOR 비교로 매칭 품질을 검증한다.

    img_a가 (0,0), img_b가 (dx,dy)에 놓였을 때
    실제 겹치는 픽셀의 불일치 비율을 반환한다.
    """
    H = W = IMG_SIZE

    ox1, oy1 = max(0, dx), max(0, dy)
    ox2, oy2 = min(W, dx + W), min(H, dy + H)
    ow, oh = ox2 - ox1, oy2 - oy1

    if ow < MIN_OVERLAP_PX or oh < MIN_OVERLAP_PX:
        return False, 1.0
    if ow * oh > H * W * 0.85:
        return False, 1.0

    a_patch = img_a[oy1:oy2, ox1:ox2]
    b_patch = img_b[(oy1 - dy):(oy2 - dy), (ox1 - dx):(ox2 - dx)]

    if a_patch.shape != b_patch.shape or a_patch.size == 0:
        return False, 1.0

    _, a_bin = cv2.threshold(a_patch, 128, 255, cv2.THRESH_BINARY)
    _, b_bin = cv2.threshold(b_patch, 128, 255, cv2.THRESH_BINARY)
    mismatch = np.count_nonzero(cv2.bitwise_xor(a_bin, b_bin)) / a_bin.size

    return mismatch < OVERLAP_MISMATCH_THRESHOLD, mismatch


def find_overlap(img_a, img_b):
    """img_a와 img_b 사이의 겹침 오프셋을 탐색한다.

    4방향 가장자리 strip을 템플릿으로 매칭한 뒤
    겹침 영역 XOR 검증을 통과한 최고 점수 후보를 반환한다.

    Returns:
        (dx, dy, score) 또는 None
        img_a가 (0,0)일 때 img_b의 캔버스 위치가 (dx, dy).
    """
    H = W = IMG_SIZE
    S = TEMPLATE_STRIP
    P = TEMPLATE_PAD
    best = None

    # (row_start, row_end, col_start, col_end) — img_a에서 잘라낼 strip
    strips = [
        (H - S, H, P, W - P),      # 하단 strip
        (0,     S, P, W - P),       # 상단 strip
        (P, H - P, W - S, W),      # 우측 strip
        (P, H - P, 0,     S),      # 좌측 strip
    ]

    for r1, r2, c1, c2 in strips:
        template = img_a[r1:r2, c1:c2]

        # 특징이 부족한 템플릿(거의 흰색)은 신뢰도가 낮으므로 건너뛴다
        if np.count_nonzero(template < 128) / template.size < 0.02:
            continue

        res = cv2.matchTemplate(img_b, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)

        if score < MATCH_SCORE_THRESHOLD:
            continue

        # img_a의 템플릿 원점 (c1, r1)이 img_b의 (loc[0], loc[1])에 대응
        # → img_b의 캔버스 위치 = (c1 - loc[0], r1 - loc[1])
        dx = c1 - loc[0]
        dy = r1 - loc[1]

        valid, mismatch = validate_overlap(img_a, img_b, dx, dy)
        if not valid:
            continue

        combined = score * (1.0 - mismatch)
        if best is None or combined > best[2]:
            best = (dx, dy, combined)

    return best


def solve_positions(cluster_nodes, edges_info):
    """최소자승법으로 클러스터 내 각 타일의 캔버스 좌표를 결정한다.

    edges_info[(i,j)]는 img_j가 img_i 대비 (dx,dy) 위치에 있음을 뜻한다.
    """
    n = len(cluster_nodes)
    node_idx = {node: i for i, node in enumerate(cluster_nodes)}

    local_edges = []
    for i_node in cluster_nodes:
        for j_node in cluster_nodes:
            if i_node < j_node and (i_node, j_node) in edges_info:
                dx, dy, sc = edges_info[(i_node, j_node)]
                local_edges.append((node_idx[i_node], node_idx[j_node], dx, dy, sc))

    if n == 1:
        return np.array([0.0]), np.array([0.0])

    m = len(local_edges)
    A = sp.lil_matrix((m + 1, n))
    bx = np.zeros(m + 1)
    by = np.zeros(m + 1)

    for e, (u, v, dx, dy, w) in enumerate(local_edges):
        # 올바른 방정식: pos[v] - pos[u] = (dx, dy)
        A[e, u] = -w
        A[e, v] = w
        bx[e] = w * dx
        by[e] = w * dy

    # 앵커: 첫 번째 노드를 원점에 고정
    A[m, 0] = 1.0

    A_csr = A.tocsr()
    X = lsqr(A_csr, bx)[0]
    Y = lsqr(A_csr, by)[0]

    # 모든 좌표를 양수로 이동
    X -= X.min()
    Y -= Y.min()
    return X, Y


def render_canvas(cluster_nodes, X, Y, image_loader):
    """타일 이미지를 캔버스 위에 평균 블렌딩으로 합성한다."""
    cw = int(np.ceil(X.max())) + IMG_SIZE
    ch = int(np.ceil(Y.max())) + IMG_SIZE

    canvas_sum = np.zeros((ch, cw, 3), dtype=np.float32)
    canvas_cnt = np.zeros((ch, cw, 1), dtype=np.float32)

    for idx, node in enumerate(cluster_nodes):
        x = int(round(X[idx]))
        y = int(round(Y[idx]))
        img = image_loader(node)
        if img is None:
            continue
        canvas_sum[y:y + IMG_SIZE, x:x + IMG_SIZE] += img.astype(np.float32)
        canvas_cnt[y:y + IMG_SIZE, x:x + IMG_SIZE] += 1

    safe_cnt = np.maximum(canvas_cnt, 1)
    result = np.where(canvas_cnt > 0, canvas_sum / safe_cnt, 255)
    return np.clip(result, 0, 255).astype(np.uint8), cw, ch


def solve_group(group_dir, group_name, out_dir):
    """하나의 그룹을 처리하여 복원 이미지와 메타데이터를 생성한다."""
    temp_files = sorted(glob.glob(os.path.join(group_dir, '*_temp.jpg')))
    if not temp_files:
        print(f"  No temp files found in {group_dir}")
        return

    N = len(temp_files)
    tile_ids = [os.path.basename(f).replace('_temp.jpg', '') for f in temp_files]
    print(f"  [{group_name}] {N} images loaded.")

    # 그레이스케일 이미지 로드 (매칭용)
    imgs = []
    for f in temp_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            imgs.append(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255)
        else:
            imgs.append(img)

    # ─── Pass 1: 인접 인덱스 매칭 ───
    G = nx.Graph()
    G.add_nodes_from(range(N))
    edges_info = {}
    matched = 0
    window = min(SEARCH_WINDOW, N - 1)

    print(f"  [{group_name}] Pass 1: window={window} matching...")
    for i in range(N):
        for j in range(i + 1, min(N, i + window + 1)):
            result = find_overlap(imgs[i], imgs[j])
            if result:
                dx, dy, score = result
                G.add_edge(i, j)
                edges_info[(i, j)] = (dx, dy, score)
                matched += 1

    # ─── Pass 2: 클러스터 간 보완 매칭 ───
    components = list(nx.connected_components(G))
    if len(components) > 1 and len(components) < 80:
        cluster_list = [sorted(c) for c in components]
        print(f"  [{group_name}] Pass 2: inter-cluster matching "
              f"({len(cluster_list)} clusters)...")
        for ci in range(len(cluster_list)):
            for cj in range(ci + 1, len(cluster_list)):
                samples_i = list(set(cluster_list[ci][:3] + cluster_list[ci][-3:]))
                samples_j = list(set(cluster_list[cj][:3] + cluster_list[cj][-3:]))
                found = False
                for si in samples_i:
                    if found:
                        break
                    for sj in samples_j:
                        a, b = min(si, sj), max(si, sj)
                        if (a, b) in edges_info:
                            continue
                        result = find_overlap(imgs[a], imgs[b])
                        if result:
                            dx_r, dy_r, score_r = result
                            edges_info[(a, b)] = (dx_r, dy_r, score_r)
                            G.add_edge(a, b)
                            matched += 1
                            found = True
                            break

    # 최종 컴포넌트 분리
    components = list(nx.connected_components(G))
    clusters = sorted(
        [sorted(c) for c in components if len(c) >= 2],
        key=lambda c: c[0]
    )
    orphans = sorted([list(c)[0] for c in components if len(c) == 1])

    print(f"  [{group_name}] {matched} overlaps → "
          f"{len(clusters)} clusters, {len(orphans)} orphans (skipped).")

    if not clusters:
        return

    # ─── 렌더링 및 저장 ───
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata = {
        "group": group_name,
        "source_dir": os.path.relpath(group_dir, project_root).replace('\\', '/'),
        "clusters": [],
        "orphans": [tile_ids[n] for n in orphans]
    }

    for c_idx, cluster_nodes in enumerate(clusters):
        X, Y = solve_positions(cluster_nodes, edges_info)

        # 정상(temp) 캔버스
        def load_temp(node):
            return cv2.imread(temp_files[node])
        result_n, cw, ch = render_canvas(cluster_nodes, X, Y, load_temp)

        # 결함(test) 캔버스
        def load_test(node):
            test_path = temp_files[node].replace('_temp.jpg', '_test.jpg')
            if os.path.exists(test_path):
                return cv2.imread(test_path)
            return np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
        result_d, _, _ = render_canvas(cluster_nodes, X, Y, load_test)

        out_n = f"{group_name}_c{c_idx}_normal.jpg"
        out_d = f"{group_name}_c{c_idx}_defect.jpg"
        cv2.imwrite(os.path.join(out_dir, out_n), result_n,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(out_dir, out_d), result_d,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 타일 및 엣지 메타데이터
        tile_info = []
        for idx, node in enumerate(cluster_nodes):
            test_basename = os.path.basename(
                temp_files[node].replace('_temp.jpg', '_test.jpg'))
            tile_info.append({
                "id": tile_ids[node],
                "temp_file": os.path.basename(temp_files[node]),
                "test_file": test_basename,
                "canvas_x": int(round(X[idx])),
                "canvas_y": int(round(Y[idx]))
            })

        edge_records = []
        for i_n in cluster_nodes:
            for j_n in cluster_nodes:
                if i_n < j_n and (i_n, j_n) in edges_info:
                    dx, dy, sc = edges_info[(i_n, j_n)]
                    edge_records.append({
                        "from": tile_ids[i_n],
                        "to": tile_ids[j_n],
                        "dx": int(round(dx)),
                        "dy": int(round(dy)),
                        "score": round(float(sc), 4)
                    })

        metadata["clusters"].append({
            "cluster_id": c_idx,
            "output_normal": out_n,
            "output_defect": out_d,
            "canvas_width": cw,
            "canvas_height": ch,
            "tile_count": len(cluster_nodes),
            "tiles": tile_info,
            "edges": edge_records
        })
        print(f"    Cluster {c_idx}: {len(cluster_nodes)} tiles "
              f"→ {cw}x{ch}px")

    meta_path = os.path.join(out_dir, f"{group_name}_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  [{group_name}] Saved {len(clusters)} clusters + metadata.")


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def main():
    parser = argparse.ArgumentParser(
        description="DeepPCB Jigsaw Reconstruction")
    parser.add_argument('--group', type=str, default='all',
                        help="Group name (e.g., group00041) or 'all'")
    parser.add_argument('--dataset_dir', type=str, default='dataset/PCBData',
                        help="Base dataset directory")
    parser.add_argument('--output_dir', type=str, default='recovered_data',
                        help="Output directory")

    # 하위 호환: 기존 스크립트에서 전달하던 무시 인자
    parser.add_argument('--threshold', type=float, default=15.0, help=argparse.SUPPRESS)
    parser.add_argument('--edge_depth', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--color_mode', type=str, default='rgb', help=argparse.SUPPRESS)
    parser.add_argument('--mutual_best', type=bool, default=True, help=argparse.SUPPRESS)
    parser.add_argument('--pattern_bonus', type=float, default=30.0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(project_root, args.dataset_dir)
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    if args.group == 'all':
        groups = sorted([
            d for d in os.listdir(dataset_path)
            if d.startswith('group')
            and os.path.isdir(os.path.join(dataset_path, d))
        ])
        if not groups:
            print(f"No group directories in {dataset_path}")
            sys.exit(1)

        print(f"Processing {len(groups)} groups...")
        print("=" * 60)
        for g in groups:
            g_start = time.time()
            g_dir = os.path.join(dataset_path, g, g.replace('group', ''))
            solve_group(g_dir, g, out_dir)
            print(f"  [{g}] elapsed: {format_time(time.time() - g_start)}")
            print("-" * 40)
    else:
        g_dir = os.path.join(dataset_path, args.group,
                             args.group.replace('group', ''))
        solve_group(g_dir, args.group, out_dir)

    elapsed = time.time() - start
    print(f"\n---> Total: {format_time(elapsed)}")
    print("All operations completed successfully!")


if __name__ == '__main__':
    main()
