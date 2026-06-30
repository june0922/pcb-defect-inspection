"""EDA 시각화 + 검사 결과 주석 이미지 생성.

사용:
    from src.visualize import plot_class_distribution, draw_inspection_result
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( str(Path(__file__).parent))

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]

# 판정별 BGR 색상
VERDICT_COLOR = {
    "OK": (0, 200, 0),       # 초록
    "NG": (0, 0, 220),       # 빨강
    "REVIEW": (0, 140, 255), # 주황
}


def plot_class_distribution(label_dir: Path, save_path: Path | None = None) -> None:
    """YOLO 라벨 디렉터리에서 클래스 분포 막대 그래프를 그린다.

    # TODO(개선): 클래스 불균형 수치를 팀 노트에 기록
    """
    counts = {cls: 0 for cls in CLASSES}
    for txt in label_dir.rglob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                cls_id = int(parts[0])
                if 0 <= cls_id < len(CLASSES):
                    counts[CLASSES[cls_id]] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values())
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[visualize] 클래스 분포 저장: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_bbox_histogram(label_dir: Path, img_size: int = 640, save_path: Path | None = None) -> None:
    """YOLO 라벨에서 bbox 너비/높이 히스토그램을 그린다.

    결함 크기 분포 확인 → anchor 설정 참고 자료로 활용.
    # TODO(개선): 클래스별 bbox 크기 분포 분리 출력
    """
    widths: list[float] = []
    heights: list[float] = []
    for txt in label_dir.rglob("*.txt"):
        for line in txt.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                widths.append(float(parts[3]) * img_size)
                heights.append(float(parts[4]) * img_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(widths, bins=30)
    axes[0].set_title("BBox Width Distribution (px)")
    axes[0].set_xlabel("Width (px)")
    axes[1].hist(heights, bins=30)
    axes[1].set_title("BBox Height Distribution (px)")
    axes[1].set_xlabel("Height (px)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[visualize] bbox 히스토그램 저장: {save_path}")
    else:
        plt.show()
    plt.close()


def draw_inspection_result(image_path: str, inspection: dict, save_path: str | None = None) -> np.ndarray:
    """inspect_image() 결과를 이미지 위에 주석으로 그린다.

    판정별 색상: OK=초록, NG=빨강, REVIEW=주황

    Args:
        image_path : 원본 이미지 경로
        inspection : inspect_image() 반환 dict
        save_path  : 저장 경로 (None 이면 반환만)

    Returns:
        주석이 그려진 BGR 이미지 (np.ndarray)

    # TODO: 박스 두께, 폰트 크기 파라미터화
    # TODO: conf 값을 박스 옆에 표시
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    verdict = inspection["verdict"]
    color = VERDICT_COLOR.get(verdict, (128, 128, 128))

    # 결함 바운딩 박스 그리기
    for defect in inspection["defects"]:
        x1, y1, x2, y2 = [int(v) for v in defect["bbox"]]
        box_color = VERDICT_COLOR.get("REVIEW", color) if defect in inspection["review"] else color
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        label = f"{defect['class_name']} {defect['conf']:.2f}"
        # TODO: 텍스트 배경 박스 추가해 가독성 개선
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # 판정 배너 (상단)
    banner_text = f"[{verdict}]  결함: {inspection['defect_count']}개"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), color, -1)
    cv2.putText(img, banner_text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"[visualize] 결과 이미지 저장: {save_path}")

    return img
