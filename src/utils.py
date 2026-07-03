"""공통 유틸리티 — config 로드 및 환경별 경로 분기."""

import sys
from pathlib import Path
import yaml

sys.path.append( str(Path(__file__).parent))


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml 을 로드해 dict 로 반환."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> dict[str, Path]:
    """env 분기에 따라 절대 경로 dict 를 반환.

    반환 키:
        raw_data    — DeepPCB 원본 (읽기전용, 서버에서는 /shared 참조)
        processed   — YOLO 포맷으로 변환된 데이터 저장 위치
        weights     — best.pt 저장 위치
        runs        — YOLO 학습 로그/결과
    """
    env = cfg["env"]
    env_paths = cfg["paths"][env]

    raw_data = Path(env_paths["raw_data"]).resolve()
    project_root = Path(env_paths["project_root"]).resolve()

    paths = {
        "raw_data": raw_data,
        "project_root": project_root,
        "processed": project_root / "preprocessed_data",
        "weights": project_root / "weights",
        "runs": project_root / "runs",
    }

    # processed / weights / runs 는 필요 시 자동 생성 (raw_data 는 건드리지 않음)
    # 심볼릭 링크가 존재하지만 대상이 사라진 경우(끊긴 링크) mkdir이 실패하므로 사전 제거
    for key in ("processed", "weights", "runs"):
        p = paths[key]
        if p.is_symlink() and not p.exists():
            # 끊긴 심볼릭 링크 제거 (Drive 폴더가 삭제된 경우 등)
            p.unlink()
        p.mkdir(parents=True, exist_ok=True)

    return paths


import time

class TotalETACallback:
    """학습 진행 속도와 전체 남은 시간(ETA)을 계산하여 YOLO progress bar에 표시하는 커스텀 콜백."""
    def __init__(self):
        self.start_time = None

    def on_train_epoch_start(self, trainer):
        if self.start_time is None:
            self.start_time = time.time()
        
        # tqdm bar_format 에 postfix 필드가 없으면 추가
        pbar = getattr(trainer, 'pbar', None)
        if pbar is not None:
            if '{postfix}' not in getattr(pbar, 'bar_format', ''):
                pbar.bar_format = getattr(pbar, 'bar_format', '') + ' {postfix}'

    def on_train_batch_end(self, trainer):
        if self.start_time is None:
            return
            
        epochs = getattr(trainer, 'epochs', 0)
        epoch = getattr(trainer, 'epoch', 0)
        
        train_loader = getattr(trainer, 'train_loader', None)
        if not train_loader: return
        batches_per_epoch = len(train_loader)
        batch_i = getattr(trainer, 'batch_i', 0)
        
        total_batches = epochs * batches_per_epoch
        batches_done = epoch * batches_per_epoch + batch_i + 1
        
        elapsed = time.time() - self.start_time
        if batches_done > 0:
            speed = elapsed / batches_done  # seconds per batch
            eta_seconds = (total_batches - batches_done) * speed
            
            m, s = divmod(int(eta_seconds), 60)
            h, m = divmod(m, 60)
            eta_str = f"[Total ETA: {h:02d}:{m:02d}:{s:02d} | {speed:.2f}s/it]"
            
            pbar = getattr(trainer, 'pbar', None)
            if pbar is not None:
                pbar.set_postfix_str(eta_str, refresh=True)
