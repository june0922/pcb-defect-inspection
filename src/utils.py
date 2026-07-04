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

    project_root = Path(env_paths["project_root"]).resolve()
    
    raw_data_path = Path(env_paths["raw_data"])
    if not raw_data_path.is_absolute():
        raw_data = (project_root / raw_data_path).resolve()
    else:
        raw_data = raw_data_path.resolve()

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

from tqdm import tqdm

class GlobalProgressCallback:
    """전체 스크립트 실행(전체 폴드, 반복 등)에 걸쳐 단일 TQDM 바를 유지하는 콜백.
    YOLO의 기본 콘솔 출력(verbose=False)을 대체하여 깔끔한 진행률 바를 제공합니다.
    """
    def __init__(self, total_epochs_per_run: int, total_runs: int = 1, run_type: str = "Fold"):
        self.total_epochs_per_run = total_epochs_per_run
        self.total_runs = total_runs
        self.run_type = run_type
        
        self.current_run = 0
        self.global_pbar = None
        self.start_time = None
        
        self.total_train_batches = 0
        self.batches_done = 0
        self.last_loss = 0.0

    def on_pretrain_routine_end(self, trainer):
        # 매 model.train() 이 시작될 때마다 (즉 새로운 폴드나 이터레이션 시작 시) 호출됨
        self.current_run += 1
        
        if self.global_pbar is None:
            train_loader = getattr(trainer, 'train_loader', None)
            batches_per_epoch = len(train_loader) if train_loader else 0
            
            self.total_train_batches = self.total_epochs_per_run * self.total_runs * batches_per_epoch
            
            # 단일 TQDM 바 생성
            self.global_pbar = tqdm(
                total=self.total_train_batches, 
                desc="Training", 
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                leave=True
            )
            self.start_time = time.time()
            
    def on_train_batch_end(self, trainer):
        if self.global_pbar is None:
            return
            
        self.batches_done += 1
        self.global_pbar.update(1)
        
        # Loss 업데이트
        loss = getattr(trainer, 'loss', None)
        if loss is not None:
            # 보통 loss 는 tensor 이므로 item() 이나 첫 번째 원소 사용
            try:
                self.last_loss = float(loss.detach().cpu().numpy()[0]) if hasattr(loss, 'detach') else float(loss)
            except:
                pass
        
        # Total ETA 계산
        elapsed = time.time() - self.start_time
        speed = elapsed / self.batches_done if self.batches_done > 0 else 0
        eta_sec = (self.total_train_batches - self.batches_done) * speed
        m, s = divmod(int(eta_sec), 60)
        h, m = divmod(m, 60)
        
        epoch = getattr(trainer, 'epoch', 0) + 1
        postfix_str = f" [{self.run_type} {self.current_run}/{self.total_runs} | Ep {epoch}/{self.total_epochs_per_run}] Loss: {self.last_loss:.3f} | Total ETA: {h:02d}:{m:02d}:{s:02d}"
        self.global_pbar.set_postfix_str(postfix_str, refresh=False)

    def on_val_end(self, validator):
        if self.global_pbar is None:
            return
            
        metrics = getattr(validator, 'metrics', None)
        if metrics is not None:
            box_metrics = getattr(metrics, 'box', None)
            if box_metrics is not None:
                map50 = getattr(box_metrics, 'map50', 0.0)
                # 현재 postfix 에 mAP50 추가 (기존 postfix 뒤에 붙임)
                current_postfix = self.global_pbar.postfix or ""
                if "mAP50" not in current_postfix:
                    new_postfix = current_postfix + f" | mAP50: {map50:.3f}"
                    self.global_pbar.set_postfix_str(new_postfix, refresh=False)
                    
    def on_train_end(self, trainer):
        # 모든 작업이 진짜로 끝났을 때 bar 를 닫음 (마지막 run일 경우)
        if self.current_run == self.total_runs and self.global_pbar is not None:
            self.global_pbar.close()

