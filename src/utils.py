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
    def __init__(
        self,
        total_epochs_per_run: int,
        total_runs: int = 1,
        run_type: str = "Fold",
        starting_run: int = 0,
        fold_train_sizes: list[int] | None = None,
    ):
        self.total_epochs_per_run = total_epochs_per_run
        self.total_runs = total_runs
        self.run_type = run_type

        self.current_run = starting_run
        self.global_pbar = None
        self.start_time = None

        self.total_train_batches = 0
        self.batches_done = 0
        self.batches_this_session = 0  # ETA 속도 계산용 (프로세스 재시작 시 0부터)
        self.last_loss = 0.0

        # fold 별 학습 이미지 수 (0-indexed, skf.split 으로 사전 계산됨).
        # batch=-1(오토배치) 환경에서 아직 시작하지 않은/완료된 fold의 배치 수를
        # "현재 fold와 동일한 batches_per_epoch" 로 근사하지 않고, fold 별 실제
        # 이미지 수 기반으로 정확히 추정하기 위해 사용.
        self.fold_train_sizes = fold_train_sizes
        self.last_batch_size = None

    def on_pretrain_routine_end(self, trainer):
        # 매 model.train() 이 시작될 때마다 (즉 새로운 폴드나 이터레이션 시작 시) 호출됨
        self.current_run += 1
        cur_fold_idx = self.current_run - 1  # 0-indexed 현재 fold 번호

        train_loader = getattr(trainer, 'train_loader', None)
        batches_per_epoch = len(train_loader) if train_loader else 0
        start_epoch = getattr(trainer, 'start_epoch', 0)

        # 실제 배치 크기 확보. YOLO의 batch=-1(오토배치) 설정 시 trainer.batch_size 에
        # 실제 확정된 값이 담기므로, 다른 fold의 배치 수를 추정할 때 재사용한다.
        batch_size = getattr(trainer, 'batch_size', None)
        if not batch_size and batches_per_epoch and self.fold_train_sizes:
            if 0 <= cur_fold_idx < len(self.fold_train_sizes):
                batch_size = max(1, round(self.fold_train_sizes[cur_fold_idx] / batches_per_epoch))
        if batch_size:
            self.last_batch_size = batch_size
        else:
            batch_size = self.last_batch_size

        if self.global_pbar is None:
            # 전체 K-Fold 작업의 그랜드 토탈(모든 fold × epoch)을 최초 1회만 계산해
            # 고정한다. fold가 바뀔 때마다 분모를 다시 추정하던 기존 방식은 완료된
            # fold의 배치 수가 분모에서 누락되거나(과소) fold마다 total이 흔들리는
            # 문제가 있었다.
            if self.fold_train_sizes and batch_size:
                # fold 별 실제 학습 이미지 수 기반으로 정확히 추정 (fold마다 split 크기가 다름)
                bpe_per_fold = [-(-size // batch_size) for size in self.fold_train_sizes]
            else:
                # fold 별 크기 정보가 없으면 현재 fold의 batches_per_epoch로 근사 (기존 방식)
                bpe_per_fold = [batches_per_epoch] * self.total_runs

            self.total_train_batches = sum(bpe_per_fold) * self.total_epochs_per_run

            # --resume 로 스크립트가 새 프로세스로 재시작된 경우, 분자(batches_done)를
            # 0으로 두면 실제로는 진행 중이어도 (완료된 fold + 남은 fold 전체를 포함하는)
            # 그랜드 토탈에 비해 한동안 0%로 보인다. 이미 완료된 fold와 이번에 이어
            # 학습하는 fold의 기완료 epoch 만큼을 분자에 미리 반영한다.
            completed_bpe = bpe_per_fold[:cur_fold_idx] if cur_fold_idx <= len(bpe_per_fold) else bpe_per_fold
            completed_batches = sum(completed_bpe) * self.total_epochs_per_run
            cur_fold_bpe = bpe_per_fold[cur_fold_idx] if cur_fold_idx < len(bpe_per_fold) else batches_per_epoch
            self.batches_done = completed_batches + start_epoch * cur_fold_bpe

            # 단일 TQDM 바 생성
            self.global_pbar = tqdm(
                total=self.total_train_batches,
                initial=self.batches_done,
                desc="Training",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                leave=True
            )
            self.start_time = time.time()
        # else: 같은 프로세스 내 다음 fold로 전환되는 경우, 그랜드 토탈은 이미
        # 고정되어 있으므로 재계산하지 않는다 (on_train_batch_end 에서 계속 누적됨).

    def on_train_batch_end(self, trainer):
        if self.global_pbar is None:
            return
            
        self.batches_done += 1
        self.batches_this_session += 1
        self.global_pbar.update(1)

        # Loss 업데이트
        loss = getattr(trainer, 'loss', None)
        if loss is not None:
            # 보통 loss 는 tensor 이므로 item() 이나 첫 번째 원소 사용
            try:
                self.last_loss = float(loss.detach().cpu().numpy()[0]) if hasattr(loss, 'detach') else float(loss)
            except:
                pass

        # Total ETA 계산 (속도는 이번 세션에서 실제로 처리한 배치 기준으로 계산.
        # batches_done 에는 resume 시 미리 반영한 기완료분이 섞여 있어 속도 계산에는 부적합)
        elapsed = time.time() - self.start_time
        speed = elapsed / self.batches_this_session if self.batches_this_session > 0 else 0
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

