# PCB 결함 검사 시스템 — Team Convex

ATI 인턴 교육 프로젝트. YOLOv8 기반 PCB 결함 검출 + **보드 단위 OK/NG/REVIEW 판정**까지 내리는 검사 시스템.

> **차별점**: 단순 bounding box 출력이 아니라 신뢰도 기반 3단계 판정(OK / NG / REVIEW)을 통해
> recall 우선 설계(불량 놓침 최소화)를 구현한다.

---

## 프로젝트 구조

```
pcb-project/
├── config.yaml              # 전체 설정 (env 한 줄로 서버/콜랩/로컬 전환)
├── data.yaml                # YOLO 데이터셋 정의
├── requirements.txt
├── .gitignore               # data/ runs/ weights/ *.pt / DeepPCB/ 제외
├── README.md
├── src/
│   ├── __init__.py
│   ├── utils.py             # load_config(), get_paths() — 환경 분기 공통화
│   ├── preprocess.py        # DeepPCB → YOLO 포맷 변환 + CLAHE + 70/20/10 split
│   ├── train.py             # YOLOv8 학습 (n→s→m)
│   ├── evaluate.py          # test 세트 mAP / recall 평가
│   ├── pcb_inspect.py       # ★ 보드 판정 레이어 (OK / NG / REVIEW)
│   └── visualize.py         # EDA + 검사 결과 시각화
└── notebooks/
    └── pcb_colab_verify.ipynb  # 콜랩 sanity check
```

### 경로 규칙

| 환경 | env 값 | raw_data | project_root |
|---|---|---|---|
| 서버 (공용) | `server` | `/shared/datasets/DeepPCB/PCBData` (읽기전용·복사 금지) | `/home/team_a/pcb-project` |
| 콜랩 | `colab` | `/content/DeepPCB/PCBData` | `/content/pcb-project` |
| 로컬 | `local` | `~/DeepPCB/PCBData` (gitignore 됨) | 레포 루트 |

`config.yaml` 맨 위 `env:` 한 줄만 바꾸면 모든 경로가 전환된다.

---

## 결함 클래스 (YOLO index 0~5)

DeepPCB 원본 type 1~6 → cls = type - 1

| YOLO idx | 클래스 | 원본 type |
|---|---|---|
| 0 | open | 1 |
| 1 | short | 2 |
| 2 | mousebite | 3 |
| 3 | spur | 4 |
| 4 | copper | 5 |
| 5 | pinhole | 6 |

---

## 실행 순서

```bash
# 0. 의존성 설치
pip install -r requirements.txt

# 1. 전처리 (DeepPCB raw → YOLO 포맷)
python src/preprocess.py

# 2. 학습
python src/train.py

# 3. 테스트 세트 평가
python src/evaluate.py

# 4. 단일 이미지 판정
python src/pcb_inspect.py <이미지 경로>
```

---

## 스모크 테스트 (신규 환경 셋업 확인 / epochs=1 한 바퀴)

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. config.yaml 에서 env 설정 확인 (server/colab/local 중 하나)
#    로컬 검증: git clone https://github.com/tangsanli5201/DeepPCB.git ~/DeepPCB

# 3. 전처리 — 50장만 처리 (스모크용)
python src/preprocess.py --limit 50
# 기대: data/processed/images/{train,val,test} 디렉터리 생성 확인

# 4. 스모크 학습 (epochs=1)
#    config.yaml 에서 train.epochs 를 1 로 임시 변경:
python src/train.py
# 기대: weights/best.pt 생성, /shared 에는 아무 파일도 기록 안 됨

# 5. 평가
python src/evaluate.py
# 기대: Recall / mAP@0.5 / mAP@0.5:0.95 수치 출력

# 6. 단일 이미지 판정
python src/pcb_inspect.py data/processed/images/test/<아무_이미지>.jpg
# 기대: 판정(OK/NG/REVIEW) + 결함 목록 출력
```

---

## 환경 전환 방법

`config.yaml` 첫 번째 줄만 변경:

```yaml
env: local   # → server 또는 colab 으로 변경하면 모든 경로 전환
```

서버에서는 `raw_data` 가 읽기전용 공용 경로를 가리키므로 전처리 결과(processed/)는 반드시 `project_root` 아래에만 기록된다.

---

## 판정 로직 개요 (recall 우선)

```
결함 없음                            → OK
conf ≥ conf_threshold 결함 존재      → NG
review_band 안 결함만 존재           → REVIEW  (수동 검토)
```

- `review_band` 하한을 낮게 잡아 애매한 케이스를 절대 자동 통과시키지 않는다.
- 클래스별 spec 룰(pinhole 1개도 NG 등)은 `src/pcb_inspect.py` `# TODO(개선)` 참고.

---

## 브랜치 분담

| 브랜치 | 내용 |
|---|---|
| `main` | 베이스라인, 리뷰 완료 코드만 merge |
| `feature/preprocessing` | `src/preprocess.py` CLAHE 파라미터 실험, EDA |
| `feature/model-training` | `src/train.py` 하이퍼파라미터 실험 (n→s→m, lr, aug) |
| `feature/eda-viz` | `src/visualize.py` 클래스 분포·bbox 히스토그램·결과 시각화 |
