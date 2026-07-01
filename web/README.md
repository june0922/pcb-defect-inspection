# PCB 결함 검사 웹 데모

Team Convex 인턴 프로젝트 — 발표 시연용 로컬 데모.  
`src/inspect.py` 의 OK / NG / REVIEW 판정 로직을 그대로 재사용하는 웹 래퍼.

---

## 준비

### best.pt 모델 파일

`best.pt`는 `.gitignore`에 등록되어 git에 포함되지 않습니다.  
아래 두 경로 중 하나에 두면 서버가 자동으로 찾습니다 (위에서부터 우선 탐색). **만약 두 경로 모두에 `best.pt`가 없으면, 사전 학습된 `yolov8n.pt`를 자동으로 다운로드하여 폴백 모드로 실행됩니다.**

```
pcb-project/
├── web/
│   └── best.pt          ← 우선 탐색 (데모용 권장)
└── weights/
    └── best.pt          ← 차선 탐색 (학습 후 자동 저장 위치)
```

---

## 샘플 이미지 준비

`web/samples/` 에 시연용 이미지 7장을 넣어두면 좌측 썸네일로 표시됩니다.

```
web/samples/
├── ok_board.jpg          # TODO: 양품 1장
├── open.jpg              # TODO: open 결함
├── short.jpg             # TODO: short 결함
├── mousebite.jpg         # TODO: mousebite 결함
├── spur.jpg              # TODO: spur 결함
├── copper.jpg            # TODO: copper 결함
└── pinhole.jpg           # TODO: pinhole 결함
```

---

## 실행

```bash
# 1. 레포 루트로 이동
cd /path/to/pcb-project

# 2. 의존성 설치 (최초 1회)
pip install -r requirements.txt

# 3. best.pt 를 web/ 또는 weights/ 에 배치

# 4. 서버 시작
uvicorn web.app:app --reload --port 8000

# 5. 브라우저에서 접속
# → http://localhost:8000
# → 좌측 썸네일 클릭 → 결과 즉시 확인
```

> `--reload` 는 코드 변경 시 자동 재시작. 발표 당일에는 빼도 됩니다.

---

## 화면 구성

```
┌──────────┬─────────────────────┬──────────────┐
│ 검사 보드  │   주석 이미지          │  판정 결과    │
│ 선택      │                     │              │
│ [thumb]  │  ┌───────────────┐  │  [ OK ]      │
│ [thumb]  │  │  bbox 결함 표시│  │  결함: 0개    │
│ [thumb]  │  └───────────────┘  │  클래스별 수  │
│ ...      │                     │  결함 목록    │
│ ↑업로드  │                     │  판정 기준    │
└──────────┴─────────────────────┴──────────────┘
```

- **OK** = 초록, **NG** = 빨강, **REVIEW** = 주황 (수동 검토 필요)
- 판정 로직은 `src/inspect.py` 에 있으며 웹은 재사용만 합니다.
- `config.yaml` 의 `judge.conf_threshold` / `judge.review_band` 가 기준값.

---

## 발표 전 체크리스트

```
[ ] best.pt 를 web/ 또는 weights/ 에 배치 확인
[ ] web/samples/ 에 양품 1장 + 결함 6종(각 클래스) 총 7장 준비
[ ] pip install -r requirements.txt 완료
[ ] uvicorn web.app:app --port 8000 실행 → 콘솔에 "워밍업 완료" 확인
[ ] 브라우저에서 http://localhost:8000 열기
[ ] 샘플 클릭 → OK / NG / REVIEW 각 1장씩 리허설 1회
[ ] 업로드 기능도 백업으로 1회 확인
```

---

## 파일 구조

```
web/
├── app.py          # FastAPI 서버 (모델 로드·라우팅)
├── static/
│   ├── index.html  # 검사 화면 (3패널 레이아웃)
│   ├── style.css   # 공장 단말 스타일 (다크 테마)
│   └── script.js   # 샘플 클릭·업로드·결과 렌더링
├── samples/        # 시연용 이미지 (git 포함)
└── README.md       # 이 파일
```
