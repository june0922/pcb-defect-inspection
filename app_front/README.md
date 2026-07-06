# app_front — PCB 실시간 검사 모니터링 프로그램

산업용 AOI(Automated Optical Inspection) 장비의 실시간 검사 모니터링 화면을 시뮬레이션하는 프로그램입니다.

## 실행 방법

```bash
cd app_front
python run.py
```

## 기능

### 자동 검사 파이프라인
- 5K-Fold YOLOv8n + WBF(Weighted Boxes Fusion) 앙상블 추론
- 서펜타인(ㄹ자) 스캔 패턴 (이미지마다 교차 방향)
- 임의 크기 이미지 지원 (640의 배수가 아닌 경우 흰색 패딩)

### 화면 구성
- **GlobalView (좌측 30%)**: 컬러 PCB 이미지 + 그리드 오버레이 + 카메라 위치 표시
- **LocalView (우측 70%)**: 현재 검사 중인 640×640 이진화 타일 + 결함 오버레이
- **FilmStrip (하단)**: 타일 검사 히스토리 (PASS/FAIL/REVIEW 색상 코딩)
- **Statistics Panel**: 검사 진행률, FPY, Throughput, 결함 분포

### 판정 기준
| 판정 | 색상 | 조건 |
|------|------|------|
| PASS | 초록 | 모든 검출 confidence < PASS 임계값 또는 검출 없음 |
| REVIEW | 노랑 | FAIL 없고, PASS 임계값 이상 검출 존재 |
| FAIL | 빨강 | FAIL 임계값 이상 검출이 하나라도 존재 |

### 조작
- **File > Open Folder**: 검사 대상 폴더 선택 (기본: `merged_data`)
- **Option > Settings**: 임계값, IoU, 경고음 설정
- **Space**: 검사 일시정지/재개

### 설정
`settings.json` 파일에 저장됩니다 (`config.yaml`과 독립).

| 항목 | 기본값 | 설명 |
|------|--------|------|
| pass_threshold | 30% | 이 값 미만의 confidence → PASS |
| fail_threshold | 70% | 이 값 이상의 confidence → FAIL |
| iou_threshold | 0.45 | WBF 앙상블 IoU 임계값 |
| alert_sound | true | FAIL 검출 시 경고음 |

### JSON 로그
검사 시작 시 `inspection_log_{timestamp}.json` 파일이 자동 생성됩니다.

```json
{
  "inspection_session": { "session_id": "...", "start_time": "...", "settings": {...} },
  "summary": { "total_tiles": 1100, "pass_rate": 95.2, "defect_distribution": {...} },
  "tile_results": [{ "tile_id": 0, "verdict": "PASS", "detections": [...] }, ...]
}
```

## 의존성

- PyQt5
- ultralytics (YOLOv8)
- ensemble_boxes
- opencv-python
- numpy
- torch (GPU 가속 시)

## 파일 구조

```
app_front/
├── __init__.py           # 패키지 초기화
├── run.py                # 진입점 (다크 테마, High DPI)
├── vision_viewer.py      # 읽기 전용 타일 뷰어
├── global_view.py        # 그리드 맵 오버레이 뷰어
├── inspection_worker.py  # 자동 검사 워커 (QThread)
├── main_ui.py            # 메인 윈도우 통합
├── settings.json         # 프로그램 설정 (자동 생성)
└── README.md             # 이 문서
```
