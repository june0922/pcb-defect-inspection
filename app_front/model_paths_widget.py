# 검사용 YOLO 모델(.pt 1~5개)을 선택·검증하는 재사용 위젯 (SettingsDialog/DefaultsEditDialog 공용)

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QApplication,
)
from PyQt5.QtCore import Qt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_model_files(paths: list, active_class_names: list) -> tuple:
    """1~5개 .pt 파일이 앙상블 추론에 쓰기 적합한지 검증.

    확인 순서:
    1) 개수가 1~5개인지
    2) 확장자가 .pt인지 (OS 파일창 필터와 별개로 코드에서도 재확인)
    3) YOLO(path) 로드 + 더미 이미지 predict()가 예외 없이 끝나는지
       (ultralytics는 .pt가 아닌 파일도 로드 시점엔 조용히 "성공"하고 predict() 시점에야
       예외를 던지므로, 로드만으로는 검증이 불충분함 — 반드시 predict까지 실행해야 한다)
    4) 2개 이상이면 전부 model.names가 완전히 동일한지, 그리고 현재 활성 클래스 목록과
       최소 1개 이상 이름이 겹치는지 (안 그러면 클래스 체계가 전혀 다른 모델도
       예외 없이 통과해버린다 — 예: COCO 80클래스 사전학습 모델)

    Returns: (성공 여부, 실패 시 사용자에게 보여줄 에러 메시지)
    """
    if not (1 <= len(paths) <= 5):
        return False, f"모델은 1~5개를 선택해야 합니다. (현재 {len(paths)}개)"

    for p in paths:
        if Path(p).suffix.lower() != ".pt":
            return False, f"'{Path(p).name}'은(는) .pt 파일이 아닙니다."
        if not Path(p).exists():
            return False, f"파일을 찾을 수 없습니다: {p}"

    from ultralytics import YOLO

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    all_names = []
    for p in paths:
        try:
            model = YOLO(p)
            model.predict(dummy, verbose=False)
        except Exception as e:
            return False, f"'{Path(p).name}' 모델 로딩/추론 검증 실패: {e}"
        all_names.append(model.names)

    first_names = all_names[0]
    for names in all_names[1:]:
        if names != first_names:
            return False, (
                "선택한 모델들의 클래스 구성(class names)이 서로 다릅니다. "
                "같은 학습 계열의 모델끼리만 함께 선택할 수 있습니다."
            )

    model_class_set = set(first_names.values())
    if active_class_names and not (model_class_set & set(active_class_names)):
        return False, (
            f"선택한 모델의 클래스({', '.join(sorted(model_class_set))})가 "
            f"현재 설정된 결함 클래스({', '.join(active_class_names)})와 전혀 겹치지 않습니다. "
            "다른 용도의 모델을 잘못 선택하지 않았는지 확인하세요."
        )

    return True, ""


class ModelPathsWidget(QWidget):
    """검사 모델(.pt 1~5개) 선택 위젯. 검증에 실패하면 이전 선택을 그대로 유지한다."""

    def __init__(self, model_paths: list, get_active_class_names, parent=None):
        """
        model_paths: 프로젝트 루트 기준 상대경로 문자열 리스트 (초기값)
        get_active_class_names: 검증 시점에 현재 활성 클래스 이름 리스트를 반환하는 콜러블
            (같은 다이얼로그의 클래스 테이블 위젯을 참조해 최신 값을 조회)
        """
        super().__init__(parent)
        self._model_paths = list(model_paths)
        self._get_active_class_names = get_active_class_names

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: #ccc;")
        layout.addWidget(self._label)

        btn = QPushButton("모델 파일 선택... (.pt, 1~5개)")
        btn.clicked.connect(self._on_select_clicked)
        layout.addWidget(btn)

        self._update_label()

    def get_model_paths(self) -> list:
        return list(self._model_paths)

    def _update_label(self):
        names = [Path(p).name for p in self._model_paths]
        self._label.setText("선택된 모델(" + str(len(names)) + "개): " + ", ".join(names))

    def _on_select_clicked(self):
        selected, _ = QFileDialog.getOpenFileNames(
            self, "검사 모델 선택 (.pt, 1~5개)",
            str(_PROJECT_ROOT), "PyTorch Model (*.pt)",
        )
        if not selected:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, err = validate_model_files(selected, self._get_active_class_names())
        finally:
            QApplication.restoreOverrideCursor()

        if not ok:
            QMessageBox.warning(self, "모델 선택 오류", err)
            return

        rel_paths = []
        for p in selected:
            resolved = Path(p).resolve()
            try:
                rel_paths.append(str(resolved.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
            except ValueError:
                rel_paths.append(str(resolved))  # 프로젝트 밖 파일은 절대경로로 유지
        self._model_paths = rel_paths
        self._update_label()
