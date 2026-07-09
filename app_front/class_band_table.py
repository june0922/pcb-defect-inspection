# 클래스별 REVIEW 밴드(min/max %)를 표시·추가·삭제하는 재사용 위젯 (SettingsDialog/DefaultsEditDialog 공용)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QPushButton, QAbstractItemView,
    QInputDialog, QMessageBox,
)
from PyQt5.QtCore import Qt


class ClassBandTableWidget(QWidget):
    """결함 클래스 목록 + 클래스별 REVIEW MIN/MAX(%) 테이블.

    [+]/[-] 버튼으로 클래스를 자유롭게 추가/삭제할 수 있다.
    최소 1개 클래스는 항상 유지되도록 마지막 남은 행은 삭제할 수 없다.
    """

    def __init__(self, classes: list[dict], parent=None):
        """classes: [{"name": str, "review_min": int, "review_max": int}, ...]"""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["결함 클래스", "REVIEW MIN (%)", "REVIEW MAX (%)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("클래스 추가 [+]")
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._remove_btn = QPushButton("클래스 삭제 [-]")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        for c in classes:
            self._append_row(c["name"], c["review_min"], c["review_max"])
        self._update_row_height()
        self._update_remove_enabled()

    # ── Public API ─────────────────────────────────────────────

    def add_class(self, name: str) -> bool:
        """클래스 추가. 빈 이름/중복 이름이면 경고 후 False 반환."""
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "클래스 추가 오류", "클래스 이름을 입력해 주세요.")
            return False
        if name in self.get_bands():
            QMessageBox.warning(self, "클래스 추가 오류", f"'{name}' 클래스는 이미 존재합니다.")
            return False
        self._append_row(name, 30, 70)
        self._update_row_height()
        self._update_remove_enabled()
        return True

    def remove_class(self, row: int) -> None:
        """지정된 행 제거. 마지막 1개 남은 행은 제거하지 않는다."""
        if not (0 <= row < self._table.rowCount()):
            return
        if self._table.rowCount() <= 1:
            return
        self._table.removeRow(row)
        self._update_row_height()
        self._update_remove_enabled()

    def get_bands(self) -> dict:
        """{class_name: (review_min, review_max)} 반환."""
        bands = {}
        for row in range(self._table.rowCount()):
            name = self._table.item(row, 0).text()
            min_spin = self._table.cellWidget(row, 1)
            max_spin = self._table.cellWidget(row, 2)
            bands[name] = (min_spin.value(), max_spin.value())
        return bands

    def validate(self) -> str | None:
        """각 행의 min < max 검증. 문제가 있으면 에러 메시지, 없으면 None."""
        for name, (mn, mx) in self.get_bands().items():
            if mn >= mx:
                return f"[{name}] REVIEW MIN({mn}%) 은 MAX({mx}%) 보다 작아야 합니다."
        return None

    # ── Internal ───────────────────────────────────────────────

    def _append_row(self, name: str, review_min: int, review_max: int):
        row = self._table.rowCount()
        self._table.insertRow(row)

        name_item = QTableWidgetItem(name)
        name_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, 0, name_item)

        min_spin = QSpinBox()
        min_spin.setRange(1, 99)
        min_spin.setSuffix(" %")
        min_spin.setValue(review_min)
        min_spin.setToolTip(f"{name}: 이 값 미만 → PASS")
        self._table.setCellWidget(row, 1, min_spin)

        max_spin = QSpinBox()
        max_spin.setRange(2, 100)
        max_spin.setSuffix(" %")
        max_spin.setValue(review_max)
        max_spin.setToolTip(f"{name}: 이 값 초과 → FAIL")
        self._table.setCellWidget(row, 2, max_spin)

    def _update_row_height(self):
        self._table.setFixedHeight(self._table.rowCount() * 30 + 30)

    def _update_remove_enabled(self):
        self._remove_btn.setEnabled(self._table.rowCount() > 1)

    def _on_add_clicked(self):
        name, ok = QInputDialog.getText(self, "클래스 추가", "새 클래스 이름:")
        if ok:
            self.add_class(name)

    def _on_remove_clicked(self):
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "클래스 삭제", "삭제할 클래스를 먼저 선택해 주세요.")
            return
        self.remove_class(row)
