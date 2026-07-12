# Options 공장 기본값(default_settings.json)을 읽고 쓰는 app_front 전용 모듈 — db 계층은 DB만 알면 되므로 이 파일이 소유한다
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db.database import update_settings

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "default_settings.json"

# default_settings.json을 읽지 못할 때의 최후 폴백(파일 삭제/손상 대비)
_FALLBACK_DEFAULT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
_FALLBACK_MODEL_PATHS = [f"app_front/models/best_fold_{i}_tune.pt" for i in range(1, 6)]


def load_defaults_raw() -> dict:
    """default_settings.json 원본 구조를 그대로 반환 ("기본값 수정" UI 편집용).

    파일이 없거나 파싱에 실패하면 내장 폴백 값을 반환한다.
    """
    try:
        with open(DEFAULT_SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "defect_classes": [
                {"name": cls, "review_min": 30, "review_max": 70}
                for cls in _FALLBACK_DEFAULT_CLASSES
            ],
            "tile_size": 640,
            "overlap_pct": 0,
            "alert_sound": True,
            "model_paths": list(_FALLBACK_MODEL_PATHS),
        }


def save_defaults_raw(data: dict) -> None:
    """default_settings.json을 원자적으로 덮어쓴다 (임시 파일 작성 후 교체 — 크래시로 인한 파일 손상 방지)."""
    tmp_path = DEFAULT_SETTINGS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DEFAULT_SETTINGS_PATH)


def _flatten_defaults(raw: dict) -> dict[str, str]:
    """default_settings.json 구조를 DB settings 테이블용 flat key-value dict로 변환."""
    classes = raw.get("defect_classes", [])
    settings: dict[str, str] = {
        "defect_classes": json.dumps([c["name"] for c in classes]),
        "tile_size": str(raw.get("tile_size", 640)),
        "overlap_pct": str(raw.get("overlap_pct", 0)),
        "alert_sound": "true" if raw.get("alert_sound", True) else "false",
        "model_paths": json.dumps(raw.get("model_paths", _FALLBACK_MODEL_PATHS)),
    }
    for c in classes:
        settings[f"review_min_{c['name']}"] = str(c["review_min"])
        settings[f"review_max_{c['name']}"] = str(c["review_max"])
    return settings


def force_reset_settings_to_defaults() -> None:
    """Options 설정을 default_settings.json 값으로 강제 덮어쓴다 (app_front 구동 시 전용).

    tiles 테이블과 db_session_id는 건드리지 않는다 — 검사 데이터 초기화와는 별개다.
    """
    update_settings(_flatten_defaults(load_defaults_raw()))
