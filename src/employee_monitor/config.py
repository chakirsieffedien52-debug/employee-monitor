"""Configuration models and validation."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Optional, Tuple, Union
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from employee_monitor.zones import DeskZone, load_zones

Source = Union[int, str]


class ConfigurationError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass(frozen=True)
class CameraConfig:
    name: str
    source: Source
    calibration_seconds: float
    zones_file: Path
    desks_file: Path
    manual_zones: Tuple[DeskZone, ...] = ()
    zone_frame_size: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    fallback: str
    imgsz: int
    person_confidence: float
    phone_confidence: float
    object_confidence: float
    use_gpu: bool
    pose_enabled: bool
    pose_model: str
    tiled_phones: bool
    tile_size: int
    tile_overlap: float


@dataclass(frozen=True)
class StatusConfig:
    stable_frames: int
    persistence_frames: int
    phone_proximity_ratio: float
    still_seconds: float


@dataclass(frozen=True)
class DeskMemoryConfig:
    enabled: bool
    min_still_seconds: float
    absent_after_seconds: float
    urgent_after_seconds: float
    margin: float
    max_desks: int
    forget_after_minutes: float
    webhook_url: Optional[str]


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    mode: str
    cameras: Tuple[CameraConfig, ...]
    model: ModelConfig
    status: StatusConfig
    zones_dir: Path
    desks_dir: Path
    desks_memory: DeskMemoryConfig
    logs_dir: Path
    output_dir: Path
    save_video: bool
    auto_discovery: bool
    clahe_enabled: bool
    debug: bool
    blur_faces: bool
    dashboard_host: str
    dashboard_port: int
    reconnect_initial_seconds: float
    reconnect_max_seconds: float
    cluster_distance_ratio: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a mapping.")
    return value


def _number(value: Any, label: str, minimum: float, *, inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a number.")
    result = float(value)
    if result < minimum or (not inclusive and result == minimum):
        operator = "at least" if inclusive else "greater than"
        raise ConfigurationError(f"{label} must be {operator} {minimum}.")
    return result


def _confidence(value: Any, label: str) -> float:
    result = _number(value, label, 0.0)
    if result > 1.0:
        raise ConfigurationError(f"{label} must be between 0 and 1.")
    return result


def _integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{label} must be an integer.")
    if value < minimum:
        raise ConfigurationError(f"{label} must be at least {minimum}.")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be true or false.")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string.")
    return value.strip()


def _camera_source(value: Any, label: str) -> Source:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ConfigurationError(
            f"{label} must be a webcam index, video path, or RTSP/HTTP URL."
        )
    if isinstance(value, int) and value < 0:
        raise ConfigurationError(f"{label} cannot be a negative camera index.")
    if isinstance(value, str) and not value.strip():
        raise ConfigurationError(f"{label} cannot be empty.")
    return value


def _safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a non-empty string.")
    name = value.strip()
    if name in (".", "..") or any(char in name for char in "\\/:"):
        raise ConfigurationError(
            f"{label} cannot contain path separators or a drive prefix."
        )
    return name


def _frame_size(value: Any, label: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, dict):
        width, height = value.get("width"), value.get("height")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = value
    else:
        raise ConfigurationError(f"{label} must be [width, height].")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ConfigurationError(f"{label} dimensions must be positive integers.")
    return width, height


def _resolve_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError(f"{label} must be a non-empty path.")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _make_camera(
    root: Path,
    raw: Mapping[str, Any],
    index: int,
    default_calibration: float,
    shared_zones: Tuple[DeskZone, ...],
    shared_frame_size: Optional[Tuple[int, int]],
    zones_dir: Path,
    desks_dir: Path,
) -> CameraConfig:
    name = _safe_name(raw.get("name", f"camera-{index}"), "camera.name")
    source_label = f"camera {name!r} source"
    source = _camera_source(
        _expand_source_environment(raw.get("source"), source_label), source_label
    )
    calibration_seconds = _number(
        raw.get("calibration_seconds", default_calibration),
        f"camera {name!r} calibration_seconds",
        0.0,
    )
    manual_zones = tuple(load_zones(raw["zones"])) if "zones" in raw else shared_zones
    frame_size = _frame_size(
        raw.get("zone_frame_size", shared_frame_size),
        f"camera {name!r} zone_frame_size",
    )
    zones_file = _resolve_path(
        root, raw.get("zones_file", str(zones_dir / f"{name}.json")), "zones_file"
    )
    desks_file = desks_dir / f"{name}.json"
    return CameraConfig(
        name=name,
        source=source,
        calibration_seconds=calibration_seconds,
        zones_file=zones_file,
        desks_file=desks_file,
        manual_zones=manual_zones,
        zone_frame_size=frame_size,
    )


def load_config(path: Union[str, Path]) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    load_dotenv(config_path.parent / ".env", override=False)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    root_config = _mapping(raw, "Configuration")
    root = config_path.parent

    model_raw = _mapping(root_config.get("model", {}), "model")
    model = ModelConfig(
        name=_nonempty_string(model_raw.get("name", "yolov8n.pt"), "model.name"),
        fallback=_nonempty_string(
            model_raw.get("fallback", "yolov8s.pt"), "model.fallback"
        ),
        imgsz=_integer(model_raw.get("imgsz", 960), "model.imgsz", 1),
        person_confidence=_confidence(
            model_raw.get("person_conf", 0.25), "model.person_conf"
        ),
        phone_confidence=_confidence(
            model_raw.get("phone_conf", 0.15), "model.phone_conf"
        ),
        object_confidence=_confidence(
            model_raw.get("object_conf", 0.2), "model.object_conf"
        ),
        use_gpu=_boolean(model_raw.get("use_gpu", True), "model.use_gpu"),
        pose_enabled=_boolean(
            model_raw.get("pose_enabled", False), "model.pose_enabled"
        ),
        pose_model=_nonempty_string(
            model_raw.get("pose_model", "yolov8n-pose.pt"), "model.pose_model"
        ),
        tiled_phones=_boolean(
            model_raw.get("tiled_phones", False), "model.tiled_phones"
        ),
        tile_size=_integer(model_raw.get("tile_size", 640), "model.tile_size", 64),
        tile_overlap=_number(
            model_raw.get("tile_overlap", 0.2), "model.tile_overlap", 0.0
        ),
    )
    if model.tile_overlap >= 1.0:
        raise ConfigurationError("model.tile_overlap must be less than 1.")

    status_raw = _mapping(root_config.get("status", {}), "status")
    status = StatusConfig(
        stable_frames=_integer(
            status_raw.get("stable_frames", 3), "status.stable_frames", 1
        ),
        persistence_frames=_integer(
            status_raw.get("persistence_frames", 3),
            "status.persistence_frames",
            0,
        ),
        phone_proximity_ratio=_number(
            status_raw.get("phone_proximity_ratio", 0.45),
            "status.phone_proximity_ratio",
            0.0,
        ),
        still_seconds=_number(
            status_raw.get("still_seconds", 30.0),
            "status.still_seconds",
            0.0,
            inclusive=False,
        ),
    )

    auto_raw = _mapping(root_config.get("auto_discovery", {}), "auto_discovery")
    mode = _nonempty_string(root_config.get("mode", "detect"), "mode").lower()
    if mode not in ("detect", "desks"):
        raise ConfigurationError("mode must be either 'detect' or 'desks'.")
    preprocessing_raw = _mapping(root_config.get("preprocessing", {}), "preprocessing")
    default_calibration = _number(
        auto_raw.get("calibration_seconds", 10.0),
        "auto_discovery.calibration_seconds",
        0.0,
    )
    zones_dir = _resolve_path(root, root_config.get("zones_dir", "zones"), "zones_dir")
    desks_dir = _resolve_path(root, root_config.get("desks_dir", "desks"), "desks_dir")
    desks_memory_raw = _mapping(root_config.get("desks_memory", {}), "desks_memory")
    webhook_url = desks_memory_raw.get("webhook_url")
    if webhook_url is not None:
        webhook_url = _nonempty_string(webhook_url, "desks_memory.webhook_url")
        webhook = urlparse(webhook_url)
        if webhook.scheme not in ("http", "https") or not webhook.netloc:
            raise ConfigurationError(
                "desks_memory.webhook_url must be an HTTP or HTTPS URL."
            )
    desks_memory = DeskMemoryConfig(
        enabled=_boolean(desks_memory_raw.get("enabled", True), "desks_memory.enabled"),
        min_still_seconds=_number(
            desks_memory_raw.get("min_still_seconds", 5.0),
            "desks_memory.min_still_seconds",
            0.0,
        ),
        absent_after_seconds=_number(
            desks_memory_raw.get("absent_after_seconds", 3.0),
            "desks_memory.absent_after_seconds",
            0.0,
        ),
        urgent_after_seconds=_number(
            desks_memory_raw.get("urgent_after_seconds", 30.0),
            "desks_memory.urgent_after_seconds",
            0.0,
        ),
        margin=_number(
            desks_memory_raw.get("margin", 0.25), "desks_memory.margin", 0.0
        ),
        max_desks=_integer(
            desks_memory_raw.get("max_desks", 12), "desks_memory.max_desks", 1
        ),
        forget_after_minutes=_number(
            desks_memory_raw.get("forget_after_minutes", 0.0),
            "desks_memory.forget_after_minutes",
            0.0,
        ),
        webhook_url=webhook_url,
    )
    if desks_memory.urgent_after_seconds < desks_memory.absent_after_seconds:
        raise ConfigurationError(
            "desks_memory.urgent_after_seconds must be at least absent_after_seconds."
        )
    legacy_input = _mapping(root_config.get("input", {}), "input")
    legacy_source = legacy_input.get("source")
    camera_items = root_config.get("cameras")
    if camera_items is None:
        if legacy_source is None:
            raise ConfigurationError("Add at least one camera to the cameras list.")
        camera_items = [{"name": "camera-1", "source": legacy_source}]
    if not isinstance(camera_items, list) or not camera_items:
        raise ConfigurationError("cameras must be a non-empty list.")
    raw_zones = root_config.get("zones", [])
    if raw_zones is None:
        raw_zones = []
    if not isinstance(raw_zones, list):
        raise ConfigurationError("zones must be a list.")
    shared_zones = tuple(load_zones(raw_zones))
    shared_frame_size = _frame_size(
        root_config.get("zone_frame_size"), "zone_frame_size"
    )
    cameras = tuple(
        _make_camera(
            root,
            _mapping(camera, f"cameras[{index}]"),
            index + 1,
            default_calibration,
            shared_zones,
            shared_frame_size,
            zones_dir,
            desks_dir,
        )
        for index, camera in enumerate(camera_items)
    )
    names = [camera.name.casefold() for camera in cameras]
    if len(set(names)) != len(names):
        raise ConfigurationError("Camera names must be unique.")

    privacy_raw = _mapping(root_config.get("privacy", {}), "privacy")
    dashboard_raw = _mapping(root_config.get("dashboard", {}), "dashboard")
    runtime_raw = _mapping(root_config.get("runtime", {}), "runtime")
    reconnect_initial = _number(
        runtime_raw.get("reconnect_initial_seconds", 1.0),
        "runtime.reconnect_initial_seconds",
        0.0,
        inclusive=False,
    )
    reconnect_max = _number(
        runtime_raw.get("reconnect_max_seconds", 30.0),
        "runtime.reconnect_max_seconds",
        reconnect_initial,
    )
    port = _integer(dashboard_raw.get("port", 8000), "dashboard.port", 1)
    if port > 65535:
        raise ConfigurationError("dashboard.port must be between 1 and 65535.")

    def path_setting(key: str, default: str) -> Path:
        return _resolve_path(root, root_config.get(key, default), key)

    return AppConfig(
        config_path=config_path,
        mode=mode,
        cameras=cameras,
        model=model,
        status=status,
        zones_dir=zones_dir,
        desks_dir=desks_dir,
        desks_memory=desks_memory,
        logs_dir=path_setting("logs_dir", "logs"),
        output_dir=path_setting("output_dir", "outputs"),
        save_video=_boolean(
            _mapping(root_config.get("output", {}), "output").get("save_video", False),
            "output.save_video",
        ),
        auto_discovery=_boolean(
            auto_raw.get("enabled", True), "auto_discovery.enabled"
        ),
        clahe_enabled=_boolean(
            preprocessing_raw.get("clahe", False), "preprocessing.clahe"
        ),
        debug=_boolean(root_config.get("debug", False), "debug"),
        blur_faces=_boolean(privacy_raw.get("blur_faces", False), "privacy.blur_faces"),
        dashboard_host=_nonempty_string(
            dashboard_raw.get("host", "127.0.0.1"), "dashboard.host"
        ),
        dashboard_port=port,
        reconnect_initial_seconds=reconnect_initial,
        reconnect_max_seconds=reconnect_max,
        cluster_distance_ratio=_number(
            auto_raw.get("cluster_distance_ratio", 0.1),
            "auto_discovery.cluster_distance_ratio",
            0.0,
            inclusive=False,
        ),
    )


def _expand_source_environment(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return Template(value).substitute(os.environ)
    except KeyError as exc:
        raise ConfigurationError(
            f"{label} references unset environment variable {exc.args[0]!r}."
        ) from exc
    except ValueError as exc:
        raise ConfigurationError(
            f"{label} contains invalid environment syntax."
        ) from exc


def source_is_network(source: Source) -> bool:
    if isinstance(source, int):
        return True
    return urlparse(source).scheme.lower() in ("rtsp", "http", "https")


def resolve_camera_source(source: Source, base_dir: Path) -> Source:
    if isinstance(source, int) or source_is_network(source):
        return source
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ConfigurationError(f"Video file not found: {path}")
    return str(path.resolve())
