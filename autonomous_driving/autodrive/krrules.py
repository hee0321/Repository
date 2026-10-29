"""rules/kr.yaml 읽기 (없거나 pyyaml이 없으면 기본값)."""
from __future__ import annotations

import functools
import os

DEFAULT = {
    "traffic_side": "right", "lane_width_m": 3.2,
    "speed_limits_kmh": {"trunk": 60, "primary": 50, "secondary": 50, "tertiary": 50,
                         "residential": 30, "living_street": 30, "unclassified": 30},
    "school_zone": {"limit_kmh": 30, "radius_m": 300, "facilities": ["초등학교", "유치원", "어린이집"]},
    "speeding_tolerance_kmh": 3,
    "traffic_signal": {"green_s": 25, "yellow_s": 3, "red_s": 20, "stop_line_before_m": 3},
    "right_turn_stop": {"enabled": True, "detect_turn_rad": 0.6, "stop_speed_mps": 0.3, "max_turn_speed_kmh": 15},
    "centerline": {"margin_m": 0.3},
    "rural": {"general_limit_kmh": 60, "narrow_limit_kmh": 40, "village_school_limit_kmh": 30,
              "reaction_time_s": 1.0, "safe_decel_mps2": 4.0, "min_edge_clearance_m": 0.25,
              "meet_speed_kmh": 15, "ped_pass_clearance_m": 1.0, "ped_pass_speed_kmh": 20,
              "overtake_min_sight_m": 100},
}
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules", "kr.yaml")


@functools.lru_cache(maxsize=None)
def load(path: str = PATH) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT.items()}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except (ImportError, FileNotFoundError):
        pass
    return cfg
