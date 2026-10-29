"""한국 실제 도로 시나리오 정의 (OpenStreetMap). 좌표는 WGS84 위도/경도."""

SCENARIOS = {
    "gangnam_station": {
        "title": "강남역 일대: 신논현역 → 강남역 → 역삼역 (강남대로·테헤란로)",
        "center": (37.5010, 127.0300), "radius": 900,
        "waypoints": [(37.5046, 127.0250), (37.4979, 127.0276), (37.5006, 127.0364)],
    },
    "gangnam_reverse": {
        "title": "강남역 일대 역방향: 역삼역 → 강남역(우회전) → 신논현역",
        "center": (37.5010, 127.0300), "radius": 900, "data": "gangnam_station",
        "waypoints": [(37.5006, 127.0364), (37.4979, 127.0276), (37.5046, 127.0250)],
    },
    "daechi_school_zone": {
        "title": "대치동 서울대치초등학교 주변 어린이보호구역",
        "center": (37.4910, 127.0624), "radius": 700,
        "waypoints": [(37.4945, 127.0570), (37.4932, 127.0598), (37.4919, 127.0607)],
    },
}

# --- 실제 시골길: 경기 양평군 서종면 정배리 (정배초등학교 주변, 중미산로·서정로·마을길) ---
# OSM 공식 API(작은 영역)로 받아 korea_roads/data/yangpyeong_jeongbae.json 에 캐시 (© OpenStreetMap contributors, ODbL)
JEONGBAE_BBOX = (127.4110, 37.5950, 127.4360, 37.6110)
SCENARIOS.update({
    "jeongbae_village": {
        "title": "양평 정배리: 중미산로 굽잇길 → 정배초등학교 앞(마을 어린이보호구역) → 서정로 갈림길",
        "bbox": JEONGBAE_BBOX, "data": "yangpyeong_jeongbae", "rural": True, "night": False, "traffic_seed": 3,
        "waypoints": [(37.6042, 127.4120), (37.6022, 127.4220), (37.5995, 127.4290)],
    },
    "jeongbae_night": {
        "title": "양평 정배리 밤길: 서정로 남쪽 굽잇길 (가로등 없음)",
        "bbox": JEONGBAE_BBOX, "data": "yangpyeong_jeongbae", "rural": True, "night": True, "traffic_seed": 5,
        "waypoints": [(37.5990, 127.4283), (37.5930, 127.4225), (37.5885, 127.4240)],
    },
    "jeongbae_lane": {
        "title": "양평 정배리 마을 안길(좁은 1차로, 중앙선 없음)",
        "bbox": JEONGBAE_BBOX, "data": "yangpyeong_jeongbae", "rural": True, "night": False, "traffic_seed": 8,
        "waypoints": [(37.6045, 127.4250), (37.6075, 127.4262), (37.6105, 127.4275)],
    },
})
