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

# --- 위성영상(브이월드)으로 확인한 OSM 도로 모양 보정 ---
# OSM 선이 실제 길보다 듬성듬성 그려진 곳(노드 두 개를 잇는 직선이 마당·건물 쪽을 가로지름)을
# 위성사진에서 읽은 길 가운데 점들로 쪼갠다. 데이터 이름(scenario["data"])마다 적용 → 같은 지도를 쓰는 시나리오 모두.
#   between: (노드 A 위경도, 노드 B 위경도)  — 각각 3 m 안의 OSM 노드를 찾아 그 사이 간선을 바꿈
#   via:     A→B 사이에 넣을 길 가운데 점들 (위성사진 2026 브이월드 z19, 0.2 m/px 에서 읽음)
GEOMETRY_FIXES = {
    "yangpyeong_jeongbae": [
        # 정배리: 중미산로(서정로 갈림) → 마을 안길 들어가는 곳. OSM 직선이 북쪽 집 마당(붉은 포장)을
        # 3~8 m 가로질렀음. 실제 마을길은 그보다 남쪽으로 동서로 뻗음.
        dict(between=((37.6008041, 127.4271583), (37.6008980, 127.4275982)),
             via=[(37.6008117, 127.4273098), (37.6008243, 127.4274459), (37.6008387, 127.4275366),
                  (37.6008638, 127.4275933)]),
    ],
}
