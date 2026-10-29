# 리더보드 (벤치마크 v2)

`python improve.py`가 사이클마다 한 줄씩 추가해요. 점수는 높을수록 좋아요.

| time | model | total | urban | rural | finished | crashes | violations | speeding_s | school_speeding_s | mean_lane_dev_m | rural_unsafe | promoted | minutes | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 05:28 | pure_pursuit(기준) | 97.96 | 97.86 | 98.09 | 18/18 | 0 | 0 | 0.0 | 0.0 | 0.185 | 0 | - | 0 | 규칙 기반 기준 제어기 (벤치마크 v2) |
| 2026-10-30 05:28 | cycle2(최고, v2 재평가) | 59.27 | 74.25 | 40.54 | 11/18 | 7 | 2 | 9.0 | 5.2 | 0.416 | 3 | - | 0 | 현재 최고 모델을 새 벤치마크로 다시 잰 '이전' 점수 |
| 2026-10-30 05:36 | cycle6-rural-mix(ppo) | 74.08 | 75.55 | 72.25 | 14/18 | 4 | 1 | 8.9 | 6.9 | 0.273 | 3 | no | 8.3 | gangnam_traffic: 어린이보호구역 과속 증가 0.5→2.2s; gangnam_traffic: 점수 크게 하락 94.14→88.79; rural_village: 시골길 위험 행동 증가; rural_villa |
