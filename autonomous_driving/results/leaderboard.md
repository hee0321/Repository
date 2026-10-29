# 리더보드 (벤치마크 v1)

`python improve.py`가 사이클마다 한 줄씩 추가해요. 점수는 높을수록 좋아요.

| time | model | total | urban | rural | finished | crashes | violations | speeding_s | school_speeding_s | mean_lane_dev_m | rural_unsafe | promoted | minutes | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 03:49 | pure_pursuit(기준) | 96.97 | 97.86 | 94.77 | 14/14 | 0 | 0 | 0.0 | 0.0 | 0.183 | 0 | - | 0 | 규칙 기반 기준 제어기 |
| 2026-10-30 03:57 | cycle1 | 25.01 | 38.4 | -8.48 | 6/14 | 8 | 2 | 162.8 | 11.2 | 0.449 | 1 | yes | 15.0 | 첫 모델 |
| 2026-10-30 04:05 | cycle2(ppo) | 57.67 | 74.25 | 16.22 | 9/14 | 5 | 2 | 8.3 | 4.0 | 0.395 | 2 | yes | 13.0 | 통과 |
| 2026-10-30 04:10 | cycle3(bc) | 58.7 | 80.29 | 4.72 | 8/14 | 6 | 1 | 6.2 | 3.7 | 0.233 | 1 | no | 5.4 | gangnam_reverse: 신호 위반 증가 0→0; gangnam_traffic: 새로 이탈; gangnam_traffic: 점수 크게 하락 94.14→48.29; daechi_school_zone: 새로 이탈; |
| 2026-10-30 04:16 | cycle4-rural(bc) | 54.33 | 52.16 | 59.76 | 8/14 | 6 | 1 | 8.3 | 3.8 | 0.217 | 1 | no | 5.5 | rules_track: 새로 이탈; rules_track: 완주율 하락 1.0→0.031; rules_track: 점수 크게 하락 66.82→-24.76; gangnam_reverse: 신호 위반 증가 0→0; ga |
| 2026-10-30 04:27 | cycle5-rural(ppo) | 62.93 | 71.73 | 40.92 | 10/14 | 4 | 3 | 5.9 | 5.9 | 0.335 | 1 | no | 5.5 | hard_wet: 새로 이탈; hard_wet: 완주율 하락 1.0→0.461; hard_wet: 점수 크게 하락 97.91→11.81; gangnam_station: 어린이보호구역 과속 증가 1.45→2.7s; g |
