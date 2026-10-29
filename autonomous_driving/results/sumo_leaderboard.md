# SUMO + PPO 리더보드 (강남역·역방향 우회전·대치동 어린이보호구역, 시드 2개씩)

| time | model | total | arrived | collisions | red_runs | right_turn_no_stop | speeding_s | school_speeding_s | harsh_brake | mean_speed_kmh | promoted | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-30 04:39 | 기준 운전자(규칙) | 99.25 | 6/6 | 0 | 0 | 0 | 0.0 | 0.0 | 9 | 31.6 | - | 제한속도 미리 보기·신호·우회전 일시정지 |
| 2026-10-30 04:48 | BC(DAgger) | 95.58 | 6/6 | 0 | 1 | 0 | 3.0 | 1.0 | 11 | 33.2 | yes | 첫 모델 |
| 2026-10-30 04:48 | BC+PPO | 74.54 | 5/6 | 1 | 0 | 0 | 50.5 | 2.0 | 11 | 34.4 | no | daechi_school@101: 충돌 증가; daechi_school@101: 완주율 하락; 총점 부족 95.58→74.54 |
