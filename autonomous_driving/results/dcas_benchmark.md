| 시나리오 | 근거 | expert | rl | vision |
|---|---|---|---|---|
| dcas_lk_curve_85 | Annex4 4.2.5.1.1.1 | ✅🚗 99.89 | ❌🚗 98.02 (lat_jerk<=5) | ❌🚗 98.89 (indicator>=3s_before_LCM) |
| dcas_lk_curve_over | Annex4 4.2.5.1.1.2 | ✅🚗 99.86 | ❌🚗 97.86 (lat_jerk<=5) | ❌🚗 98.57 (lat_acc<=3.3) |
| dcas_lk_scurve | Annex4 4.2.4.1 (S자 클로소이드 표) | ✅🚗 99.96 | ❌🚗 97.76 (lat_jerk<=5) | ✅🚗 98.8 |
| dcas_lc_request | Annex4 4.2.5.1.3.1 | ✅🚗 99.59 | ❌🚗 97.77 (lane_change_completed) | ❌🚗 98.67 (indicator>=3s_before_LCM, lane_change_completed) |
| dcas_lc_rear_approach | Annex4 4.2.5.1.3.2.1 b) | ✅🚗 99.63 | ❌🚗 97.64 (lane_change_completed) | ❌🚗 98.64 (indicator>=3s_before_LCM, LCP_decel<=2, lane_change_completed) |
| dcas_lc_beside | Annex4 4.2.5.1.3.2.1 c) | ✅🚗 99.71 | ❌🚗 88.31 (lane_change_completed) | ❌🚗 85.7 (no_speeding, lane_change_completed) |
| dcas_lc_same_space | Annex4 4.2.5.1.3.2.1 d) | ✅🚗 89.69 | ❌🚗 97.4 (lane_change_completed) | ❌🚗 87.21 (no_speeding, lane_change_completed) |
| dcas_lc_oncoming_refuse | 6.2.2 (반대 방향 차로로 변경 금지) | ✅🚗 100.0 | ❌✋ 1.99 (no_collision) | ✅🚗 97.66 |
| dcas_lane_end_merge | 6.1.2 (차로 합류·감소) | ✅🚗 99.7 | ❌✋ 32.11 (no_collision, LCP_decel<=2, lane_change_completed) | ❌🚗 97.43 (lat_jerk<=5, indicator>=3s_before_LCM, LC_lat_acc<=3.5, LCP_decel<=2) |
| dcas_stationary_70 | Annex4 4.2.5.2.1 | ❌🚗 99.65 (LCP_decel<=2) | ❌✋ 10.87 (no_collision, lat_jerk<=5, indicator>=3s_before_LCM) | ❌✋ 12.37 (no_collision) |
| dcas_stationary_curve | Annex4 4.2.5.2.2 (4.2.4.1 굽이, 차로 가운데서 0.5 m) | ✅🚗 99.61 | ❌✋ 13.9 (no_collision, lat_jerk<=5, indicator>=3s_before_LCM) | ❌✋ 15.45 (no_collision) |
| dcas_slower_lead | Annex4 4.2.5.2.3 (50 km/h 느린 차) | ✅🚗 100.0 | ❌✋ -27.46 (no_road_departure, lat_jerk<=5, indicator>=3s_before_LCM, LC_lat_acc<=3.5) | ✅🚗 97.43 |
| dcas_lead_brake_4 | Annex4 4.2.5.2.4 (50 km/h, 앞차 4 m/s² 감속) | ✅🚗 100.0 | ✅🚗 98.23 | ✅🚗 98.12 |
| dcas_lead_brake_6 | Annex4 4.2.5.2.4.2 확장(더 큰 감속·다른 속도) | ✅🚗 100.0 | ❌✋ -5.17 (no_road_departure, lat_jerk<=5, indicator>=3s_before_LCM) | ❌✋ -3.93 (no_collision) |
| dcas_cut_out | Annex4 4.2.5.2.5 (앞차 빠짐: 70/50 km/h, TTC 3 s, 1.5 m/s², 44 m) | ❌🚗 99.7 (LCP_decel<=2) | ❌✋ 27.5 (no_collision, lat_jerk<=5, indicator>=3s_before_LCM, LC_lat_acc<=3.5) | ❌✋ 21.94 (no_collision, indicator>=3s_before_LCM) |
| dcas_cut_in_1 | Annex4 4.2.5.2.6 Type 1 (50/10 km/h, 0.5 m/s², 14 m, 완료 TTC 0 s) | ✅🚗 89.61 | ❌✋ -1.04 (no_collision) | ❌🚗 88.29 (no_speeding) |
| dcas_cut_in_2 | Annex4 4.2.5.2.6 Type 2 (120/70 → 같은 상대속도 70/20 km/h, 1.5 m/s², 완료 TTC 1.5 s) | ✅🚗 89.6 | ❌✋ 9.95 (no_collision, lat_jerk<=5) | ✅🚗 97.64 |
| dcas_ped_standing | Annex4 4.2.5.2.7 (차로에 선 보행자) | ✅🚗 99.7 | ❌✋ 9.86 (no_collision) | ❌✋ 6.91 (no_collision, no_speeding) |
| dcas_ped_crossing | Annex4 4.2.5.2.9 (5 km/h 직각 횡단, TTC ≥ 4 s, 충돌점 = 차 중심선) | ✅🚗 100.0 | ❌🚗 87.86 (lat_jerk<=5, indicator>=3s_before_LCM) | ❌✋ 6.7 (no_collision, no_speeding) |
| dcas_bike_crossing | Annex4 4.2.5.2.10 (자전거 15 km/h 횡단) | ✅🚗 100.0 | ❌✋ 9.61 (no_collision, lat_acc<=3.3, lat_jerk<=5) | ❌✋ 6.77 (no_collision, no_speeding) |
| **합계** | | pass 18/20, 자율완주 20/20, 점수 98.29 | pass 1/20, 자율완주 9/20, 점수 47.15 | pass 5/20, 자율완주 13/20, 점수 65.46 |

✅ = R171 판정 모두 통과, 🚗 = 자율 완주(개입 0·충돌 0·도착), ✋ = 도착 못 함/개입 필요
