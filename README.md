# Pandemic Defense Innovation 프로젝트 구조

아래는 본 프로젝트의 전체 폴더 및 파일 구조입니다. 데이터 파일명까지 모두 포함되어 있습니다.

```
pandemic_defense_innovation/
│
├─ app.py
│
├─ pandemic_military_analysis.py
│
├─ fonts/
│   └─ NotoSansKR-VariableFont_wght.ttf
│
├─ modules/
│   ├─ city_analysis.py
│   ├─ donut_charts.py
│   ├─ future_strategy.py
│   ├─ health_defense_causality_fixed.py
│   ├─ health_prediction.py
│   ├─ import_export_analysis.py
│   ├─ localization_analysis.py
│   ├─ policy_simulation_engine.py
│   ├─ rnd_analysis.py
│   ├─ strategy_effectiveness_analysis.py
│   └─ tech_trend_analysis.py
│
├─ scripts/
│   ├─ clean_and_rename_csvs.py
│   ├─ kdca_utf8_converter.py
│   ├─ process_kdca_infections.py
│   ├─ process_kdca_master.py
│   └─ process_mma_data.py
│
├─ data/
│   ├─ dapa/
│   │   ├─ dapa_companies_financial_status.csv
│   │   ├─ dapa_companies_overview.csv
│   │   ├─ dapa_cyber_command_status.csv
│   │   ├─ dapa_cyber_education_completion.csv
│   │   ├─ dapa_cyber_education_evaluation.csv
│   │   ├─ dapa_export_company_designation.csv
│   │   ├─ dapa_export_key_items.csv
│   │   ├─ dapa_export_overseas_info_2021.csv
│   │   ├─ dapa_export_overseas_info_2023.csv
│   │   ├─ dapa_foreign_category_status.csv
│   │   ├─ dapa_foreign_contracts.csv
│   │   ├─ dapa_foreign_outsourced_items.csv
│   │   ├─ dapa_foreign_packaged_items.csv
│   │   ├─ dapa_informatization_projects.csv
│   │   ├─ dapa_localization_items.csv
│   │   ├─ dapa_new_tech_announcements.csv
│   │   └─ dapa_rnd_budget_and_tasks.csv
│   │
│   ├─ kdca/
│   │   ├─ kdca_covid19_detailed.csv.xlsx
│   │   ├─ kdca_covid19.csv.csv
│   │   ├─ kdca_health_risk.csv
│   │   ├─ kdca_infections_1.csv
│   │   ├─ kdca_infections.csv
│   │   ├─ kdca_infections.csv.csv
│   │   └─ kdca_pandemic_impact.csv
│   │
│   └─ mma/
│       ├─ mma_bmi.csv
│       ├─ mma_exemption.csv
│       ├─ mma_health_grade.csv
│       ├─ mma_height.csv
│       ├─ mma_simulation_bmi.csv
│       ├─ mma_simulation_exemption.csv
│       ├─ mma_simulation_health_grade.csv
│       ├─ mma_simulation_height.csv
│       ├─ mma_simulation_total_subjects.csv
│       ├─ mma_simulation_weight.csv
│       ├─ mma_total_subjects.csv
│       └─ mma_weight.csv
│
├─ raw_data/
│   ├─ dapa/
│   │   ├─ 방위사업청_국방전자조달시스템_국산화개발품목_20230823.csv
│   │   ├─ 방위사업청_국외조달 계약정보_20241031.csv
│   │   ├─ 방위사업청_국외조달 무기체계 가격정보 획득현황_20231231.csv
│   │   ├─ 방위사업청_국외조달 조달계획_20241031.csv
│   │   ├─ 방위사업청_군수품 목록화 현황_20231231.csv
│   │   ├─ 방위사업청_무기체계 주요 기능별 계약현황_20231231.csv
│   │   ├─ 방위사업청_방산물자 및 업체 지정현황_20221231.csv
│   │   ├─ 방위사업청_방산업체 경영실태_20211231.csv
│   │   ├─ 방위사업청_사이버교육 운영 현황_20230930.csv
│   │   ├─ 방위사업청_사이버교육과정 수료 현황_20230930.csv
│   │   ├─ 방위사업청_사이버교육과정 평가 현황_20230930.csv
│   │   ├─ 방위사업청_신기술 입찰공고 사업_20230930.csv
│   │   ├─ 방위사업청_연도별 무기체계 핵심기술 연구개발 기초연구 예산 및 과제수 현황_20231231.csv
│   │   ├─ 방위사업청_정보화 사업현황_20220731.csv
│   │   └─ 산업통상자원부_방산업체 현황_20250531.csv
│   │
│   ├─ kdca/
│   │   ├─ 2019_전국_감염병.csv
│   │   ├─ 2020_전국_감염병.csv
│   │   ├─ 2021_전국_감염병.csv
│   │   ├─ 2022_전국_감염병.csv
│   │   ├─ 2023_전국_감염병.csv
│   │   ├─ 2024_전국_감염병.csv
│   │   ├─ 간편통계_기간별.csv
│   │   ├─ 질병관리청_신증후군출혈열 지역별 발생현황_20171231..xlsx
│   │   ├─ 질병관리청_코로나19 시군구별 월별 확진자 및 사망 발생 현황_20230831.xlsx
│   │   ├─ 질병관리청_코로나19 확진자 발생현황(전수감시)_20230831.xlsx
│   │   └─ 코로나바이러스감염증-19_확진환자_발생현황_230904_최종v2.xlsx
│   │
│   └─ mma/
│       ├─ 병무청_병역판정검사 결과(4급5급6급)_20181231.csv
│       ├─ 병무청_병역판정검사 대상자 현황_20240101.csv
│       ├─ 병무청_병역판정검사 본인선택 현황_20231231.csv
│       ├─ 병무청_병역판정검사 부령별 현황_20241231.csv
│       ├─ 병무청_병역판정검사 신장 분포 및 청별 현황_20231231.csv
│       ├─ 병무청_병역판정검사 신체질량지수(BMI) 및 청별 현황_20231231.csv
│       ├─ 병무청_병역판정검사 잠복결핵검사 실적 및 청별 현황_20231231.csv
│       ├─ 병무청_병역판정검사 체중 분포 및 청별 현황_20231231.csv
│       ├─ 병무청_병역판정검사 현역 처분_20191231.csv
│       ├─ 병무청_병역판정검사 현황_20241231.csv
│       ├─ 병무청_병역판정검사 혈액형 분포 및 청별 현황_20231231.csv
│       ├─ 병무청_병역판정검사_각 검사결과 병리검사 등_Kdata_20240318.csv
│       ├─ 병무청_병역판정검사_방사선및잠복결핵_Kdata_20220114.csv
│       ├─ 병무청_병역판정검사_병리정보_Kdata_20221025.csv
│       └─ 신체검사_표본데이터.csv
```

---

> 각 폴더/파일명은 실제 프로젝트 내에 존재하는 이름 그대로 표기했습니다.