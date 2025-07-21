import streamlit as st

def show_future_strategy_box():
    st.markdown('---')
    st.markdown('## 🔮 미래 전략 제시 (프로젝트 분석 기반)')

    st.markdown("""
<span style='color:#1e3a8a; font-weight:bold; font-size:1.2em; border-bottom:3px solid #F97316; padding-bottom:2px;'>🤖 AI 기반 감염병 예측 시스템</span><br/>
<span style='color:gray'>데이터 근거: city_analysis.py + pandemic_military_analysis.py</span><br/>
- 17개 도시별 감염병-건강등급 상관관계 → LSTM 예측 모델<br/>
- 부대별 실시간 위험도 알림 (감염병 발생률 예측)<br/>
- 병력 배치 최적화 (고위험 지역 사전 대비)<br/>
<br/>
<span style='color:#1e3a8a; font-weight:bold; font-size:1.2em; border-bottom:3px solid #F97316; padding-bottom:2px;'>🏥 스마트 군인 건강관리 플랫폼</span><br/>
<span style='color:gray'>데이터 근거: mma_health_grade.csv + kdca_infections.csv</span><br/>
- 개인 맞춤형 건강관리 (건강등급별 차별화)<br/>
- 웨어러블 연동 실시간 모니터링<br/>
- AI 건강 위험도 예측 (면제율 사전 예방)<br/>
<br/>
<span style='color:#1e3a8a; font-weight:bold; font-size:1.2em; border-bottom:3px solid #F97316; padding-bottom:2px;'>🛡️ 무인/자동화 국방시스템</span><br/>
<span style='color:gray'>데이터 근거: rnd_analysis.py + tech_trend_analysis.py</span><br/>
- AI 무기체계 개발 (팬데믹 시 인력 부족 대비)<br/>
- 자율 방역 로봇 부대 투입<br/>
- 무인 감시 시스템 (사이버+UAV 융합)<br/>
<br/>
<span style='color:#1e3a8a; font-weight:bold; font-size:1.2em; border-bottom:3px solid #F97316; padding-bottom:2px;'>📊 국산화율 78.5% → 85% 달성</span><br/>
<span style='color:gray'>데이터 근거: localization_analysis.py</span><br/>
- 핵심 기술 국산화 우선순위 선정<br/>
- R&D 투자 확대 (현재 2.3조 → 3조)<br/>
- 민-군 기술협력 생태계 구축<br/>
<br/>
<span style='color:#1e3a8a; font-weight:bold; font-size:1.2em; border-bottom:3px solid #F97316; padding-bottom:2px;'>🌐 통합 데이터 플랫폼</span><br/>
<span style='color:gray'>모든 분석 모듈 통합</span><br/>
- 실시간 대시보드 (17개 도시 모니터링)<br/>
- 예측 알고리즘 통합 운영<br/>
- 정책 의사결정 지원 시스템<br/>
""", unsafe_allow_html=True) 