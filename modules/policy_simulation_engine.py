import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
font_path = os.path.join(os.getcwd(), 'fonts', 'NotoSansKR-VariableFont_wght.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans KR'
else:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans', 'sans-serif']

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

class PolicySimulationEngine:
    """정책 시뮬레이션 엔진"""
    
    def __init__(self):
        self.baseline_data = self.create_baseline_scenario()
        self.policy_effects = self.define_policy_effects()
        
    def create_baseline_scenario(self):
        """기준 시나리오 (현재 정책 유지)"""
        years = list(range(2025, 2031))  # 2025-2030년 예측
        
        baseline = {
            'years': years,
            'rnd_budget': [25000, 26000, 27000, 28000, 29000, 30000],  # 연 4% 증가
            'health_grade': [3.2, 3.1, 3.0, 2.9, 2.8, 2.7],  # 점진적 개선
            'infection_rate': [1.8, 1.6, 1.4, 1.3, 1.2, 1.1],  # 팬데믹 회복
            'exemption_rate': [4.2, 4.4, 4.6, 4.8, 5.0, 5.2],  # 지속 증가
            'defense_export': [2200, 2300, 2400, 2500, 2600, 2700],  # 연 5% 증가
            'automation_investment': [450, 480, 510, 540, 570, 600],  # 연 7% 증가
            'cyber_security': [650, 700, 750, 800, 850, 900],  # 연 6% 증가
            'localization_rate': [78.5, 79.2, 79.9, 80.6, 81.3, 82.0],  # 점진적 증가
            'gdp_impact': [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # GDP 대비 국방 기여도
        }
        
        return baseline
    
    def define_policy_effects(self):
        """정책별 효과 정의"""
        return {
            'ai_health_investment': {
                'name': 'AI 건강관리 투자',
                'health_grade_effect': -0.3,  # 건강등급 개선
                'infection_rate_effect': -0.4,  # 감염병 감소
                'exemption_rate_effect': -0.8,  # 면제율 감소
                'cost_per_year': 1000,  # 연간 1000억원
                'implementation_time': 2  # 2년 후부터 효과
            },
            'automation_acceleration': {
                'name': '무인화 기술 가속화',
                'automation_boost': 1.5,  # 자동화 투자 50% 증가
                'localization_boost': 2.0,  # 국산화율 2%p 추가 증가
                'exemption_tolerance': 0.3,  # 면제율 영향 30% 감소
                'cost_per_year': 1500,  # 연간 1500억원
                'implementation_time': 1  # 1년 후부터 효과
            },
            'rnd_expansion': {
                'name': 'R&D 투자 확대',
                'rnd_boost': 1.3,  # R&D 예산 30% 증가
                'export_boost': 1.2,  # 방산수출 20% 증가 (2년 지연)
                'tech_innovation': 0.5,  # 기술혁신 지수 증가
                'cost_per_year': 2000,  # 연간 2000억원 추가
                'implementation_time': 0  # 즉시 시행
            },
            'integrated_defense_health': {
                'name': '국방-보건 통합 체계',
                'health_grade_effect': -0.5,  # 건강등급 대폭 개선
                'infection_early_warning': -0.6,  # 감염병 조기 대응
                'defense_readiness': 1.2,  # 방위태세 20% 향상
                'cost_per_year': 800,  # 연간 800억원
                'implementation_time': 1  # 1년 후부터 효과
            },
            'cyber_security_enhancement': {
                'name': '사이버보안 강화',
                'cyber_boost': 2.0,  # 사이버보안 예산 100% 증가
                'digital_resilience': 1.3,  # 디지털 복원력 30% 향상
                'export_protection': 0.1,  # 수출 보호 효과
                'cost_per_year': 600,  # 연간 600억원
                'implementation_time': 0  # 즉시 시행
            }
        }

def simulate_policy_scenario(engine, selected_policies, simulation_years=6):
    """정책 시뮬레이션 실행"""
    baseline = engine.baseline_data.copy()
    policy_effects = engine.policy_effects
    
    # 시뮬레이션 결과 초기화
    simulated = {key: baseline[key].copy() for key in baseline.keys()}
    total_cost = 0
    
    for year_idx in range(simulation_years):
        year = baseline['years'][year_idx]
        annual_cost = 0
        
        for policy_name in selected_policies:
            if policy_name not in policy_effects:
                continue
                
            policy = policy_effects[policy_name]
            implementation_year = 2025 + policy['implementation_time']
            
            # 정책 효과가 시작되는 연도 확인
            if year >= implementation_year:
                # AI 건강관리 투자 효과
                if policy_name == 'ai_health_investment':
                    effect_years = year - implementation_year + 1
                    simulated['health_grade'][year_idx] += policy['health_grade_effect'] * min(effect_years * 0.3, 1.0)
                    simulated['infection_rate'][year_idx] += policy['infection_rate_effect'] * min(effect_years * 0.4, 1.0)
                    simulated['exemption_rate'][year_idx] += policy['exemption_rate_effect'] * min(effect_years * 0.2, 1.0)
                    annual_cost += policy['cost_per_year']
                
                # 무인화 기술 가속화 효과
                elif policy_name == 'automation_acceleration':
                    simulated['automation_investment'][year_idx] *= policy['automation_boost']
                    simulated['localization_rate'][year_idx] += policy['localization_boost']
                    # 면제율 영향 감소
                    exemption_impact = (simulated['exemption_rate'][year_idx] - baseline['exemption_rate'][year_idx]) * policy['exemption_tolerance']
                    simulated['exemption_rate'][year_idx] = baseline['exemption_rate'][year_idx] + exemption_impact
                    annual_cost += policy['cost_per_year']
                
                # R&D 투자 확대 효과
                elif policy_name == 'rnd_expansion':
                    simulated['rnd_budget'][year_idx] *= policy['rnd_boost']
                    # 수출 증가 효과 (2년 지연)
                    if year_idx >= 2:
                        simulated['defense_export'][year_idx] *= policy['export_boost']
                    annual_cost += policy['cost_per_year']
                
                # 통합 체계 효과
                elif policy_name == 'integrated_defense_health':
                    effect_strength = min((year - implementation_year + 1) * 0.4, 1.0)
                    simulated['health_grade'][year_idx] += policy['health_grade_effect'] * effect_strength
                    simulated['infection_rate'][year_idx] += policy['infection_early_warning'] * effect_strength
                    simulated['defense_export'][year_idx] *= (1 + (policy['defense_readiness'] - 1) * effect_strength)
                    annual_cost += policy['cost_per_year']
                
                # 사이버보안 강화 효과
                elif policy_name == 'cyber_security_enhancement':
                    simulated['cyber_security'][year_idx] *= policy['cyber_boost']
                    simulated['defense_export'][year_idx] *= (1 + policy['export_protection'])
                    annual_cost += policy['cost_per_year']
        
        total_cost += annual_cost
    
    # 결과 검증 및 조정
    for key in simulated.keys():
        if key in ['health_grade', 'infection_rate', 'exemption_rate', 'localization_rate']:
            simulated[key] = [max(1.0, min(5.0, val)) if 'grade' in key 
                            else max(0.1, val) if 'rate' in key 
                            else max(0, min(100, val)) if 'localization' in key
                            else val for val in simulated[key]]
    
    simulated['total_cost'] = total_cost
    simulated['annual_costs'] = [total_cost / simulation_years] * simulation_years
    
    return simulated

def calculate_policy_roi(baseline, simulated, total_cost):
    """정책 투자 수익률 계산"""
    
    # 경제적 효과 계산
    export_gain = sum(simulated['defense_export']) - sum(baseline['defense_export'])
    automation_value = sum(simulated['automation_investment']) - sum(baseline['automation_investment'])
    health_cost_savings = 0
    
    # 건강 개선으로 인한 비용 절감 (면제율 감소 효과)
    exemption_reduction = sum(baseline['exemption_rate']) - sum(simulated['exemption_rate'])
    health_cost_savings = exemption_reduction * 100  # 면제율 1%p당 100억원 절감 가정
    
    # 사이버보안 투자로 인한 리스크 절감
    cyber_investment_increase = sum(simulated['cyber_security']) - sum(baseline['cyber_security'])
    cyber_risk_reduction = cyber_investment_increase * 0.5  # 투자 대비 50% 리스크 절감
    
    total_benefits = export_gain + automation_value + health_cost_savings + cyber_risk_reduction
    roi_percentage = ((total_benefits - total_cost) / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        'total_benefits': total_benefits,
        'export_gain': export_gain,
        'automation_value': automation_value,
        'health_savings': health_cost_savings,
        'cyber_savings': cyber_risk_reduction,
        'total_cost': total_cost,
        'roi_percentage': roi_percentage,
        'payback_period': total_cost / (total_benefits / 6) if total_benefits > 0 else float('inf')
    }

def create_scenario_comparison_chart(baseline, scenarios):
    """시나리오 비교 차트 생성"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('건강등급 변화', '방산수출 증가', '국산화율 향상', '총 투자 비용'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    years = baseline['years']
    
    # 1. 건강등급 비교
    fig.add_trace(
        go.Scatter(x=years, y=baseline['health_grade'], 
                  mode='lines+markers', name='기준 시나리오',
                  line=dict(color='#6B7280', width=2)),
        row=1, col=1
    )
    
    colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6']
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        fig.add_trace(
            go.Scatter(x=years, y=scenario_data['health_grade'],
                      mode='lines+markers', name=scenario_name,
                      line=dict(color=colors[i % len(colors)], width=3)),
            row=1, col=1
        )
    
    # 2. 방산수출 비교
    fig.add_trace(
        go.Scatter(x=years, y=baseline['defense_export'],
                  mode='lines+markers', name='기준 시나리오',
                  line=dict(color='#6B7280', width=2), showlegend=False),
        row=1, col=2
    )
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        fig.add_trace(
            go.Scatter(x=years, y=scenario_data['defense_export'],
                      mode='lines+markers', name=scenario_name,
                      line=dict(color=colors[i % len(colors)], width=3), showlegend=False),
            row=1, col=2
        )
    
    # 3. 국산화율 비교
    fig.add_trace(
        go.Scatter(x=years, y=baseline['localization_rate'],
                  mode='lines+markers', name='기준 시나리오',
                  line=dict(color='#6B7280', width=2), showlegend=False),
        row=2, col=1
    )
    
    for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
        fig.add_trace(
            go.Scatter(x=years, y=scenario_data['localization_rate'],
                      mode='lines+markers', name=scenario_name,
                      line=dict(color=colors[i % len(colors)], width=3), showlegend=False),
        row=2, col=1
        )
    
    # 4. 총 비용 비교 (막대그래프)
    scenario_names = list(scenarios.keys())
    total_costs = [scenarios[name]['total_cost'] for name in scenario_names]
    
    fig.add_trace(
        go.Bar(x=scenario_names, y=total_costs,
               marker_color=colors[:len(scenario_names)],
               name='총 투자비용', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=700, title_text="🔀 정책 시나리오 비교 분석")
    fig.update_xaxes(title_text="연도", row=1, col=1)
    fig.update_xaxes(title_text="연도", row=1, col=2)
    fig.update_xaxes(title_text="연도", row=2, col=1)
    fig.update_xaxes(title_text="시나리오", row=2, col=2)
    fig.update_yaxes(title_text="건강등급", row=1, col=1)
    fig.update_yaxes(title_text="수출액 (억원)", row=1, col=2)
    fig.update_yaxes(title_text="국산화율 (%)", row=2, col=1)
    fig.update_yaxes(title_text="총비용 (억원)", row=2, col=2)
    
    return fig

def show_policy_effectiveness_ranking(scenarios, baseline):
    """정책 효과성 순위 표시 - HTML 제거하고 Streamlit 네이티브 방식 사용"""
    st.markdown("#### 🏆 정책 효과성 순위")
    
    effectiveness_scores = []
    
    for scenario_name, scenario_data in scenarios.items():
        # 효과성 점수 계산 (여러 지표 종합)
        health_improvement = baseline['health_grade'][0] - scenario_data['health_grade'][-1]
        export_growth = (scenario_data['defense_export'][-1] - baseline['defense_export'][-1]) / baseline['defense_export'][-1]
        localization_growth = scenario_data['localization_rate'][-1] - baseline['localization_rate'][-1]
        cost_efficiency = 1 / (scenario_data['total_cost'] / 10000)  # 비용 효율성
        
        total_score = (health_improvement * 20 + export_growth * 30 + 
                      localization_growth * 25 + cost_efficiency * 25)
        
        effectiveness_scores.append({
            'scenario': scenario_name,
            'score': total_score,
            'health_improvement': health_improvement,
            'export_growth': export_growth * 100,
            'localization_growth': localization_growth,
            'cost': scenario_data['total_cost']
        })
    
    # 점수순 정렬
    effectiveness_scores.sort(key=lambda x: x['score'], reverse=True)
    
    for i, score_data in enumerate(effectiveness_scores):
        if i == 0:  # 1순위
            st.success(f"🏆 {i+1}순위: {score_data['scenario']}")
            st.success(f"효과성 점수: {score_data['score']:.1f}/100")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"건강 개선: {score_data['health_improvement']:.2f}등급")
            with col2:
                st.success(f"수출 증가: {score_data['export_growth']:+.1f}%")
            st.success(f"총 투자비용: {score_data['cost']:,.0f}억원")
            
        elif i == 1:  # 2순위
            st.warning(f"🥈 {i+1}순위: {score_data['scenario']}")
            st.warning(f"효과성 점수: {score_data['score']:.1f}/100")
            col1, col2 = st.columns(2)
            with col1:
                st.warning(f"건강 개선: {score_data['health_improvement']:.2f}등급")
            with col2:
                st.warning(f"수출 증가: {score_data['export_growth']:+.1f}%")
            st.warning(f"총 투자비용: {score_data['cost']:,.0f}억원")
            
        else:  # 3순위 이하
            st.info(f"🥉 {i+1}순위: {score_data['scenario']}")
            st.info(f"효과성 점수: {score_data['score']:.1f}/100")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"건강 개선: {score_data['health_improvement']:.2f}등급")
            with col2:
                st.info(f"수출 증가: {score_data['export_growth']:+.1f}%")
            st.info(f"총 투자비용: {score_data['cost']:,.0f}억원")

def create_policy_simulation_dashboard():
    """정책 시뮬레이션 대시보드 메인"""
    st.header("🎮 정책 시뮬레이션 엔진")
    st.markdown("**다양한 정책 조합의 효과를 시뮬레이션하고 최적 전략을 도출합니다.**")
    
    # 시뮬레이션 엔진 초기화
    engine = PolicySimulationEngine()
    
    # 정책 선택 인터페이스
    st.markdown("### 🎯 정책 시나리오 설계")
    
    policy_options = list(engine.policy_effects.keys())
    policy_names = [engine.policy_effects[key]['name'] for key in policy_options]
    
    # 사전 정의된 시나리오들
    predefined_scenarios = {
        "🤖 AI 중심 전략": ['ai_health_investment', 'automation_acceleration'],
        "🚀 기술혁신 전략": ['rnd_expansion', 'automation_acceleration', 'cyber_security_enhancement'],
        "🔗 통합운영 전략": ['integrated_defense_health', 'ai_health_investment'],
        "🛡️ 종합보안 전략": ['cyber_security_enhancement', 'integrated_defense_health', 'rnd_expansion'],
        "💪 올인원 전략": policy_options  # 모든 정책 적용
    }
    
    st.markdown("#### 📋 사전 정의된 정책 패키지")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for scenario_name in list(predefined_scenarios.keys())[:3]:
            if st.button(scenario_name, key=f"preset_{scenario_name}"):
                st.session_state.selected_policies = predefined_scenarios[scenario_name]
    
    with col2:
        for scenario_name in list(predefined_scenarios.keys())[3:]:
            if st.button(scenario_name, key=f"preset_{scenario_name}"):
                st.session_state.selected_policies = predefined_scenarios[scenario_name]
    
    # 커스텀 정책 선택
    st.markdown("#### 🎛️ 커스텀 정책 조합")
    
    if 'selected_policies' not in st.session_state:
        st.session_state.selected_policies = []
    
    selected_policies = st.multiselect(
        "적용할 정책을 선택하세요",
        options=policy_options,
        default=st.session_state.selected_policies,
        format_func=lambda x: engine.policy_effects[x]['name'],
        key="custom_policies"
    )
    
    # 정책 상세 정보 표시
    if selected_policies:
        st.markdown("#### 📊 선택된 정책 상세 정보")
        
        for policy_key in selected_policies:
            policy = engine.policy_effects[policy_key]
            
            with st.expander(f"📋 {policy['name']} 상세"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("연간 비용", f"{policy['cost_per_year']:,}억원")
                    st.metric("시행 시점", f"{policy['implementation_time']}년 후")
                
                with col2:
                    # 정책별 주요 효과 표시
                    if 'health_grade_effect' in policy:
                        st.metric("건강등급 개선", f"{policy['health_grade_effect']:+.1f}등급")
                    if 'automation_boost' in policy:
                        st.metric("자동화 투자 증가", f"{(policy['automation_boost']-1)*100:+.0f}%")
                    if 'rnd_boost' in policy:
                        st.metric("R&D 투자 증가", f"{(policy['rnd_boost']-1)*100:+.0f}%")
    
    # 시뮬레이션 실행
    if selected_policies:
        st.markdown("---")
        st.markdown("### 🔮 시뮬레이션 결과")
        
        with st.spinner("정책 시뮬레이션 실행 중..."):
            # 기준 시나리오
            baseline = engine.baseline_data
            
            # 선택된 정책 시나리오 시뮬레이션
            simulated = simulate_policy_scenario(engine, selected_policies)
            
            # ROI 계산
            roi_analysis = calculate_policy_roi(baseline, simulated, simulated['total_cost'])
        
        # 핵심 결과 요약
        st.markdown("#### 🎯 핵심 시뮬레이션 결과")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_improvement = baseline['health_grade'][0] - simulated['health_grade'][-1]
            st.metric("건강등급 개선", f"{health_improvement:+.2f}등급")
        
        with col2:
            export_increase = ((simulated['defense_export'][-1] - baseline['defense_export'][-1]) / 
                             baseline['defense_export'][-1]) * 100
            st.metric("방산수출 증가", f"{export_increase:+.1f}%")
        
        with col3:
            localization_increase = simulated['localization_rate'][-1] - baseline['localization_rate'][-1]
            st.metric("국산화율 향상", f"{localization_increase:+.1f}%p")
        
        with col4:
            st.metric("투자 ROI", f"{roi_analysis['roi_percentage']:+.1f}%")
        
        # 상세 비교 차트
        scenarios = {"선택된 정책 조합": simulated}
        fig = create_scenario_comparison_chart(baseline, scenarios)
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI 상세 분석
        st.markdown("#### 💰 투자 수익률 상세 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💵 비용 구성**")
            st.metric("총 투자비용", f"{roi_analysis['total_cost']:,.0f}억원")
            st.metric("6년간 연평균", f"{roi_analysis['total_cost']/6:,.0f}억원")
            
        with col2:
            st.markdown("**📈 수익 구성**")
            st.metric("총 경제적 효과", f"{roi_analysis['total_benefits']:,.0f}억원")
            st.metric("회수 기간", f"{roi_analysis['payback_period']:.1f}년")
        
        # 수익 구성 상세
        st.markdown("**🔍 수익 구성 상세**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("방산수출 증가", f"{roi_analysis['export_gain']:,.0f}억원")
        
        with col2:
            st.metric("자동화 투자 효과", f"{roi_analysis['automation_value']:,.0f}억원")
        
        with col3:
            st.metric("건강비용 절감", f"{roi_analysis['health_savings']:,.0f}억원")
        
        with col4:
            st.metric("사이버 리스크 절감", f"{roi_analysis['cyber_savings']:,.0f}억원")
        
        # 정책 추천
        st.markdown("---")
        st.markdown("#### 🎯 AI 정책 추천")
        
        if roi_analysis['roi_percentage'] > 50:
            st.success(f"🚀 **강력 추천**: ROI {roi_analysis['roi_percentage']:.1f}%로 매우 효과적인 정책 조합입니다!")
        elif roi_analysis['roi_percentage'] > 20:
            st.warning(f"✅ **추천**: ROI {roi_analysis['roi_percentage']:.1f}%로 양호한 정책 조합입니다.")
        elif roi_analysis['roi_percentage'] > 0:
            st.info(f"⚠️ **주의 검토**: ROI {roi_analysis['roi_percentage']:.1f}%로 신중한 검토가 필요합니다.")
        else:
            st.error(f"❌ **재검토 필요**: ROI {roi_analysis['roi_percentage']:.1f}%로 정책 조합 재설계를 권장합니다.")
    
    else:
        st.info("👆 정책을 선택하면 시뮬레이션이 시작됩니다.")
    
    # 전체 시나리오 비교 (사전 정의된 시나리오들)
    st.markdown("---")
    st.markdown("### 🔀 전체 정책 시나리오 비교")
    
    if st.button("🔄 전체 시나리오 분석 실행", key="run_all_scenarios"):
        with st.spinner("모든 정책 시나리오 분석 중... (약 10초 소요)"):
            all_scenarios = {}
            
            for scenario_name, policies in predefined_scenarios.items():
                if scenario_name != "💪 올인원 전략":  # 너무 비싼 시나리오 제외
                    simulated = simulate_policy_scenario(engine, policies)
                    all_scenarios[scenario_name] = simulated
            
            # 전체 비교 차트
            fig_all = create_scenario_comparison_chart(baseline, all_scenarios)
            st.plotly_chart(fig_all, use_container_width=True)
            
            # 효과성 순위
            show_policy_effectiveness_ranking(all_scenarios, baseline)

if __name__ == "__main__":
    create_policy_simulation_dashboard()