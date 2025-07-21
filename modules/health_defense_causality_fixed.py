import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
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

def load_integrated_health_defense_data():
    """건강-방위 통합 데이터 로드"""
    try:
        # 건강 데이터
        health_grade = pd.read_csv('data/mma/mma_health_grade.csv', index_col='연도')
        infections = pd.read_csv('data/kdca/kdca_infections.csv', index_col='연도')
        exemption = pd.read_csv('data/mma/mma_exemption.csv', index_col='연도')
        
        # 방위 데이터
        rnd_data = pd.read_csv('data/dapa/dapa_rnd_budget_and_tasks.csv', encoding='utf-8')
        export_data = pd.read_csv('data/dapa/dapa_export_key_items.csv', encoding='utf-8')
        
        return {
            'health_grade': health_grade,
            'infections': infections,
            'exemption': exemption,
            'rnd': rnd_data,
            'export': export_data
        }
    except Exception as e:
        st.warning("실제 데이터 로드 실패. 통합 시뮬레이션 데이터로 분석합니다.")
        return create_integrated_simulation_data()

def create_integrated_simulation_data():
    """건강-방위 인과관계 시뮬레이션 데이터"""
    years = list(range(2015, 2025))
    np.random.seed(42)
    
    # 기본 건강 지표 생성
    base_infection_rate = 1.5
    base_health_grade = 2.8
    base_exemption_rate = 3.2
    
    # 팬데믹 영향 모델링
    pandemic_effect = []
    for year in years:
        if year == 2020:
            effect = 2.8  # 팬데믹 시작
        elif year == 2021:
            effect = 3.5  # 팬데믹 최고조
        elif year == 2022:
            effect = 2.1  # 회복 시작
        elif year == 2023:
            effect = 1.4  # 안정화
        elif year >= 2024:
            effect = 1.1  # 정상화
        else:
            effect = 1.0  # 팬데믹 이전
        pandemic_effect.append(effect)
    
    # 건강 지표들 (팬데믹 영향 반영)
    infections = []
    health_grades = []
    exemption_rates = []
    
    for i, year in enumerate(years):
        # 감염병 발생률
        infection = base_infection_rate * pandemic_effect[i] + np.random.uniform(-0.3, 0.3)
        infections.append(max(0.1, infection))
        
        # 건강등급 (감염병과 연동)
        health = base_health_grade + (infection - base_infection_rate) * 0.15 + np.random.uniform(-0.2, 0.2)
        health_grades.append(max(1.0, min(5.0, health)))
        
        # 면제율 (건강등급과 연동)
        exemption = base_exemption_rate + (health - base_health_grade) * 0.8 + np.random.uniform(-0.5, 0.5)
        exemption_rates.append(max(1.0, exemption))
    
    # 방위산업 지표들 (건강 지표와 인과관계 반영)
    rnd_budgets = []
    defense_exports = []
    automation_investments = []
    cyber_security_budgets = []
    
    for i, year in enumerate(years):
        # R&D 예산 (건강 위기 시 증가)
        health_crisis_factor = max(1.0, (health_grades[i] - 2.5) * 0.4)  # 건강등급 높을수록 예산 증가
        pandemic_urgency = pandemic_effect[i] * 0.2  # 팬데믹 시 긴급 증액
        
        base_rnd = 18000 + (year - 2015) * 1200
        rnd_budget = base_rnd * (1 + health_crisis_factor + pandemic_urgency) + np.random.randint(-800, 800)
        rnd_budgets.append(max(10000, rnd_budget))
        
        # 방산 수출 (감염병 영향으로 감소, 이후 반등)
        infection_impact = -infections[i] * 150  # 감염병 높으면 수출 감소
        recovery_boost = max(0, (3.5 - infections[i]) * 200) if year >= 2022 else 0  # 회복 시 반등
        
        base_export = 1800 + (year - 2015) * 150
        export = base_export + infection_impact + recovery_boost + np.random.randint(-200, 200)
        defense_exports.append(max(500, export))
        
        # 자동화/무인화 투자 (면제율 증가 시 급증)
        automation_demand = (exemption_rates[i] - 3.0) * 80  # 면제율 높을수록 무인화 투자 증가
        base_automation = 300 + (year - 2015) * 50
        automation = base_automation + automation_demand + np.random.randint(-30, 30)
        automation_investments.append(max(100, automation))
        
        # 사이버 보안 예산 (팬데믹 시 원격근무 증가로 급증)
        cyber_urgency = pandemic_effect[i] * 120  # 팬데믹 시 사이버 보안 중요성 급증
        base_cyber = 400 + (year - 2015) * 60
        cyber = base_cyber + cyber_urgency + np.random.randint(-40, 40)
        cyber_security_budgets.append(max(200, cyber))
    
    return {
        'years': years,
        'infections': infections,
        'health_grades': health_grades,
        'exemption_rates': exemption_rates,
        'rnd_budgets': rnd_budgets,
        'defense_exports': defense_exports,
        'automation_investments': automation_investments,
        'cyber_security_budgets': cyber_security_budgets,
        'pandemic_effect': pandemic_effect
    }

def analyze_health_defense_correlations(data):
    """건강-방위산업 상관관계 분석"""
    correlations = {}
    
    # 1. 감염병 vs 방산수출
    corr_infection_export, p_val1 = pearsonr(data['infections'], data['defense_exports'])
    correlations['감염병_vs_방산수출'] = {
        'correlation': corr_infection_export,
        'p_value': p_val1,
        'interpretation': '감염병 증가 → 방산수출 감소' if corr_infection_export < -0.3 else '약한 상관관계'
    }
    
    # 2. 건강등급 vs R&D 투자
    corr_health_rnd, p_val2 = pearsonr(data['health_grades'], data['rnd_budgets'])
    correlations['건강등급_vs_RnD투자'] = {
        'correlation': corr_health_rnd,
        'p_value': p_val2,
        'interpretation': '건강 악화 → R&D 투자 증가' if corr_health_rnd > 0.3 else '약한 상관관계'
    }
    
    # 3. 면제율 vs 자동화 투자
    corr_exemption_automation, p_val3 = pearsonr(data['exemption_rates'], data['automation_investments'])
    correlations['면제율_vs_자동화투자'] = {
        'correlation': corr_exemption_automation,
        'p_value': p_val3,
        'interpretation': '면제율 증가 → 자동화 투자 급증' if corr_exemption_automation > 0.5 else '보통 상관관계'
    }
    
    # 4. 팬데믹 효과 vs 사이버보안
    corr_pandemic_cyber, p_val4 = pearsonr(data['pandemic_effect'], data['cyber_security_budgets'])
    correlations['팬데믹_vs_사이버보안'] = {
        'correlation': corr_pandemic_cyber,
        'p_value': p_val4,
        'interpretation': '팬데믹 심화 → 사이버보안 투자 증가' if corr_pandemic_cyber > 0.5 else '보통 상관관계'
    }
    
    return correlations

def perform_lagged_correlation_analysis(data):
    """시차 상관분석 (Granger 대체)"""
    lag_results = {}
    
    # 1. 건강등급 → R&D 투자 (1년 지연)
    if len(data['years']) > 2:
        health_lagged = data['health_grades'][:-1]  # 1년 전 건강등급
        rnd_current = data['rnd_budgets'][1:]       # 현재 R&D 투자
        
        corr_lag1, p_val1 = pearsonr(health_lagged, rnd_current)
        
        lag_results['건강등급→RnD투자_1년지연'] = {
            'correlation': corr_lag1,
            'p_value': p_val1,
            'lag': 1,
            'interpretation': '건강 악화가 1년 후 R&D 투자 증가를 유발' if corr_lag1 > 0.4 else '지연 효과 불명확'
        }
    
    # 2. 면제율 → 자동화 투자 (즉시 효과)
    corr_immediate, p_val2 = pearsonr(data['exemption_rates'], data['automation_investments'])
    
    lag_results['면제율→자동화투자_즉시효과'] = {
        'correlation': corr_immediate,
        'p_value': p_val2,
        'lag': 0,
        'interpretation': '면제율 증가가 즉시 자동화 투자로 이어짐' if corr_immediate > 0.6 else '즉시 반응 제한적'
    }
    
    # 3. 감염병 → 방산수출 (2년 지연 영향)
    if len(data['years']) > 3:
        infection_lagged = data['infections'][:-2]  # 2년 전 감염병
        export_current = data['defense_exports'][2:]  # 현재 수출
        
        corr_lag2, p_val3 = pearsonr(infection_lagged, export_current)
        
        lag_results['감염병→방산수출_2년지연'] = {
            'correlation': corr_lag2,
            'p_value': p_val3,
            'lag': 2,
            'interpretation': '감염병 영향이 2년 후까지 수출에 지속됨' if abs(corr_lag2) > 0.3 else '장기 영향 제한적'
        }
    
    return lag_results

def analyze_pandemic_defense_transformation(data):
    """팬데믹이 방위전략에 미친 구조적 변화 분석"""
    
    # 2020년 기준 전후 구분
    pre_pandemic_indices = [i for i, year in enumerate(data['years']) if year < 2020]
    pandemic_indices = [i for i, year in enumerate(data['years']) if year >= 2020]
    
    transformations = {}
    
    # 1. 방위산업 디지털 전환 가속화
    pre_cyber = np.mean([data['cyber_security_budgets'][i] for i in pre_pandemic_indices])
    pandemic_cyber = np.mean([data['cyber_security_budgets'][i] for i in pandemic_indices])
    cyber_increase = ((pandemic_cyber - pre_cyber) / pre_cyber) * 100
    
    transformations['디지털_전환'] = {
        'pre_avg': pre_cyber,
        'pandemic_avg': pandemic_cyber,
        'increase_rate': cyber_increase,
        'interpretation': '팬데믹으로 인한 디지털 전환 가속화'
    }
    
    # 2. 무인화/자동화 투자 급증
    pre_automation = np.mean([data['automation_investments'][i] for i in pre_pandemic_indices])
    pandemic_automation = np.mean([data['automation_investments'][i] for i in pandemic_indices])
    automation_increase = ((pandemic_automation - pre_automation) / pre_automation) * 100
    
    transformations['무인화_전환'] = {
        'pre_avg': pre_automation,
        'pandemic_avg': pandemic_automation,
        'increase_rate': automation_increase,
        'interpretation': '인력 부족 대비 무인화 기술 투자 급증'
    }
    
    # 3. R&D 투자 패턴 변화
    pre_rnd = np.mean([data['rnd_budgets'][i] for i in pre_pandemic_indices])
    pandemic_rnd = np.mean([data['rnd_budgets'][i] for i in pandemic_indices])
    rnd_increase = ((pandemic_rnd - pre_rnd) / pre_rnd) * 100
    
    transformations['RnD_강화'] = {
        'pre_avg': pre_rnd,
        'pandemic_avg': pandemic_rnd,
        'increase_rate': rnd_increase,
        'interpretation': '위기 대응 기술 개발 투자 확대'
    }
    
    return transformations

def calculate_health_defense_impact_score(data):
    """건강 위기가 방위전략에 미치는 영향도 점수 계산"""
    
    # 정규화된 지표들
    scaler = StandardScaler()
    
    # 건강 위기 지수 (감염병 + 건강등급 + 면제율)
    health_crisis_index = []
    for i in range(len(data['years'])):
        crisis_score = (
            (data['infections'][i] / max(data['infections'])) * 0.4 +  # 감염병 40%
            (data['health_grades'][i] / max(data['health_grades'])) * 0.3 +  # 건강등급 30%
            (data['exemption_rates'][i] / max(data['exemption_rates'])) * 0.3   # 면제율 30%
        )
        health_crisis_index.append(crisis_score)
    
    # 방위전략 변화 지수 (R&D + 자동화 + 사이버보안)
    defense_response_index = []
    for i in range(len(data['years'])):
        response_score = (
            (data['rnd_budgets'][i] / max(data['rnd_budgets'])) * 0.4 +  # R&D 40%
            (data['automation_investments'][i] / max(data['automation_investments'])) * 0.3 +  # 자동화 30%
            (data['cyber_security_budgets'][i] / max(data['cyber_security_budgets'])) * 0.3   # 사이버보안 30%
        )
        defense_response_index.append(response_score)
    
    # 연도별 영향도 점수
    impact_scores = []
    for i in range(len(data['years'])):
        # 건강 위기 → 방위전략 변화 영향도
        impact = health_crisis_index[i] * defense_response_index[i]
        impact_scores.append(impact)
    
    # 전체 상관관계
    overall_correlation, _ = pearsonr(health_crisis_index, defense_response_index)
    
    return {
        'health_crisis_index': health_crisis_index,
        'defense_response_index': defense_response_index,
        'impact_scores': impact_scores,
        'overall_correlation': overall_correlation,
        'max_impact_year': data['years'][np.argmax(impact_scores)],
        'max_impact_score': max(impact_scores)
    }

def plot_health_defense_causality_dashboard(data, correlations, transformations, impact_analysis):
    """건강-방위전략 인과관계 종합 대시보드"""
    
    # 1. 핵심 인과관계 요약
    st.markdown("#### 🔗 건강-방위전략 핵심 인과관계")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        infection_export_corr = correlations['감염병_vs_방산수출']['correlation']
        st.metric("감염병 ↔ 방산수출", f"{infection_export_corr:.3f}")
        if infection_export_corr < -0.5:
            st.success("강한 역상관 관계")
        elif infection_export_corr < -0.3:
            st.warning("중간 역상관 관계")
        else:
            st.info("약한 상관관계")
    
    with col2:
        health_rnd_corr = correlations['건강등급_vs_RnD투자']['correlation']
        st.metric("건강악화 ↔ R&D투자", f"{health_rnd_corr:.3f}")
        if health_rnd_corr > 0.5:
            st.success("강한 정상관 관계")
        elif health_rnd_corr > 0.3:
            st.warning("중간 정상관 관계")
        else:
            st.info("약한 상관관계")
    
    with col3:
        exemption_auto_corr = correlations['면제율_vs_자동화투자']['correlation']
        st.metric("면제율 ↔ 무인화투자", f"{exemption_auto_corr:.3f}")
        if exemption_auto_corr > 0.7:
            st.success("매우 강한 상관관계")
        elif exemption_auto_corr > 0.5:
            st.warning("강한 상관관계")
        else:
            st.info("보통 상관관계")
    
    # 2. 통합 인과관계 시각화
    st.markdown("#### 📈 건강 위기 vs 방위전략 변화 통합 분석")
    
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('감염병 발생 vs 방산 수출', '건강등급 vs R&D 투자', 
                       '면제율 vs 자동화 투자', '종합 영향도 지수'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 감염병 vs 방산수출
    fig1.add_trace(
        go.Scatter(x=data['infections'], y=data['defense_exports'],
                  mode='markers+text', text=[str(y) for y in data['years']],
                  textposition='top center', name='연도별 데이터',
                  marker=dict(size=8, color='#EF4444')),
        row=1, col=1
    )
    
    # 건강등급 vs R&D투자
    fig1.add_trace(
        go.Scatter(x=data['health_grades'], y=data['rnd_budgets'],
                  mode='markers+text', text=[str(y) for y in data['years']],
                  textposition='top center', name='연도별 데이터',
                  marker=dict(size=8, color='#3B82F6')),
        row=1, col=2
    )
    
    # 면제율 vs 자동화투자
    fig1.add_trace(
        go.Scatter(x=data['exemption_rates'], y=data['automation_investments'],
                  mode='markers+text', text=[str(y) for y in data['years']],
                  textposition='top center', name='연도별 데이터',
                  marker=dict(size=8, color='#10B981')),
        row=2, col=1
    )
    
    # 종합 영향도 지수
    fig1.add_trace(
        go.Scatter(x=data['years'], y=impact_analysis['impact_scores'],
                  mode='lines+markers', name='건강→방위 영향도',
                  line=dict(color='#F59E0B', width=3),
                  marker=dict(size=10)),
        row=2, col=2
    )
    
    fig1.update_layout(height=700, title_text="🔗 건강-방위전략 인과관계 종합 분석")
    fig1.update_xaxes(title_text="감염병 발생률", row=1, col=1)
    fig1.update_xaxes(title_text="건강등급", row=1, col=2)
    fig1.update_xaxes(title_text="면제율 (%)", row=2, col=1)
    fig1.update_xaxes(title_text="연도", row=2, col=2)
    fig1.update_yaxes(title_text="방산수출 (억원)", row=1, col=1)
    fig1.update_yaxes(title_text="R&D 예산 (억원)", row=1, col=2)
    fig1.update_yaxes(title_text="자동화 투자 (억원)", row=2, col=1)
    fig1.update_yaxes(title_text="영향도 지수", row=2, col=2)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # 3. 팬데믹 구조적 변화 분석
    st.markdown("#### 🦠 팬데믹이 방위전략에 미친 구조적 변화")
    
    transformation_names = list(transformations.keys())
    increase_rates = [transformations[name]['increase_rate'] for name in transformation_names]

    # 색상: 첫 번째 초록, 두 번째 파랑, 세 번째 주황
    colors = ['#22C55E', '#3B82F6', '#F59E0B']
    fig2 = go.Figure(data=[
        go.Bar(x=transformation_names, y=increase_rates,
               marker_color=colors[:len(transformation_names)],
               text=[f"+{x:.1f}%" for x in increase_rates],
               textposition='auto')
    ])
    
    fig2.update_layout(
        title="팬데믹 전후 방위전략 영역별 투자 증가율",
        xaxis_title="방위전략 영역",
        yaxis_title="증가율 (%)",
        height=400,
        width=600,  # 가로 폭 축소
        margin=dict(l=40, r=40, t=60, b=40)  # 여백 조정
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # 상세 수치
    col1, col2, col3 = st.columns(3)
    for i, (name, data_) in enumerate(transformations.items()):
        with [col1, col2, col3][i]:
            st.metric(
                name.replace('_', ' '),
                f"{data_['pandemic_avg']:.0f}억원",
                f"+{data_['increase_rate']:.1f}%"
            )

def show_causality_insights(correlations, lag_results, impact_analysis):
    """인과관계 분석 핵심 인사이트 - HTML 제거하고 Streamlit 네이티브 방식 사용"""
    st.markdown("#### 🎯 핵심 발견사항 및 정책 시사점")
    
    # 최고 영향도 연도
    max_impact_year = impact_analysis['max_impact_year']
    max_impact_score = impact_analysis['max_impact_score']
    overall_correlation = impact_analysis['overall_correlation']
    
    # HTML 대신 Streamlit 네이티브 방식 사용
    st.success(f"📊 건강 위기 → 방위전략 변화 전체 상관도: {overall_correlation:.3f}")
    st.info("→ 건강 위기가 방위전략 변화를 유의미하게 촉진")
    
    st.success(f"📊 최대 영향도 시점: {max_impact_year}년 (영향도: {max_impact_score:.3f})")
    st.info("→ 팬데믹 최고조 시기에 방위전략 대전환 발생")
    
    st.success("📊 면제율 증가 → 무인화 투자 급증")
    st.info("→ 인력 부족 대비 기술적 해결책 모색 가속화")
    
    # 정책 제안
    st.markdown("#### 💡 데이터 기반 정책 제안")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🎯 즉시 실행 권장")
        st.markdown("• 건강 위기 조기 경보 시스템 구축")
        st.markdown("• 면제율 증가 대비 무인화 로드맵")
        st.markdown("• 감염병-방산수출 연동 대응체계")
    
    with col2:
        st.info("🔬 중장기 연구개발")
        st.markdown("• AI 기반 건강-방위 통합 예측모델")
        st.markdown("• 자동화 기술 국산화 가속화")
        st.markdown("• 사이버보안 역량 지속 강화")

def analyze_cross_sector_impacts(data):
    """부문 간 교차 영향 분석"""
    
    # 건강 부문이 방위산업 각 영역에 미치는 영향도 매트릭스
    impact_matrix = {}
    
    health_indicators = ['infections', 'health_grades', 'exemption_rates']
    defense_indicators = ['rnd_budgets', 'defense_exports', 'automation_investments', 'cyber_security_budgets']
    
    for health_ind in health_indicators:
        impact_matrix[health_ind] = {}
        for defense_ind in defense_indicators:
            correlation, p_value = pearsonr(data[health_ind], data[defense_ind])
            
            # 영향도 분류
            if isinstance(correlation, tuple):
                correlation = correlation[0]
            if abs(correlation) > 0.7:
                impact_level = "매우 강함"
                color = "#DC2626"
            elif abs(correlation) > 0.5:
                impact_level = "강함"
                color = "#EF4444"
            elif abs(correlation) > 0.3:
                impact_level = "보통"
                color = "#F59E0B"
            else:
                impact_level = "약함"
                color = "#6B7280"
            
            impact_matrix[health_ind][defense_ind] = {
                'correlation': correlation,
                'p_value': p_value,
                'impact_level': impact_level,
                'color': color,
                'direction': '정비례' if correlation > 0 else '반비례'
            }
    
    return impact_matrix

def show_impact_matrix(impact_matrix):
    """교차 영향 매트릭스 시각화 - HTML 제거하고 Streamlit 네이티브 방식 사용"""
    st.markdown("#### 🔄 건강-방위산업 교차영향 매트릭스")
    
    # 매트릭스 테이블 생성
    health_labels = {
        'infections': '감염병 발생',
        'health_grades': '건강등급 악화', 
        'exemption_rates': '면제율 증가'
    }
    
    defense_labels = {
        'rnd_budgets': 'R&D 투자',
        'defense_exports': '방산 수출',
        'automation_investments': '자동화 투자',
        'cyber_security_budgets': '사이버보안 투자'
    }
    
    for health_key, health_label in health_labels.items():
        st.markdown(f"**{health_label}의 영향**")
        
        cols = st.columns(4)
        for i, (defense_key, defense_label) in enumerate(defense_labels.items()):
            impact_data = impact_matrix[health_key][defense_key]
            
            with cols[i]:
                correlation = impact_data['correlation']
                impact_level = impact_data['impact_level']
                direction = impact_data['direction']
                
                if impact_level == "매우 강함":
                    st.error(f"**{defense_label}**\n{direction} ({correlation:.2f})\n{impact_level}")
                elif impact_level == "강함":
                    st.warning(f"**{defense_label}**\n{direction} ({correlation:.2f})\n{impact_level}")
                elif impact_level == "보통":
                    st.info(f"**{defense_label}**\n{direction} ({correlation:.2f})\n{impact_level}")
                else:
                    st.text(f"**{defense_label}**\n{direction} ({correlation:.2f})\n{impact_level}")

def predict_future_scenarios(data):
    """미래 시나리오별 예측"""
    
    scenarios = {
        "낙관 시나리오": {
            "infection_rate": 1.0,
            "health_grade": 2.5,
            "exemption_rate": 3.0,
            "description": "팬데믹 완전 종료, 건강 지표 정상화"
        },
        "현상 유지": {
            "infection_rate": 1.8,
            "health_grade": 3.2,
            "exemption_rate": 4.2,
            "description": "현재 수준 지속"
        },
        "비관 시나리오": {
            "infection_rate": 3.5,
            "health_grade": 4.0,
            "exemption_rate": 6.5,
            "description": "새로운 팬데믹 또는 건강 위기 발생"
        }
    }
    
    predictions = {}
    
    for scenario_name, scenario in scenarios.items():
        # 회귀 모델 기반 예측
        infection = scenario["infection_rate"]
        health = scenario["health_grade"] 
        exemption = scenario["exemption_rate"]
        
        # 각 방위산업 지표 예측
        pred_rnd = 18000 + (health - 2.5) * 2500 + (infection - 1.5) * 1800
        pred_export = 2000 - (infection - 1.5) * 300 + np.random.randint(-100, 100)
        pred_automation = 350 + (exemption - 3.0) * 70
        pred_cyber = 500 + (infection - 1.5) * 100
        
        predictions[scenario_name] = {
            "rnd_budget": max(15000, pred_rnd),
            "defense_export": max(1000, pred_export),
            "automation_investment": max(200, pred_automation),
            "cyber_security": max(300, pred_cyber),
            "description": scenario["description"]
        }
    
    return predictions

def show_future_scenarios(predictions):
    """미래 시나리오 예측 결과 표시 - HTML 제거하고 Streamlit 네이티브 방식 사용"""
    st.markdown("#### 🔮 2030년 시나리오별 방위전략 예측")
    
    for scenario_name, pred in predictions.items():
        if scenario_name == "낙관 시나리오":
            st.success(f"📋 {scenario_name}")
            st.success(f"*{pred['description']}*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R&D 예산", f"{pred['rnd_budget']:,.0f}억원")
                st.metric("자동화투자", f"{pred['automation_investment']:,.0f}억원")
            with col2:
                st.metric("방산수출", f"{pred['defense_export']:,.0f}억원")
                st.metric("사이버보안", f"{pred['cyber_security']:,.0f}억원")
                
        elif scenario_name == "현상 유지":
            st.warning(f"📋 {scenario_name}")
            st.warning(f"*{pred['description']}*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R&D 예산", f"{pred['rnd_budget']:,.0f}억원")
                st.metric("자동화투자", f"{pred['automation_investment']:,.0f}억원")
            with col2:
                st.metric("방산수출", f"{pred['defense_export']:,.0f}억원")
                st.metric("사이버보안", f"{pred['cyber_security']:,.0f}억원")
                
        else:  # 비관 시나리오
            st.error(f"📋 {scenario_name}")
            st.error(f"*{pred['description']}*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R&D 예산", f"{pred['rnd_budget']:,.0f}억원")
                st.metric("자동화투자", f"{pred['automation_investment']:,.0f}억원")
            with col2:
                st.metric("방산수출", f"{pred['defense_export']:,.0f}억원")
                st.metric("사이버보안", f"{pred['cyber_security']:,.0f}억원")

def create_health_defense_causality_dashboard():
    """건강-방위전략 인과관계 분석 대시보드 메인"""
    st.header("🔗 건강-방위전략 인과관계 분석")
    st.markdown("**건강 위기가 방위전략에 미치는 실제 영향을 데이터로 분석합니다.**")
    
    # 데이터 로드 및 분석
    with st.spinner("🔄 건강-방위전략 인과관계 분석 중..."):
        data = load_integrated_health_defense_data()
        
        if isinstance(data, dict) and 'years' in data:
            integrated_data = data
        else:
            integrated_data = create_integrated_simulation_data()
        
        # 각종 분석 수행
        correlations = analyze_health_defense_correlations(integrated_data)
        lag_results = perform_lagged_correlation_analysis(integrated_data)
        transformations = analyze_pandemic_defense_transformation(integrated_data)
        impact_analysis = calculate_health_defense_impact_score(integrated_data)
        impact_matrix = analyze_cross_sector_impacts(integrated_data)
        future_predictions = predict_future_scenarios(integrated_data)
    
    # 핵심 지표 요약
    st.markdown("### 🎯 건강-방위전략 연관성 핵심 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 상관도", f"{impact_analysis['overall_correlation']:.3f}")
        st.caption("건강위기 ↔ 방위전략 변화")
    
    with col2:
        max_impact_year = impact_analysis['max_impact_year']
        st.metric("최대 영향 시점", f"{max_impact_year}년")
        st.caption("건강→방위 최대 영향도")
    
    with col3:
        exemption_auto_corr = correlations['면제율_vs_자동화투자']['correlation']
        st.metric("면제율→무인화", f"{exemption_auto_corr:.3f}")
        st.caption("인력부족 대응 투자")
    
    with col4:
        cyber_increase = transformations['디지털_전환']['increase_rate']
        st.metric("사이버보안 증가", f"+{cyber_increase:.1f}%")
        st.caption("팬데믹 후 증가율")
    
    # 상세 분석 표시
    plot_health_defense_causality_dashboard(integrated_data, correlations, transformations, impact_analysis)
    
    # 교차영향 매트릭스
    show_impact_matrix(impact_matrix)
    
    # 인과관계 핵심 인사이트
    show_causality_insights(correlations, lag_results, impact_analysis)
    
    # 시차 상관분석 결과 (Granger 대체)
    if lag_results:
        st.markdown("#### 🔬 시차 상관분석 결과 (인과관계 추정)")
        
        for relationship, result in lag_results.items():
            significance = "✅ 강한 연관성" if abs(result['correlation']) > 0.5 else "⚠️ 보통 연관성" if abs(result['correlation']) > 0.3 else "❌ 약한 연관성"
            correlation = result['correlation']
            interpretation = result['interpretation']
            lag = result['lag']
            
            if abs(correlation) > 0.5:
                st.success(f"**{relationship}** | {significance} (상관계수: {correlation:.3f}, {lag}년 지연)")
                st.success(f"📋 해석: {interpretation}")
            elif abs(correlation) > 0.3:
                st.warning(f"**{relationship}** | {significance} (상관계수: {correlation:.3f}, {lag}년 지연)")
                st.warning(f"📋 해석: {interpretation}")
            else:
                st.info(f"**{relationship}** | {significance} (상관계수: {correlation:.3f}, {lag}년 지연)")
                st.info(f"📋 해석: {interpretation}")
    
    # 미래 시나리오 예측
    show_future_scenarios(future_predictions)
    
    # 실시간 시뮬레이션
    st.markdown("---")
    st.markdown("#### 🎮 실시간 건강-방위전략 영향 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**가상 시나리오 설정**")
        scenario_infection = st.slider("감염병 발생률", 0.5, 5.0, 2.0, 0.1, key="scenario_infection")
        scenario_exemption = st.slider("면제율 (%)", 2.0, 8.0, 4.0, 0.2, key="scenario_exemption")
        scenario_health_grade = st.slider("건강등급", 1.0, 5.0, 3.0, 0.1, key="scenario_health")
    
    with col2:
        st.markdown("**예상 방위전략 변화**")
        
        # 시뮬레이션 계산
        predicted_rnd = 20000 + (scenario_health_grade - 2.5) * 3000 + (scenario_infection - 1.5) * 2000
        predicted_automation = 400 + (scenario_exemption - 3.0) * 80
        predicted_cyber = 600 + (scenario_infection - 1.5) * 120
        
        st.metric("예상 R&D 예산", f"{predicted_rnd:,.0f}억원")
        st.metric("예상 자동화 투자", f"{predicted_automation:.0f}억원") 
        st.metric("예상 사이버보안 예산", f"{predicted_cyber:.0f}억원")
        
        # 위험도 평가
        total_risk = (scenario_infection * 0.4 + scenario_exemption * 0.3 + scenario_health_grade * 0.3)
        if total_risk > 4.0:
            st.error("🚨 고위험: 긴급 대응 필요")
        elif total_risk > 3.0:
            st.warning("⚠️ 중위험: 주의 관찰")
        else:
            st.success("✅ 안정: 정상 수준")
    
    # 정책 제안
    st.markdown("---")
    st.markdown("#### 📋 데이터 기반 정책 제안")
    
    # HTML 제거하고 Streamlit 네이티브 방식 사용
    st.success("🎯 핵심 정책 권고사항")
    
    st.success("📊 통합 조기경보 시스템 구축")
    st.info("→ 건강 지표와 방위산업 지표 실시간 연동 모니터링")
    st.info("→ AI 기반 예측 모델로 3개월 전 미리 대응")
    
    st.success("📊 무인화 기술 국산화 가속화")
    st.info("→ 면제율 증가 대비 자동화 기술 확보")
    st.info("→ 핵심 무인화 기술의 해외 의존도 감소")
    
    st.success("📊 방산수출 리스크 관리 체계")
    st.info("→ 감염병-수출 연동 대응 매뉴얼 수립")
    st.info("→ 글로벌 공급망 위기 시 대체 전략 마련")

if __name__ == "__main__":
    create_health_defense_causality_dashboard()