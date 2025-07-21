import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import warnings

# 기존 분석 모듈들 import
from modules.rnd_analysis import *
from modules.localization_analysis import *
from modules.tech_trend_analysis import *
from modules.import_export_analysis import *
from modules.city_analysis import *
from modules.donut_charts import *
from pandemic_military_analysis import create_pandemic_military_dashboard
from modules.future_strategy import show_future_strategy_box
from modules.health_prediction import create_health_prediction_dashboard

# 새로 추가된 고급 분석 모듈들 import
try:
    from modules.strategy_effectiveness_analysis import create_strategy_effectiveness_dashboard
    STRATEGY_ANALYSIS_AVAILABLE = True
except ImportError:
    STRATEGY_ANALYSIS_AVAILABLE = False
    st.warning("전략 효과 분석 모듈을 찾을 수 없습니다.")

try:
    from modules.health_defense_causality_fixed import create_health_defense_causality_dashboard
    CAUSALITY_ANALYSIS_AVAILABLE = True
except ImportError:
    CAUSALITY_ANALYSIS_AVAILABLE = False
    st.warning("건강-방위전략 인과관계 분석 모듈을 찾을 수 없습니다.")

try:
    from modules.policy_simulation_engine import create_policy_simulation_dashboard
    POLICY_SIMULATION_AVAILABLE = True
except ImportError:
    POLICY_SIMULATION_AVAILABLE = False
    st.warning("정책 시뮬레이션 엔진을 찾을 수 없습니다.")

# 페이지 설정
st.set_page_config(
    page_title="팬데믹 국방력 혁신 대시보드",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 상단 고정 배너 스타일 및 삽입
st.markdown("""
    <style>
        .block-container {
            padding-top:  0rem !important;
            padding-bottom: 1rem;
        }
        .fixed-banner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 54px;
            background-color: rgba(15, 23, 42, 0.95);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .fixed-banner-text {
            color: #F3F4F6;
            font-size: 22px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .fixed-banner-desc {
            color: #F3F4F6;
            font-size: 15px;
            margin-left: 18px;
            opacity: 0.85;
        }
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 5px solid #F97316;
        }
        [data-testid="stSidebar"] * {
            color: #F3F4F6 !important;
            font-weight: 500;
        }
        .menu-box {
            background-color: #0F172A;
            border-left: 5px solid #F97316;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
        }
        .menu-box:hover {
            background-color: #1E293B;
        }
        .menu-box a {
            color: #F3F4F6;
            text-decoration: none;
            font-weight: bold;
        }
        .main {
            margin-top: 54px !important;
        }
    </style>
    <div class="fixed-banner">
        <span class="fixed-banner-text">🛡️ 팬데믹 국방력 혁신 대시보드</span>
        <span class="fixed-banner-desc">팬데믹이 군인 건강과 국방력 혁신에 미치는 영향 분석</span>
    </div>
""", unsafe_allow_html=True)

# CSS 스타일링
def load_css():
    st.markdown("""
    <style>
    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 헤더 스타일 */
    .header {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 50%, #ff6b35 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        color: white;
        text-align: center;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* 섹션 스타일 */
    .section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #ff6b35;
    }
    
    .section h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff6b35;
        padding-bottom: 0.5rem;
    }
    
    /* 카드 스타일 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card h3 {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255,107,53,0.3);
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Noto Sans KR 웹폰트 및 전체 font-family 적용
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class^="css"], .main, .fixed-banner, .fixed-banner-text, .fixed-banner-desc, .menu-box, .section, .metric-card, .stDataFrame, .stTable, .stMarkdown, .stText, .stHeader, .stSubheader {
            font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
        }
        .stDataFrame, .stTable {
            font-size: 16px !important;
        }
        .metric-card, .section, .fixed-banner, .fixed-banner-text, .fixed-banner-desc {
            text-align: left !important;
        }
    </style>
""", unsafe_allow_html=True)

# 이미지 로드 함수
def load_image(image_path):
    try:
        with open(image_path, "rb") as f:
            image = f.read()
        return base64.b64encode(image).decode()
    except:
        return None

# 메인 함수
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    load_css()
    
    # 헤더
    st.markdown("""
    <div class="header fade-in">
        <h1>🛡️ 팬데믹 국방력 혁신 대시보드</h1>
        <p>팬데믹이 군인 건강과 국방력 혁신에 미치는 영향 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if 'current_menu' not in st.session_state:
        st.session_state.current_menu = 'overview'

    # 사이드바 네비게이션 (기존 스타일 유지 + 클릭 기능 추가)
    with st.sidebar:
        st.markdown("""
            <h3 style="margin-bottom: 20px;">탐색 메뉴</h3>
        """, unsafe_allow_html=True)
        
        # 각 메뉴를 버튼으로 만들되 기존 스타일 적용
        st.markdown("""
        <style>
        .menu-button {
            background-color: #0F172A;
            border-left: 5px solid #F97316;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
            width: 100%;
            border: none;
            color: #F3F4F6;
            text-align: left;
            cursor: pointer;
        }
        .menu-button:hover {
            background-color: #1E293B;
        }
        .stButton > button {
            background-color: #0F172A !important;
            border-left: 5px solid #F97316 !important;
            padding: 12px !important;
            margin-bottom: 10px !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            color: #F3F4F6 !important;
            width: 100% !important;
            text-align: left !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #1E293B !important;
        }
        .stButton > button:focus {
            background-color: #1E293B !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 메뉴 버튼들 (기존 디자인 유지)
        if st.button("\U0001F4CA 대시보드 개요", key="menu_overview"):
            st.session_state.current_menu = 'overview'
        
        if st.button("\U0001F525 팬데믹 국방력 영향분석", key="menu_pandemic"):
            st.session_state.current_menu = 'pandemic_military'
        
        if st.button("\U0001F916 건강위험도 AI 예측", key="menu_health"):
            st.session_state.current_menu = 'health_prediction'
        
        if st.button("\U0001F3D9️ 도시별 분석", key="menu_city"):
            st.session_state.current_menu = 'city_analysis'
        
        if st.button("\U0001F3AF 정책 제안", key="menu_policy"):
            st.session_state.current_menu = 'policy'
        
        # 추가 분석 (접기 메뉴)
        with st.expander("\U0001F4C2 추가 분석"):
            if st.button("\U0001F4B0 R&D 투자 변화", key="menu_rnd"):
                st.session_state.current_menu = 'rnd'
            
            if st.button("\U0001F3ED 국산화율 분석", key="menu_localization"):
                st.session_state.current_menu = 'localization'
            
            if st.button("\U0001F680 신기술 트렌드", key="menu_tech"):
                st.session_state.current_menu = 'tech_trend'
            
            if st.button("\U0001F30D 수출입 분석", key="menu_import_export"):
                st.session_state.current_menu = 'import_export'
            
            if st.button("\U0001F4CA 전략 효과 분석", key="menu_strategy"):
                st.session_state.current_menu = 'strategy_effectiveness'
            
            if st.button("\U0001F517 건강-방위전략 인과관계", key="menu_causality"):
                st.session_state.current_menu = 'health_defense_causality'
            
            if st.button("\U0001F3AE 정책 시뮬레이션", key="menu_simulation"):
                st.session_state.current_menu = 'policy_simulation'
            
            if st.button("\U0001F680 미래 전략", key="menu_future"):
                st.session_state.current_menu = 'future'
        
        # 기존 지표 표시 유지
        st.markdown("### 📊 핵심 성과지표")
        st.sidebar.markdown('''
<div style='background:#A78BFA; color:white; border-radius:10px; padding:10px 0; margin-bottom:10px; text-align:center; font-size:1.1em; width:180px; margin-left:auto; margin-right:auto;'>
  <div style='font-size:1em;'>면제율 개선도</div>
  <div style='font-size:1.3em; font-weight:bold;'>0.083%p↓</div>
</div>
<div style='background:#34D399; color:white; border-radius:10px; padding:10px 0; text-align:center; font-size:1.1em; width:180px; margin-left:auto; margin-right:auto;'>
  <div style='font-size:1em;'>R&D 예산 증가</div>
  <div style='font-size:1.3em; font-weight:bold;'>+24.3%</div>
</div>
''', unsafe_allow_html=True)

        # 분석 요약 (병무청 발표용)
        # st.sidebar.markdown("---")
        # st.sidebar.markdown("### 📋 분석 요약")
        # st.sidebar.markdown("""
# **분석 기간**: 2019-2023년 (5개년)  
# **분석 대상**: 전국 17개 시도  
# **데이터 소스**: 병무청, 질병관리청, 방위사업청  
# **AI 모델**: Random Forest, Neural Network 등 4종  
# **주요 발견**: 팬데믹 후 건강관리 시스템 개선으로 면제율 감소, R&D 투자 급증을 통한 국방혁신 가속화
# """)
    
    # 메인 콘텐츠 영역에서 선택된 메뉴에 따라 표시
    if st.session_state.current_menu == 'overview':
        show_overview()
    elif st.session_state.current_menu == 'pandemic_military':
        show_pandemic_military_analysis()
    elif st.session_state.current_menu == 'health_prediction':
        show_health_prediction()
    elif st.session_state.current_menu == 'city_analysis':
        show_city_analysis()
    elif st.session_state.current_menu == 'policy':
        show_policy_recommendations()
    elif st.session_state.current_menu == 'rnd':
        show_rnd_analysis()
    elif st.session_state.current_menu == 'localization':
        show_localization_analysis()
    elif st.session_state.current_menu == 'tech_trend':
        show_tech_trend_analysis()
    elif st.session_state.current_menu == 'import_export':
        show_import_export_analysis()
    elif st.session_state.current_menu == 'strategy_effectiveness':
        show_strategy_effectiveness_analysis()
    elif st.session_state.current_menu == 'health_defense_causality':
        show_health_defense_causality_analysis()
    elif st.session_state.current_menu == 'policy_simulation':
        show_policy_simulation()
    elif st.session_state.current_menu == 'future':
        show_future_strategy()
    else:
        show_overview()  # 기본값

# 개요 섹션
def show_overview():
    st.markdown('<a name="overview"></a>', unsafe_allow_html=True)
    # 대시보드 개요 박스
    st.markdown("""
    <div style="background: #fff; border-radius: 18px; border-left: 6px solid #ff6b00; padding: 24px 32px 18px 32px; margin-bottom: 24px; box-shadow: 0 4px 16px rgba(37,99,235,0.45);">
        <div style="font-size: 1.5em; font-weight: bold; color: #222; margin-bottom: 6px;">
            📊 대시보드 개요
        </div>
        <div style="font-size: 1.05em; color: #444;">
            팬데믹이 군인 건강과 국방력 혁신에 미치는 영향을 종합적으로 분석한 대시보드입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4개 카드(단색 배경, 진한 군청색 그림자 - 더 진하게)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background: #A78BFA; border-radius: 16px; padding: 18px; color: white; text-align: center; box-shadow: 0 4px 16px rgba(37,99,235,0.45); font-family: 'Noto Sans KR', sans-serif;">
            <div style="font-size: 1.1em; font-weight: bold;">군인 건강 영향</div>
            <div style="font-size: 2em; font-weight: bold;">17개 시·도</div>
            <div style="font-size: 0.95em;">전국 건강지표 분석</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #34D399; border-radius: 16px; padding: 18px; color: white; text-align: center; box-shadow: 0 4px 16px rgba(37,99,235,0.45); font-family: 'Noto Sans KR', sans-serif;">
            <div style="font-size: 1.1em; font-weight: bold;">국방 R&D</div>
            <div style="font-size: 2em; font-weight: bold;">+15.2%</div>
            <div style="font-size: 0.95em;">팬데믹 후 예산 증가율</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #FBBF24; border-radius: 16px; padding: 18px; color: white; text-align: center; box-shadow: 0 4px 16px rgba(37,99,235,0.45); font-family: 'Noto Sans KR', sans-serif;">
            <div style="font-size: 1.1em; font-weight: bold;">AI 예측</div>
            <div style="font-size: 2em; font-weight: bold;">3종</div>
            <div style="font-size: 0.95em;">건강등급·감염·면제율 예측</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background: #38BDF8; border-radius: 16px; padding: 18px; color: white; text-align: center; box-shadow: 0 4px 16px rgba(37,99,235,0.45); font-family: 'Noto Sans KR', sans-serif;">
            <div style="font-size: 1.1em; font-weight: bold;">고급 분석</div>
            <div style="font-size: 2em; font-weight: bold;">3개</div>
            <div style="font-size: 0.95em;">건강·감염·면제 분석 모듈</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 분석 개요
    st.markdown("""
    <div class="section fade-in">
        <h2>🎯 분석 목표</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 분석 목표를 컬럼으로 정리
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 건강 영향 분석")
        st.markdown("• 17개 도시별 감염병 발생 분석")
        st.markdown("• 병역자원 변화 추이 분석")
        st.markdown("• AI 기반 건강 위험도 예측")
        
        st.markdown("#### 🛡️ 국방력 혁신 분석")
        st.markdown("• R&D 투자 변화 분석")
        st.markdown("• 국산화율 현황 분석")
        st.markdown("• 신기술 투자 트렌드 분석")
    
    with col2:
        st.markdown("#### 🔬 고급 분석")
        st.markdown("• 전략 효과성 검증 및 ROI 분석")
        st.markdown("• 건강-방위전략 상관관계 측정")
        st.markdown("• 정책 시뮬레이션 및 최적 전략 도출")
        
        st.markdown("#### 📈 미래 전략")
        st.markdown("• 데이터 기반 정책 제안")
        st.markdown("• 단계별 로드맵 제시")
        st.markdown("• 우선순위 기반 실행 계획")

# 기존 섹션들 (변경 없음)
def show_rnd_analysis():
    st.markdown('<a name="rnd"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>📈 R&D 투자 변화 분석</h2>
        <p>팬데믹 전후 국방 R&D 투자 변화와 핵심기술 개발 현황을 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        data = load_rnd_data()
        if data is not None:
            analysis_results = analyze_rnd_trends(data)
            display_rnd_insights(analysis_results)
            
            st.markdown("### 📈 R&D 투자 트렌드")
            fig1 = plot_rnd_budget_trend(data)
            st.pyplot(fig1)
            
            st.markdown("### 🔬 연구 유형별 비교")
            fig2 = plot_research_type_comparison(data)
            st.pyplot(fig2)
        else:
            st.error("R&D 데이터를 불러올 수 없습니다.")
    except Exception as e:
        st.error(f"R&D 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")

def show_localization_analysis():
    st.markdown('<a name="localization"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🏭 국산화율 분석</h2>
        <p>국방 분야 국산화율 변화와 해외조달 의존도 변화를 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏭 국산화율 vs 해외조달 비중")
            labels = ['국산화', '해외조달']
            values = [78.5, 21.5]
            fig = plot_donut_chart(labels, values, '국산화율 vs 해외조달 비중')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 연도별 국산화율 변화")
            years = [2019, 2020, 2021, 2022, 2023]
            localization_rates = [75.2, 76.8, 77.5, 78.1, 78.5]
            
            fig = plt.figure()
            plt.plot(years, localization_rates, marker='o', linestyle='-', color='#F97316')
            plt.title('연도별 국산화율 변화')
            plt.xlabel('연도')
            plt.ylabel('국산화율 (%)')
            plt.grid(True)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"국산화율 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")

def show_tech_trend_analysis():
    st.markdown('<a name="tech_trend"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🤖 신기술 트렌드 분석</h2>
        <p>AI, 무인기, 사이버 등 신기술 분야 투자 트렌드를 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_tech_trend_dashboard()
    except Exception as e:
        st.error(f"신기술 트렌드 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")

def show_import_export_analysis():
    st.markdown('<a name="import_export"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🌍 수출입 분석</h2>
        <p>해외조달 vs 국내개발 비중 변화와 수출입 현황을 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_import_export_dashboard()
    except Exception as e:
        st.error(f"수출입 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")

def show_city_analysis():
    st.markdown('<a name="city_analysis"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🏙️ 도시별 분석</h2>
        <p>17개 도시별 군인 건강, 감염병 발생, 면제율 등을 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_city_dashboard()
    except Exception as e:
        st.error(f"도시별 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")

def show_health_prediction():
    st.markdown('<a name="health_prediction"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🩺 AI 기반 건강 위험도 예측</h2>
        <p>머신러닝을 활용한 군인 개별 건강 위험도 및 부대별 감염병 발생 확률을 예측합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_health_prediction_dashboard()
    except Exception as e:
        st.error(f"건강 위험도 예측 대시보드 오류: {e}")

def show_pandemic_military_analysis():
    st.markdown('<a name="pandemic_military"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🦠⚔️ 팬데믹 군인 영향 분석</h2>
        <p>팬데믹이 군인 건강과 병역 자원에 미친 영향을 5개 상관관계도로 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    create_pandemic_military_dashboard()

# 새로 추가된 고급 분석 섹션들
def show_strategy_effectiveness_analysis():
    st.markdown('<a name="strategy_effectiveness"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>📊 전략 효과 분석</h2>
        <p>과거 전략의 실제 성과를 데이터로 분석하고, 미래 전략 우선순위를 제시합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_strategy_effectiveness_dashboard()
    except Exception as e:
        st.error(f"전략 효과 분석 모듈 오류: {e}")

def show_health_defense_causality_analysis():
    st.markdown('<a name="health_defense_causality"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🔗 건강-방위전략 인과관계 분석</h2>
        <p>건강 위기가 방위전략에 미치는 실제 영향을 데이터로 분석합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_health_defense_causality_dashboard()
    except Exception as e:
        st.error(f"건강-방위전략 인과관계 분석 모듈 오류: {e}")

def show_policy_simulation():
    st.markdown('<a name="policy_simulation"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🎮 정책 시뮬레이션 엔진</h2>
        <p>다양한 정책 조합의 효과를 시뮬레이션하고 최적 전략을 도출합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        create_policy_simulation_dashboard()
    except Exception as e:
        st.error(f"정책 시뮬레이션 엔진 오류: {e}")

def show_policy_recommendations():
    st.markdown('<a name="policy"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>📋 실제 데이터 기반 정책 제안</h2>
        <p><strong>병무청·질병관리청·방위사업청 데이터 분석 결과를 바탕으로 한 구체적 정책 제안</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 핵심 발견사항 요약
    st.markdown("### 🎯 데이터 분석 핵심 발견사항")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: #A78BFA; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">면제율 개선</h4>
            <h2 style="margin: 5px 0;">-0.083%p</h2>
            <p style="margin: 0;">팬데믹 후 감소</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #34D399; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">R&D 투자 급증</h4>
            <h2 style="margin: 5px 0;">+24.3%</h2>
            <p style="margin: 0;">국방혁신 가속화</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #FBBF24; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">감염병-건강등급</h4>
            <h2 style="margin: 5px 0;">0.340</h2>
            <p style="margin: 0;">상관계수 (중간)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: #38BDF8; color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">국산화율</h4>
            <h2 style="margin: 5px 0;">78.5%</h2>
            <p style="margin: 0;">+3.3%p 증가</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 정책 제안 메인 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏥 건강 관리 정책")
        
        st.markdown("#### ✅ 1. AI 기반 건강 위험도 예측 시스템 도입")
        st.markdown("**🔍 데이터 근거**: 건강등급-감염병 상관계수 0.340, BMI-건강등급 연관성 확인")
        st.markdown("• 병역판정검사 데이터 기반 머신러닝 모델 개발")
        st.markdown("• 건강등급 하락 예측 조기경보 시스템 구축") 
        st.markdown("• 개인 맞춤형 건강 관리 프로그램 제공")
        
        st.markdown("#### ✅ 2. 감염병 대응 체계 강화")
        st.markdown("**🔍 데이터 근거**: 감염병발생률과 팬데믹영향도 강한 상관관계 (0.650)")
        st.markdown("• 부대별 신속 진단키트 상시 비축 (17개 시도 전체)")
        st.markdown("• 비대면 원격 진료 체계 구축")
        st.markdown("• 감염병 실시간 모니터링 시스템 도입")
        
        st.markdown("#### ✅ 3. 정기적 건강 스크리닝 강화")
        st.markdown("**🔍 데이터 근거**: 면제율 변화 -0.083%p, 건강관리 시스템 효과 입증")
        st.markdown("• 입영 전 지역보건소 연계 사전 스크리닝")
        st.markdown("• 체력·정신건강 회복 프로그램 운영")
        st.markdown("• 디지털 헬스 기반 모니터링 확대")
    
    with col2:
        st.markdown("### 🛡️ 국방력 혁신 정책")
        
        st.markdown("#### ✅ 1. 국방 R&D 투자 확대")
        st.markdown("**🔍 데이터 근거**: R&D 예산 24.3% 급증, 기술혁신 가속화 확인")
        st.markdown("• AI, 무인기, 사이버보안 분야 집중 투자")
        st.markdown("• 민관 협력 연구개발 프로그램 확대")
        st.markdown("• 국방 스타트업 육성 지원 강화")
        
        st.markdown("#### ✅ 2. 국산화율 제고")
        st.markdown("**🔍 데이터 근거**: 현재 국산화율 78.5%, 해외 의존도 감소 필요")
        st.markdown("• 핵심 부품 국산화 로드맵 수립")
        st.markdown("• 방산업체 기술 역량 강화 지원")
        st.markdown("• 전략적 공급망 다변화 추진")
        
        st.markdown("#### ✅ 3. 무인화·자동화 기술 투자")
        st.markdown("**🔍 데이터 근거**: 자동화 투자 28.7% 증가, 사이버보안 70.5% 급증")
        st.markdown("• 면제율 증가 대비 무인 시스템 도입")
        st.markdown("• 원격 운영 시스템 구축")
        st.markdown("• 사이버 방어 역량 강화")
    
    # 실제 수치 기반 핵심 지표
    st.markdown("---")
    st.markdown("### 📊 정책 근거 핵심 수치")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 R&D 예산 증가",
            "20,290억원",
            "+24.3% (3,956억원 증가)"
        )
    
    with col2:
        st.metric(
            "🏭 국산화율 향상",
            "78.5%",
            "+3.3%p (지속 개선)"
        )
    
    with col3:
        st.metric(
            "🔧 자동화 투자 급증",
            "434억원",
            "+28.7% (무인화 가속)"
        )
    
    with col4:
        st.metric(
            "🛡️ 사이버보안 강화",
            "530억원",
            "+70.5% (디지털 전환)"
        )
    
    # 지역별 맞춤 정책
    st.markdown("---")
    st.markdown("### 🗺️ 17개 시도별 맞춤 정책 (클러스터링 결과 기반)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("#### 🔴 고위험 지역 (서울·경기·인천)")
        st.error("**집중 관리 전략**")
        st.error("• 의료진 추가 배치 및 응급 의료체계 강화")
        st.error("• 실시간 감염병 모니터링 시스템 우선 구축")
        st.error("• 월별 건강검진 의무화 및 AI 위험도 예측")
        st.error("• 면제율 급증 방지 예방적 건강관리 프로그램")
    
    with col2:
        st.warning("#### 🟡 중위험 지역 (부산·대구·대전·광주)")
        st.warning("**예방 중심 전략**")
        st.warning("• 정기적 건강 모니터링 체계 구축")
        st.warning("• 예방접종 및 건강증진 프로그램 확대")
        st.warning("• 체력단련 시설 확충 및 운동 프로그램")
        st.warning("• 디지털 헬스케어 플랫폼 도입")
    
    with col3:
        st.success("#### 🟢 저위험 지역 (제주·강원·충남 등)")
        st.success("**모범 사례 확산 전략**")
        st.success("• 우수 사례 발굴 및 타 지역 벤치마킹")
        st.success("• 건강관리 노하우 공유 시스템 구축")
        st.success("• 연구개발 거점으로 활용")
        st.success("• 지속적 우수성 유지 인센티브 제공")
    
    # 정책 우선순위 및 예산
    st.markdown("---")
    st.markdown("### 🎯 정책 우선순위 및 예산 계획")
    
    st.markdown("#### 📈 단계별 투자 로드맵")
    
    roadmap_col1, roadmap_col2, roadmap_col3 = st.columns(3)
    
    with roadmap_col1:
        st.info("**🗓️ 1단계 (2025-2026년)**")
        st.info("• AI 예측 시스템 도입: 2,000억원")
        st.info("• 감염병 대응 체계 구축: 1,500억원") 
        st.info("• 기존 시설 개선: 1,500억원")
        st.info("• **총 예산: 5,000억원**")
    
    with roadmap_col2:
        st.warning("**🗓️ 2단계 (2026-2028년)**")
        st.warning("• 신기술 전면 도입: 3,000억원")
        st.warning("• 국산화율 80% 달성: 2,500억원")
        st.warning("• 민관 협력 확대: 2,500억원")
        st.warning("• **총 예산: 8,000억원**")
    
    with roadmap_col3:
        st.success("**🗓️ 3단계 (2028-2030년)**")
        st.success("• 완전 자동화 시스템: 5,000억원")
        st.success("• 글로벌 기술 선도: 4,000억원")
        st.success("• 차세대 국방 혁신: 3,000억원")
        st.success("• **총 예산: 12,000억원**")
    
    # 투자 수익률 (ROI) 분석
    st.markdown("#### 💰 투자 수익률 (ROI) 예측")
    
    roi_data = {
        "정책 분야": ["AI 예측 시스템", "감염병 대응", "R&D 투자 확대", "국산화 추진", "자동화 기술"],
        "투자 금액 (억원)": [2000, 1500, 8000, 3000, 2500],
        "예상 수익률 (%)": [320, 280, 250, 180, 220],
        "회수 기간 (년)": [2.5, 3.0, 4.0, 5.0, 3.5],
        "데이터 근거": [
            "면제율 감소 효과",
            "의료비 절감 효과", 
            "수출 증대 효과",
            "수입 대체 효과",
            "인력 절감 효과"
        ]
    }
    
    roi_df = pd.DataFrame(roi_data)
    st.dataframe(roi_df, use_container_width=True)
    
    # 최종 권고사항
    st.markdown("---")
    st.markdown("### 📝 최종 권고사항 (데이터 검증)")
    
    st.success("#### ✅ 즉시 실행 권장 (데이터 검증 완료)")
    st.success("• **AI 건강관리 시스템**: 감염병-건강등급 상관관계 0.340 입증")
    st.success("• **R&D 투자 확대**: 24.3% 증가 효과로 기술혁신 가속화 확인")
    st.success("• **감염병 대응체계**: 팬데믹영향도-감염병 강한 상관관계 0.650")
    st.success("• **자동화 투자**: 28.7% 급증으로 무인화 트렌드 가속화")
    
    st.warning("#### ⚠️ 중장기 과제 (지속 모니터링)")
    st.warning("• **국산화율 85% 달성**: 현재 78.5%에서 점진적 개선")
    st.warning("• **전문 인력 양성**: 신기술 분야 인력 부족 해결")
    st.warning("• **국제 협력 확대**: 선진국 기술 협력 및 표준화")
    st.warning("• **사이버보안 강화**: 70.5% 급증 트렌드 지속 필요")
    
    # 병무청 발표용 핵심 메시지
    st.markdown("---")
    st.markdown("### 🎤 분석 결과")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0;">
        <h3 style="color: white; margin: 0 0 15px 0;">📊 데이터가 증명하는 팬데믹 이후 국방혁신</h3>
        <div style="font-size: 1.1em; line-height: 1.6;">
            <strong>🔥 핵심 발견:</strong> 팬데믹이 오히려 군 건강관리를 개선시켰습니다<br>
            • 면제율 0.083%p 감소로 건강관리 시스템 효과 입증<br>
            • R&D 투자 24.3% 급증으로 국방혁신 가속화<br>
            • 17개 시도별 맞춤형 정책으로 효율성 극대화<br><br>
            <strong>🎯 정책 방향:</strong> 데이터 기반 예측형 국방정책으로 전환<br>
            • AI 건강예측 → 선제적 관리<br>
            • 무인화 기술 → 미래 전력 확보<br>
            • 국산화 추진 → 기술 자립 달성
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_future_strategy():
    st.markdown('<a name="future"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section fade-in">
        <h2>🚀 미래 전략</h2>
        <p>데이터 분석을 통해 도출된 미래 전략 방향성을 제시합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 전략 우선순위 - Streamlit 네이티브 방식
    st.markdown("""
    <div style="font-size:1.3em; font-weight:bold; color:#444; margin-bottom: 16px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="color:#e64980;">🎯 데이터 기반 전략 우선순위</span>
    </div>
    """, unsafe_allow_html=True)

    # 1순위: 선명한 그린
    st.markdown("""
    <div style="background:#16A34A; color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:14px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="font-size:1.15em; font-weight:bold;">🏆 1순위: 신기술 투자</span><br>
        효과성 점수: 100.0/100<br>
        <b>근거:</b> 신기술 투자 증가율: 379.9%<br>
        <b>권장사항:</b> AI, 사이버보안, 무인기 기술 특화 투자
    </div>
    """, unsafe_allow_html=True)

    # 2순위: 선명한 오렌지
    st.markdown("""
    <div style="background:#F59E42; color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:14px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="font-size:1.15em; font-weight:bold;">🥈 2순위: 수출 경쟁력</span><br>
        효과성 점수: 100.0/100<br>
        <b>근거:</b> 방산 수출 증가율: 141.0%<br>
        <b>권장사항:</b> 글로벌 시장 진출을 위한 품질 표준화
    </div>
    """, unsafe_allow_html=True)

    # 3순위: 선명한 블루
    st.markdown("""
    <div style="background:#2563EB; color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:14px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="font-size:1.15em; font-weight:bold;">🥉 3순위: R&D 투자 확대</span><br>
        효과성 점수: 75.5/100<br>
        <b>근거:</b> R&D 투자와 수출 성장 상관계수: 0.76<br>
        <b>권장사항:</b> AI/무인화 기술 중심 R&D 예산 30% 증액
    </div>
    """, unsafe_allow_html=True)

    # 4순위: 선명한 퍼플
    st.markdown("""
    <div style="background:#A21CAF; color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:14px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="font-size:1.15em; font-weight:bold;">🏅 4순위: 국산화율 향상</span><br>
        효과성 점수: 22.9/100<br>
        <b>근거:</b> 연간 국산화율 증가: 2.3%p<br>
        <b>권장사항:</b> 핵심 부품 국산화 로드맵 수립 및 집중 투자
    </div>
    """, unsafe_allow_html=True)
    
    # 로드맵
    st.markdown("""
    <div style="font-size:1.3em; font-weight:bold; color:#222; margin-bottom: 16px; font-family: 'Noto Sans KR', sans-serif;">
        <span style="color:#3B82F6;">🗺️ 단계별 로드맵</span>
    </div>
    <div style="display:flex; justify-content:center; gap:24px; margin-bottom:24px;">
        <div style="background:#E0F2FE; border:2px solid #38BDF8; border-radius:16px; padding:20px 32px; min-width:240px; text-align:center; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:1.15em; font-weight:bold; color:#2563EB; margin-bottom:8px;">🗓️ 1단계 (2024-2025)</div>
            <ul style="text-align:left; margin:0; padding-left:18px;">
                <li>AI 예측 시스템 도입</li>
                <li>감염병 대응 체계 구축</li>
                <li>기존 시설 개선</li>
                <li><b>예산: 5,000억원</b></li>
            </ul>
        </div>
        <div style="background:#F3E8FF; border:2px solid #A21CAF; border-radius:16px; padding:20px 32px; min-width:240px; text-align:center; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:1.15em; font-weight:bold; color:#A21CAF; margin-bottom:8px;">🗓️ 2단계 (2025-2027)</div>
            <ul style="text-align:left; margin:0; padding-left:18px;">
                <li>신기술 전면 도입</li>
                <li>국산화율 80% 달성</li>
                <li>민관 협력 확대</li>
                <li><b>예산: 8,000억원</b></li>
            </ul>
        </div>
        <div style="background:#DCFCE7; border:2px solid #16A34A; border-radius:16px; padding:20px 32px; min-width:240px; text-align:center; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:1.15em; font-weight:bold; color:#16A34A; margin-bottom:8px;">🗓️ 3단계 (2027-2030)</div>
            <ul style="text-align:left; margin:0; padding-left:18px;">
                <li>완전 자동화 시스템</li>
                <li>글로벌 기술 선도</li>
                <li>차세대 국방 혁신</li>
                <li><b>예산: 12,000억원</b></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 투자 수익률 (ROI) 분석
    st.markdown("### 💰 투자 수익률 (ROI) 분석")
    
    roi_data = {
        "전략 분야": ["AI 예측 시스템", "감염병 대응", "R&D 투자", "국산화", "신기술"],
        "투자 금액 (억원)": [2000, 1500, 8000, 3000, 2500],
        "예상 수익률 (%)": [320, 280, 250, 180, 220],
        "회수 기간 (년)": [2.5, 3.0, 4.0, 5.0, 3.5]
    }
    
    roi_df = pd.DataFrame(roi_data)
    st.dataframe(roi_df, use_container_width=True)
    
    # 위험도 평가
    st.markdown("### ⚠️ 위험도 평가")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 고위험 요소")
        st.markdown("• 기술 변화 속도")
        st.markdown("• 예산 확보 불확실성")
        st.markdown("• 인력 부족")
        st.markdown("• 국제 정세 변화")
    
    with col2:
        st.markdown("#### 🟢 대응 방안")
        st.markdown("• 단계적 도입 전략")
        st.markdown("• 다년도 예산 확보")
        st.markdown("• 전문 인력 양성")
        st.markdown("• 국제 협력 강화")
    
    # 성공 지표
    st.markdown("### 🎯 성공 지표 (KPI)")
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        st.markdown("#### 2025년 목표")
        st.markdown("• 국산화율: **80%**")
        st.markdown("• AI 시스템 도입률: **70%**")
        st.markdown("• 감염병 대응 시간: **24시간 이내**")
    
    with kpi_col2:
        st.markdown("#### 2027년 목표")
        st.markdown("• 방산 수출: **3,000억원**")
        st.markdown("• 무인화율: **50%**")
        st.markdown("• 사이버 보안 수준: **Level 5**")
    
    with kpi_col3:
        st.markdown("#### 2030년 목표")
        st.markdown("• 기술 자립도: **95%**")
        st.markdown("• 글로벌 경쟁력: **Top 3**")
        st.markdown("• 완전 디지털화: **100%**")
    
    # 최종 권고사항
    st.markdown("### 📝 최종 권고사항")
    
    st.markdown("""
    <div style="background:#f8f9fa; padding:20px; border-radius:10px; border-left:5px solid #28a745;">
        <h4 style="color:#28a745; margin-bottom:15px;">✅ 즉시 실행 항목</h4>
        <div>• <b>신기술 투자 확대:</b> AI, 무인기, 사이버보안 분야 집중 투자</div>
        <div>• <b>방산 수출 활성화:</b> 글로벌 표준 준수 및 품질 향상</div>
        <div>• <b>감염병 대응 체계:</b> 부대별 방역 시설 및 장비 확충</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#fff3cd; padding:20px; border-radius:10px; border-left:5px solid #ffc107; margin-top:15px;">
        <h4 style="color:#856404; margin-bottom:15px;">⚠️ 중장기 과제</h4>
        <div>• <b>국산화율 제고:</b> 단계적 접근을 통한 점진적 개선</div>
        <div>• <b>인력 양성:</b> 신기술 분야 전문 인력 확보 및 교육</div>
        <div>• <b>국제 협력:</b> 선진국과의 기술 협력 확대</div>
    </div>
    """, unsafe_allow_html=True)

# 앱 실행
if __name__ == "__main__":
    main()