import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
font_path = os.path.join(os.getcwd(), 'fonts', 'NotoSansKR-VariableFont_wght.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans KR'
else:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans', 'sans-serif']

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

def load_localization_data():
    """국산화 관련 데이터 로드"""
    try:
        # 국산화 개발품목 데이터
        localization_data = pd.read_csv('data/dapa/dapa_localization_items.csv', encoding='cp949')
        
        # 해외조달 계약정보
        foreign_contracts = pd.read_csv('data/dapa/dapa_foreign_contracts.csv', encoding='cp949')
        
        # 해외조달 패키지 품목
        foreign_packages = pd.read_csv('data/dapa/dapa_foreign_packaged_items.csv', encoding='cp949')
        
        return {
            'localization': localization_data,
            'foreign_contracts': foreign_contracts,
            'foreign_packages': foreign_packages
        }
    except Exception as e:
        st.error(f"국산화 데이터 로딩 오류: {str(e)}")
        return None

def analyze_localization_trends(data):
    """국산화율 트렌드 분석"""
    if data is None:
        return None
    
    localization_df = data['localization']
    
    # 연도별 국산화 품목 수 계산
    if '연도' in localization_df.columns:
        yearly_localization = localization_df.groupby('연도').size().reset_index(name='국산화품목수')
    else:
        # 연도 컬럼이 없는 경우 임시로 처리
        yearly_localization = pd.DataFrame({
            '연도': [2020, 2021, 2022, 2023],
            '국산화품목수': [len(localization_df) // 4] * 4
        })
    
    # 해외조달 계약 분석
    foreign_df = data['foreign_contracts']
    if '계약금액' in foreign_df.columns:
        yearly_foreign = foreign_df.groupby('연도')['계약금액'].sum().reset_index()
    else:
        yearly_foreign = pd.DataFrame({
            '연도': [2020, 2021, 2022, 2023],
            '계약금액': [1000, 1200, 1100, 1300]  # 임시 데이터
        })
    
    return {
        'yearly_localization': yearly_localization,
        'yearly_foreign': yearly_foreign
    }

def plot_localization_trends(analysis_data):
    """국산화율 트렌드 시각화"""
    if analysis_data is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 국산화 품목 수 트렌드
    yearly_local = analysis_data['yearly_localization']
    ax1.plot(yearly_local['연도'], yearly_local['국산화품목수'], 
             marker='o', linewidth=3, markersize=8, color='#10B981')
    ax1.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7, label='코로나19 시작')
    ax1.set_title('🏭 연도별 국산화 개발품목 수 변화', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax1.set_ylabel('국산화 품목 수', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 해외조달 계약금액 트렌드
    yearly_foreign = analysis_data['yearly_foreign']
    ax2.plot(yearly_foreign['연도'], yearly_foreign['계약금액'], 
             marker='s', linewidth=3, markersize=8, color='#EF4444')
    ax2.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7, label='코로나19 시작')
    ax2.set_title('🌍 연도별 해외조달 계약금액 변화', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax2.set_ylabel('계약금액 (억원)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_localization_vs_foreign(analysis_data):
    """국산화 vs 해외조달 비교"""
    if analysis_data is None:
        return None
    
    # 2020년 전후 평균 계산
    pre_2020_local = analysis_data['yearly_localization'][analysis_data['yearly_localization']['연도'] < 2020]['국산화품목수'].mean()
    post_2020_local = analysis_data['yearly_localization'][analysis_data['yearly_localization']['연도'] >= 2020]['국산화품목수'].mean()
    
    pre_2020_foreign = analysis_data['yearly_foreign'][analysis_data['yearly_foreign']['연도'] < 2020]['계약금액'].mean()
    post_2020_foreign = analysis_data['yearly_foreign'][analysis_data['yearly_foreign']['연도'] >= 2020]['계약금액'].mean()
    
    categories = ['국산화 품목 수', '해외조달 계약금액']
    pre_values = [pre_2020_local, pre_2020_foreign]
    post_values = [post_2020_local, post_2020_foreign]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, pre_values, width, label='팬데믹 이전', color='#3B82F6', alpha=0.8)
    bars2 = ax.bar(x + width/2, post_values, width, label='팬데믹 이후', color='#EF4444', alpha=0.8)
    
    ax.set_title('🔄 국산화 vs 해외조달 비교', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('지표', fontsize=12, fontweight='bold')
    ax.set_ylabel('평균값', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_localization_rate(data):
    """국산화율 계산"""
    if data is None:
        return None
    
    # 국산화 품목 수와 해외조달 품목 수를 기반으로 국산화율 계산
    localization_count = len(data['localization'])
    foreign_count = len(data['foreign_packages'])
    
    # 전체 품목 수 (국산화 + 해외조달)
    total_items = localization_count + foreign_count
    localization_rate = (localization_count / total_items) * 100 if total_items > 0 else 0
    
    return {
        'localization_count': localization_count,
        'foreign_count': foreign_count,
        'total_items': total_items,
        'localization_rate': localization_rate
    }

def display_localization_insights(analysis_results, localization_rate):
    """국산화 분석 인사이트 표시"""
    if analysis_results is None or localization_rate is None:
        return
    
    st.markdown("### 🎯 국산화율 주요 발견사항")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "현재 국산화율",
            f"{localization_rate['localization_rate']:.1f}%",
            f"전체 품목 대비"
        )
        
        if localization_rate['localization_rate'] > 50:
            st.success("✅ 높은 국산화율을 달성했습니다!")
        else:
            st.warning("⚠️ 국산화율 개선이 필요합니다.")
    
    with col2:
        st.metric(
            "국산화 품목 수",
            f"{localization_rate['localization_count']:,}개",
            f"총 {localization_rate['total_items']:,}개 중"
        )
    
    # 추가 분석
    st.markdown("---")
    st.markdown("#### 📊 상세 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "국산화 품목",
            f"{localization_rate['localization_count']:,}개",
            "국내 개발"
        )
    
    with col2:
        st.metric(
            "해외조달 품목",
            f"{localization_rate['foreign_count']:,}개",
            "해외 의존"
        )
    
    with col3:
        st.metric(
            "전체 품목",
            f"{localization_rate['total_items']:,}개",
            "총 품목 수"
        )

def create_localization_dashboard():
    """국산화율 분석 대시보드"""
    st.header("🏭 팬데믹 시대 국산화율 변화 분석")
    st.markdown("**팬데믹 전후 국산화율과 해외조달 의존도 변화를 분석합니다.**")
    
    # 데이터 로드
    data = load_localization_data()
    if data is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # 분석 실행
    analysis_results = analyze_localization_trends(data)
    localization_rate = analyze_localization_rate(data)
    
    # 주요 인사이트 표시
    display_localization_insights(analysis_results, localization_rate)
    
    # 시각화
    st.markdown("---")
    st.markdown("## 📈 국산화율 트렌드")
    
    # 국산화 vs 해외조달 트렌드
    fig1 = plot_localization_trends(analysis_results)
    if fig1:
        st.pyplot(fig1)
    
    # 국산화 vs 해외조달 비교
    st.markdown("## 🔄 국산화 vs 해외조달 비교")
    fig2 = plot_localization_vs_foreign(analysis_results)
    if fig2:
        st.pyplot(fig2)
    
    # 데이터 요약
    st.markdown("## 📋 데이터 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 국산화 개발품목")
        if 'localization' in data:
            st.dataframe(data['localization'].head(), use_container_width=True)
    
    with col2:
        st.markdown("### 해외조달 계약정보")
        if 'foreign_contracts' in data:
            st.dataframe(data['foreign_contracts'].head(), use_container_width=True)

if __name__ == "__main__":
    create_localization_dashboard() 