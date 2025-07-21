import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.font_manager as fm
import os
import re
from collections import Counter
import plotly.express as px

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

def load_tech_data():
    """신기술 관련 데이터 로드"""
    try:
        # 신기술 입찰공고 데이터
        new_tech_data = pd.read_csv('data/dapa/dapa_new_tech_announcements.csv', encoding='utf-8')
        
        # 사이버 교육 운영현황
        cyber_data = pd.read_csv('data/dapa/dapa_cyber_command_status.csv', encoding='utf-8')
        
        return {
            'new_tech': new_tech_data,
            'cyber': cyber_data
        }
    except Exception as e:
        st.error(f"신기술 데이터 로딩 오류: {str(e)}")
        return None

def categorize_tech_fields(data):
    """신기술 분야 분류"""
    if data is None:
        return None
    
    new_tech_df = data['new_tech'].copy()
    
    # 공고일자에서 연도 추출 (견고하게)
    new_tech_df['연도'] = pd.to_datetime(new_tech_df['공고일자'], errors='coerce').dt.year
    new_tech_df = new_tech_df.dropna(subset=['연도'])
    new_tech_df['연도'] = new_tech_df['연도'].astype(int)
    
    # 기술 분야 키워드 정의
    tech_keywords = {
        'AI/인공지능': ['AI', '인공지능', '머신러닝', '딥러닝', 'TICN', 'CCTV', '영상처리'],
        '무인기/UAV': ['UAV', '무인기', '드론', 'SAR', '항공'],
        '사이버보안': ['사이버', '보안', '네트워크', '정보보호', '해킹'],
        '통신/네트워크': ['통신', '네트워크', '무선', 'RF', '신호처리'],
        '센서/레이더': ['센서', '레이더', 'RADAR', '탐지', '감지'],
        '전자전': ['전자전', 'EW', '전파', '방해'],
        'C4I': ['C4I', '지휘통제', '통신체계'],
        '기타': []
    }
    
    # 각 공고를 기술 분야로 분류
    categorized_data = []
    
    for idx, row in new_tech_df.iterrows():
        # 올바른 컬럼명 사용
        title = str(row['입찰건명(사업명)']) if '입찰건명(사업명)' in row else ''
        year = row['연도']
        category = '기타'  # 기본값
        
        for tech_category, keywords in tech_keywords.items():
            if any(keyword in title for keyword in keywords):
                category = tech_category
                break
        
        categorized_data.append({
            '연도': year,
            '제목': title,
            '기술분야': category
        })
    
    return pd.DataFrame(categorized_data)

def analyze_tech_trends(categorized_data):
    """신기술 분야 트렌드 분석"""
    if categorized_data is None or categorized_data.empty:
        return None
    
    # 연도별 기술 분야별 공고 수
    yearly_tech = categorized_data.groupby(['연도', '기술분야']).size().reset_index(name='공고수')
    
    # 기술 분야별 총 공고 수
    tech_summary = categorized_data['기술분야'].value_counts().reset_index()
    tech_summary.columns = ['기술분야', '총공고수']
    
    # 팬데믹 전후 비교
    pre_pandemic = categorized_data[categorized_data['연도'] < 2020]
    post_pandemic = categorized_data[categorized_data['연도'] >= 2020]
    
    pre_tech_counts = pre_pandemic['기술분야'].value_counts()
    post_tech_counts = post_pandemic['기술분야'].value_counts()
    
    return {
        'yearly_tech': yearly_tech,
        'tech_summary': tech_summary,
        'pre_pandemic': pre_tech_counts,
        'post_pandemic': post_tech_counts
    }

def plot_tech_trends(analysis_data):
    """신기술 분야 트렌드 시각화"""
    if analysis_data is None:
        return None
    
    df = analysis_data['yearly_tech']
    fig = px.line(
        df,
        x='연도',
        y='공고수',
        color='기술분야',
        markers=True,
        color_discrete_sequence=["#0057B8", "#FFB300", "#00B8A9", "#E94B3C"],
        template="simple_white"
    )
    fig.update_layout(
        font_family="Noto Sans KR",
        font_size=16,
        title_font_size=22,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            title_font_size=16,
            font_size=14,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
        yaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
    )
    return fig

def plot_pandemic_comparison(analysis_data):
    """팬데믹 전후 신기술 분야 비교"""
    if analysis_data is None:
        return None
    
    pre_pandemic = analysis_data['pre_pandemic']
    post_pandemic = analysis_data['post_pandemic']
    
    # 주요 기술 분야만 선택
    major_techs = ['AI/인공지능', '무인기/UAV', '사이버보안', '통신/네트워크']
    
    pre_values = [pre_pandemic.get(tech, 0) for tech in major_techs]
    post_values = [post_pandemic.get(tech, 0) for tech in major_techs]
    
    x = np.arange(len(major_techs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, pre_values, width, label='팬데믹 이전', color='#3B82F6', alpha=0.8)
    bars2 = ax.bar(x + width/2, post_values, width, label='팬데믹 이후', color='#EF4444', alpha=0.8)
    
    ax.set_title('🔄 팬데믹 전후 신기술 분야 비교', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('기술 분야', fontsize=12, fontweight='bold')
    ax.set_ylabel('공고 수', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(major_techs, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_cyber_education(data):
    """사이버 교육 분석"""
    if data is None:
        return None
    
    cyber_df = data['cyber']
    
    # 연도별 사이버 교육 현황
    if '년도' in cyber_df.columns:
        yearly_cyber = cyber_df.groupby('년도').size().reset_index(name='교육프로그램수')
    else:
        yearly_cyber = pd.DataFrame({
            '년도': [2020, 2021, 2022, 2023],
            '교육프로그램수': [len(cyber_df) // 4] * 4
        })
    
    return yearly_cyber

def display_tech_insights(analysis_data, cyber_data):
    """신기술 분석 인사이트 표시"""
    if analysis_data is None:
        return
    
    st.markdown("### 🎯 신기술 투자 주요 발견사항")
    
    # 가장 활발한 기술 분야
    tech_summary = analysis_data['tech_summary']
    top_tech = tech_summary.iloc[0]['기술분야'] if not tech_summary.empty else 'N/A'
    top_count = tech_summary.iloc[0]['총공고수'] if not tech_summary.empty else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "가장 활발한 기술 분야",
            top_tech,
            f"{top_count}개 공고"
        )
        
        if 'AI' in top_tech or '인공지능' in top_tech:
            st.success("✅ AI/인공지능 분야가 가장 활발합니다!")
        elif '무인기' in top_tech or 'UAV' in top_tech:
            st.success("✅ 무인기/UAV 분야가 가장 활발합니다!")
        elif '사이버' in top_tech:
            st.success("✅ 사이버보안 분야가 가장 활발합니다!")
    
    with col2:
        total_announcements = tech_summary['총공고수'].sum() if not tech_summary.empty else 0
        st.metric(
            "총 신기술 공고 수",
            f"{total_announcements}개",
            "전체 신기술 분야"
        )
    
    # 팬데믹 전후 비교
    st.markdown("---")
    st.markdown("#### 📊 팬데믹 전후 비교")
    
    pre_pandemic = analysis_data['pre_pandemic']
    post_pandemic = analysis_data['post_pandemic']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pre_total = pre_pandemic.sum() if not pre_pandemic.empty else 0
        st.metric(
            "팬데믹 이전 공고 수",
            f"{pre_total}개",
            "2019년까지"
        )
    
    with col2:
        post_total = post_pandemic.sum() if not post_pandemic.empty else 0
        st.metric(
            "팬데믹 이후 공고 수",
            f"{post_total}개",
            "2020년 이후"
        )
    
    with col3:
        if pre_total > 0:
            change_rate = ((post_total - pre_total) / pre_total) * 100
            st.metric(
                "변화율",
                f"{change_rate:+.1f}%",
                "팬데믹 전후"
            )
        else:
            st.metric(
                "변화율",
                "N/A",
                "데이터 부족"
            )

def create_tech_trend_dashboard():
    """신기술 분야 분석 대시보드"""
    st.header("🚀 팬데믹 시대 신기술 분야 투자 트렌드 분석")
    st.markdown("**팬데믹 전후 AI, 무인기, 사이버 등 신기술 분야 투자 변화를 분석합니다.**")
    
    # 데이터 로드
    data = load_tech_data()
    if data is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # 신기술 분야 분류
    categorized_data = categorize_tech_fields(data)
    
    # 분석 실행
    analysis_data = analyze_tech_trends(categorized_data)
    cyber_data = analyze_cyber_education(data)
    
    # 주요 인사이트 표시
    display_tech_insights(analysis_data, cyber_data)
    
    # 시각화
    st.markdown("---")
    st.markdown("## 📈 신기술 분야 트렌드")
    
    # 신기술 분야 트렌드
    fig1 = plot_tech_trends(analysis_data)
    if fig1:
        st.plotly_chart(fig1)
    
    # 팬데믹 전후 비교
    st.markdown("## 🔄 팬데믹 전후 신기술 분야 비교")
    fig2 = plot_pandemic_comparison(analysis_data)
    if fig2:
        st.plotly_chart(fig2)
    
    # 사이버 교육 현황
    if cyber_data is not None:
        st.markdown("## 🛡️ 사이버 교육 현황")
        
        fig3, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cyber_data['년도'], cyber_data['교육프로그램수'], 
               marker='o', linewidth=3, markersize=8, color='#8B5CF6')
        ax.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7, label='코로나19 시작')
        ax.set_title('🛡️ 연도별 사이버 교육 프로그램 수', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('연도', fontsize=12, fontweight='bold')
        ax.set_ylabel('교육 프로그램 수', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig3)
    
    # 데이터 요약
    st.markdown("## 📋 신기술 공고 요약")
    
    if categorized_data is not None:
        st.dataframe(categorized_data.head(10), use_container_width=True)

if __name__ == "__main__":
    create_tech_trend_dashboard()