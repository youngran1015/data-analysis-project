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

def load_rnd_data():
    """R&D 데이터 로드"""
    try:
        # UTF-8로 시도
        rnd_data = pd.read_csv('data/dapa/dapa_rnd_budget_and_tasks.csv', encoding='utf-8')
        rnd_data.columns = ['연도', '예산(억원)', '기초연구(개)', '응용연구(개)', '개발연구(개)', '총과제수(개)']
        rnd_data['연도'] = rnd_data['연도'].astype(int)
        return rnd_data
    except UnicodeDecodeError:
        try:
            # cp949로 시도
            rnd_data = pd.read_csv('data/dapa/dapa_rnd_budget_and_tasks.csv', encoding='cp949')
            rnd_data.columns = ['연도', '예산(억원)', '기초연구(개)', '응용연구(개)', '개발연구(개)', '총과제수(개)']
            rnd_data['연도'] = rnd_data['연도'].astype(int)
            return rnd_data
        except:
            st.error("R&D 데이터 인코딩 문제")
            return None
    except Exception as e:
        st.error(f"R&D 데이터 로딩 오류: {str(e)}")
        return None

def analyze_rnd_trends(data):
    """R&D 투자 트렌드 분석"""
    if data is None:
        return None
    
    # 팬데믹 전후 구분
    pre_pandemic = data[data['연도'] < 2020]
    post_pandemic = data[data['연도'] >= 2020]
    
    # 평균값 계산
    pre_avg_budget = pre_pandemic['예산(억원)'].mean()
    post_avg_budget = post_pandemic['예산(억원)'].mean()
    pre_avg_tasks = pre_pandemic['총과제수(개)'].mean()
    post_avg_tasks = post_pandemic['총과제수(개)'].mean()
    
    # 변화율 계산
    budget_change = ((post_avg_budget - pre_avg_budget) / pre_avg_budget) * 100
    tasks_change = ((post_avg_tasks - pre_avg_tasks) / pre_avg_tasks) * 100
    
    return {
        'pre_avg_budget': pre_avg_budget,
        'post_avg_budget': post_avg_budget,
        'pre_avg_tasks': pre_avg_tasks,
        'post_avg_tasks': post_avg_tasks,
        'budget_change': budget_change,
        'tasks_change': tasks_change
    }

def plot_rnd_budget_trend(data):
    """R&D 예산 트렌드 시각화"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 예산 트렌드
    ax1.plot(data['연도'], data['예산(억원)'], marker='o', linewidth=3, markersize=8, color='#3B82F6')
    ax1.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7, label='코로나19 시작')
    ax1.set_title('연도별 국방 R&D 예산 변화', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax1.set_ylabel('예산 (억원)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 과제 수 트렌드
    ax2.plot(data['연도'], data['총과제수(개)'], marker='s', linewidth=3, markersize=8, color='#10B981')
    ax2.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7, label='코로나19 시작')
    ax2.set_title('연도별 R&D 과제 수 변화', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('연도', fontsize=12, fontweight='bold')
    ax2.set_ylabel('과제 수 (개)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_research_type_comparison(data):
    """연구 유형별 비교"""
    # 2020년 전후 평균 계산
    pre_2020 = data[data['연도'] < 2020].mean()
    post_2020 = data[data['연도'] >= 2020].mean()
    
    research_types = ['기초연구(개)', '응용연구(개)', '개발연구(개)']
    pre_values = [pre_2020['기초연구(개)'], pre_2020['응용연구(개)'], pre_2020['개발연구(개)']]
    post_values = [post_2020['기초연구(개)'], post_2020['응용연구(개)'], post_2020['개발연구(개)']]
    
    x = np.arange(len(research_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, pre_values, width, label='팬데믹 이전 (2012-2019)', color='#3B82F6', alpha=0.8)
    bars2 = ax.bar(x + width/2, post_values, width, label='팬데믹 이후 (2020-2023)', color='#EF4444', alpha=0.8)
    
    ax.set_title('연구 유형별 과제 수 변화', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('연구 유형', fontsize=12, fontweight='bold')
    ax.set_ylabel('평균 과제 수 (개)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(research_types)
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

def display_rnd_insights(analysis_results):
    """R&D 분석 인사이트 표시"""
    if analysis_results is None:
        return
    
    st.markdown("### 🎯 R&D 투자 주요 발견사항")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "R&D 예산 변화",
            f"{analysis_results['budget_change']:+.1f}%",
            f"팬데믹 전후 평균"
        )
        
        if analysis_results['budget_change'] > 0:
            st.success("✅ 팬데믹 이후 R&D 예산이 증가했습니다!")
        else:
            st.warning("⚠️ 팬데믹 이후 R&D 예산이 감소했습니다.")
    
    with col2:
        st.metric(
            "R&D 과제 수 변화",
            f"{analysis_results['tasks_change']:+.1f}%",
            f"팬데믹 전후 평균"
        )
        
        if analysis_results['tasks_change'] > 0:
            st.success("✅ 팬데믹 이후 R&D 과제 수가 증가했습니다!")
        else:
            st.warning("⚠️ 팬데믹 이후 R&D 과제 수가 감소했습니다.")
    
    # 추가 분석
    st.markdown("---")
    st.markdown("#### 📊 상세 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "팬데믹 이전 평균 예산",
            f"{analysis_results['pre_avg_budget']:.1f}억원",
            "2012-2019"
        )
    
    with col2:
        st.metric(
            "팬데믹 이후 평균 예산",
            f"{analysis_results['post_avg_budget']:.1f}억원",
            "2020-2023"
        )
    
    with col3:
        st.metric(
            "예산 변화량",
            f"{analysis_results['post_avg_budget'] - analysis_results['pre_avg_budget']:+.1f}억원",
            "절대적 변화"
        )

def create_rnd_dashboard():
    """R&D 분석 대시보드"""
    st.header("💰 팬데믹 시대 국방 R&D 투자 변화 분석")
    st.markdown("**팬데믹 전후 국방 R&D 예산과 과제 수 변화를 분석합니다.**")
    
    # 데이터 로드
    data = load_rnd_data()
    if data is None:
        st.error("데이터를 불러올 수 없습니다.")
        return
    
    # 분석 실행
    analysis_results = analyze_rnd_trends(data)
    
    # 주요 인사이트 표시
    display_rnd_insights(analysis_results)
    
    # 시각화
    st.markdown("---")
    st.markdown("## 📈 R&D 투자 트렌드")
    
    # 예산 및 과제 수 트렌드
    fig1 = plot_rnd_budget_trend(data)
    st.pyplot(fig1)
    
    # 연구 유형별 비교
    st.markdown("## 🔬 연구 유형별 변화")
    fig2 = plot_research_type_comparison(data)
    st.pyplot(fig2)
    
    # 데이터 테이블
    st.markdown("## 📋 상세 데이터")
    st.dataframe(data, use_container_width=True)

if __name__ == "__main__":
    create_rnd_dashboard() 