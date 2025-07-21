import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

CITY_LIST = [
    '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
    '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
]

# 도시명 매핑 (전체명 -> 2글자 축약)
CITY_MAPPING = {
    '서울특별시': '서울',
    '부산광역시': '부산', 
    '대구광역시': '대구',
    '인천광역시': '인천',
    '광주광역시': '광주',
    '대전광역시': '대전',
    '울산광역시': '울산',
    '세종특별자치시': '세종',
    '경기도': '경기',
    '강원특별자치도': '강원',
    '충청북도': '충북',
    '충청남도': '충남', 
    '전라북도': '전북',
    '전라남도': '전남',
    '경상북도': '경북',
    '경상남도': '경남',
    '제주특별자치도': '제주'
}

# 1. 데이터 로드 함수
def load_city_data(filepath):
    """17개 도시별 주요 지표 데이터 로드 (csv) - wide format을 long format으로 변환"""
    # 파일 존재 여부 확인
    if not os.path.exists(filepath):
        st.error("필요한 데이터 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    
    try:
        # UTF-8로 파일 로드 (CSV 정보에서 확인됨)
        df = pd.read_csv(filepath, encoding='utf-8')
        # 파일 로드 성공 메시지는 대시보드에서 한 번만 표시
        
        # wide format을 long format으로 변환
        # 연도 컬럼을 제외한 나머지 도시 컬럼들을 melt
        city_columns = [col for col in df.columns if col != '연도']
        df_long = df.melt(id_vars=['연도'], 
                         value_vars=city_columns,
                         var_name='도시명', 
                         value_name='값')
        
        # 도시명을 2글자로 축약
        df_long['도시'] = df_long['도시명'].map(CITY_MAPPING.get)
        
        # 매핑되지 않은 도시는 제거
        df_long = df_long.dropna(subset=['도시'])
        
        # 도시 순서 정렬
        df_long['도시'] = pd.Categorical(df_long['도시'], categories=CITY_LIST, ordered=True)
        df_long = df_long.sort_values(by=['연도', '도시']).reset_index(drop=True)
        
        # 최종 컬럼 정리 (기존 코드와 호환성을 위해 '건강등급' 컬럼명 사용)
        df_final = df_long[['연도', '도시', '값']].copy()
        df_final.columns = ['연도', '도시', '건강등급']
        
        return df_final
        
    except Exception as e:
        st.error("데이터 로드 중 오류가 발생했습니다.")
        return pd.DataFrame()

# 2. 세로 막대그래프
def plot_city_bar_chart(df, value_col, title, color='#F97316'):
    """도시별 데이터를 막대그래프로 시각화"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 데이터 정렬
    sorted_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    # 막대그래프 생성
    bars = ax.bar(sorted_df['도시'], sorted_df[value_col], color=color, alpha=0.85)
    
    # 제목 및 레이블 설정
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('도시', fontsize=14, fontweight='bold')
    ax.set_ylabel(value_col, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):  # NaN 값 체크
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# 3. 산점도/상관도
def plot_city_scatter(df, x_col, y_col, title, color='#0F172A'):
    """두 변수 간의 상관관계를 산점도로 시각화"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 유효한 데이터만 필터링
    valid_data = df.dropna(subset=[x_col, y_col])
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, '유효한 데이터가 없습니다', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig, 0
    
    # 산점도 생성
    ax.scatter(valid_data[x_col], valid_data[y_col], 
               s=120, c=color, alpha=0.7, edgecolors='black')
    
    # 도시명 표시 (기존 annotate 방식)
    for i, row in valid_data.iterrows():
        ax.annotate(row['도시'], (row[x_col], row[y_col]), 
                    fontsize=11, ha='center', va='bottom')
    
    # 상관계수 계산 (NaN/inf 방지)
    mask = np.isfinite(valid_data[x_col]) & np.isfinite(valid_data[y_col])
    if mask.sum() > 1:
        corr = valid_data[x_col][mask].corr(valid_data[y_col][mask])
        if pd.isna(corr):
            corr = 0
    else:
        corr = 0
    
    ax.set_title(f'{title}\n(상관계수: {corr:.2f})', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(x_col, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, corr

# 4. 표준화 + 클러스터링
def cluster_cities(df, feature_cols, n_clusters=3):
    """도시들을 특성 기반으로 클러스터링"""
    # 유효한 데이터만 필터링
    valid_data = df.dropna(subset=feature_cols)
    
    if len(valid_data) < n_clusters:
        st.warning(f"클러스터링을 위한 유효한 데이터가 부족합니다. (필요: {n_clusters}개, 현재: {len(valid_data)}개)")
        return df, None
    
    # 표준화
    scaler = StandardScaler()
    X = scaler.fit_transform(valid_data[feature_cols])
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    
    # 결과 저장
    df_clustered = df.copy()
    df_clustered['클러스터'] = -1  # 기본값
    df_clustered.loc[valid_data.index, '클러스터'] = labels
    
    return df_clustered, kmeans

# Pastel color palette
pastel_colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', 
                '#ADD8E6', '#B0E0E6', '#E6E6FA', '#FFA07A', '#20B2AA', 
                '#87CEFA', '#FAEBD7', '#F0FFF0', '#E0FFFF', '#F5F5DC', 
                '#FFE4E1', '#D8BFD8']

def plot_bar_chart_city_health(data, title, y_col='건강등급', color_idx=0):
    """건강등급 또는 기타 지표를 막대그래프로 시각화"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 유효한 데이터만 필터링
    valid_data = data.dropna(subset=[y_col])
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, '표시할 데이터가 없습니다', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # 막대그래프 생성
    bars = ax.bar(valid_data['도시'], valid_data[y_col], 
                  color=pastel_colors[color_idx % len(pastel_colors)], 
                  edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('도시', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_sample_health_data():
    """샘플 건강등급 데이터 생성"""
    np.random.seed(42)
    data = []
    for year in [2022, 2023, 2024]:
        for city in CITY_LIST:
            health_grade = np.random.uniform(3.0, 4.5)  # 3.0-4.5 범위의 건강등급
            data.append({'연도': year, '도시': city, '건강등급': health_grade})
    
    return pd.DataFrame(data)

def create_sample_infect_data():
    """샘플 감염병 데이터 생성"""
    np.random.seed(43)
    data = []
    for year in [2022, 2023, 2024]:
        for city in CITY_LIST:
            infect_rate = np.random.uniform(0.5, 3.0)  # 0.5-3.0 범위의 감염병 발생률
            data.append({'연도': year, '도시': city, '건강등급': infect_rate})
    
    return pd.DataFrame(data)

def analyze_data(df_health, df_infect, data_type="실제"):
    """데이터 분석 실행 - 안전한 DataFrame 처리"""
    st.markdown(f"### 📊 {data_type} 데이터 분석 결과")
    
    # 데이터 유효성 검사
    health_valid = isinstance(df_health, pd.DataFrame) and not df_health.empty
    infect_valid = isinstance(df_infect, pd.DataFrame) and not df_infect.empty
    
    if not health_valid and not infect_valid:
        st.warning("분석할 유효한 데이터가 없습니다.")
        return

    # 건강등급 데이터 분석
    if health_valid:
        try:
            st.markdown('#### 도시별 건강등급 분석')
            
            # 최신 연도 데이터 추출
            latest_year = df_health['연도'].max()
            df_health_latest = df_health[df_health['연도'] == latest_year].copy()
            
            if len(df_health_latest) > 0:
                # 건강등급 막대그래프
                fig1 = plot_bar_chart_city_health(df_health_latest, 
                                                 f'{latest_year}년 도시별 건강등급', 
                                                 '건강등급', 0)
                st.pyplot(fig1)
                # 자동 해설 추가
                health_series = pd.Series(df_health_latest['건강등급'])
                max_idx = health_series.idxmax()
                min_idx = health_series.idxmin()
                max_city = df_health_latest.loc[max_idx, '도시']
                min_city = df_health_latest.loc[min_idx, '도시']
                st.info(f"{latest_year}년 기준, 건강등급이 가장 높은 도시는 **{max_city}**, 가장 낮은 도시는 **{min_city}**입니다.")
                
                # 기본 통계 (안전한 계산)
                st.markdown('##### 건강등급 기본 통계')
                valid_values = pd.Series(df_health_latest['건강등급']).dropna()
                valid_values = valid_values[np.isfinite(valid_values)]
                if len(valid_values) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric('평균', f'{valid_values.mean():.2f}')
                    with col2:
                        st.metric('최대값', f'{valid_values.max():.2f}')
                    with col3:
                        st.metric('최소값', f'{valid_values.min():.2f}')
                    with col4:
                        st.metric('표준편차', f'{valid_values.std():.2f}')
                else:
                    st.warning("유효한 건강등급 데이터가 없습니다.")
            else:
                st.warning("최신 연도 건강등급 데이터가 없습니다.")
        except Exception as e:
            st.error(f"건강등급 데이터 분석 중 오류: {e}")
    
    # 감염병 데이터 분석
    if infect_valid:
        try:
            st.markdown('#### 도시별 감염병 발생 분석')
            
            # 최신 연도 데이터 추출
            latest_year_infect = df_infect['연도'].max()
            df_infect_latest = df_infect[df_infect['연도'] == latest_year_infect].copy()
            
            if len(df_infect_latest) > 0:
                # 감염병 발생 막대그래프
                fig2 = plot_bar_chart_city_health(df_infect_latest, 
                                                 f'{latest_year_infect}년 도시별 감염병 발생', 
                                                 '건강등급', 1)  # 컬럼명이 '건강등급'으로 통일됨
                st.pyplot(fig2)
                # 자동 해설 추가
                infect_series = pd.Series(df_infect_latest['건강등급'])
                max_idx = infect_series.idxmax()
                min_idx = infect_series.idxmin()
                max_city = df_infect_latest.loc[max_idx, '도시']
                min_city = df_infect_latest.loc[min_idx, '도시']
                st.info(f"{latest_year_infect}년 기준, 감염병 발생률이 가장 높은 도시는 **{max_city}**, 가장 낮은 도시는 **{min_city}**입니다.")
            else:
                st.warning("최신 연도 감염병 데이터가 없습니다.")
        except Exception as e:
            st.error(f"감염병 데이터 분석 중 오류: {e}")
    
    # 상관관계 분석 (두 데이터 모두 있을 때만)
    if health_valid and infect_valid:
        try:
            st.markdown('#### 건강등급 vs 감염병 발생 상관관계')
            
            # 데이터 병합 (안전한 방식)
            df_merged = pd.merge(df_health, df_infect, 
                               on=['연도', '도시'], 
                               suffixes=('_건강', '_감염'),
                               how='inner')  # inner join으로 안전하게
            
            if len(df_merged) > 0:
                latest_year_merged = df_merged['연도'].max()
                df_latest = df_merged[df_merged['연도'] == latest_year_merged].copy()
                
                if len(df_latest) >= 2:  # 상관분석을 위해 최소 2개 데이터 필요
                    # 상관관계 산점도
                    fig3, corr = plot_city_scatter(df_latest, 
                                                 '건강등급_건강', '건강등급_감염',
                                                 '건강등급 vs 감염병 발생률')
                    st.pyplot(fig3)
                    
                    if corr != 0:
                        st.info(f'상관계수: {corr:.3f}')
                        if abs(corr) > 0.7:
                            st.success('강한 상관관계가 있습니다!')
                        elif abs(corr) > 0.3:
                            st.warning('보통 정도의 상관관계가 있습니다.')
                        else:
                            st.info('약한 상관관계입니다.')
                    
                    # 클러스터링 분석 (안전한 처리)
                    st.markdown('##### 도시별 클러스터링 분석')
                    
                    df_clustered, kmeans = cluster_cities(df_latest, 
                                                        ['건강등급_건강', '건강등급_감염'], 
                                                        n_clusters=3)
                    
                    if kmeans is not None and isinstance(df_clustered, pd.DataFrame):
                        # 클러스터링 결과 표시 (안전한 컬럼 접근)
                        display_columns = ['도시', '건강등급_건강', '건강등급_감염', '클러스터']
                        available_columns = [col for col in display_columns if col in df_clustered.columns]
                        
                        if len(available_columns) >= 3:
                            cluster_data = df_clustered[available_columns].copy()
                            # 컬럼명 정리
                            rename_dict = {
                                '건강등급_건강': '건강등급',
                                '건강등급_감염': '감염병발생'
                            }
                            cluster_data = cluster_data.rename(columns=rename_dict)  # type: ignore
                            
                            st.dataframe(cluster_data, use_container_width=True)
                            
                            # 클러스터별 도시 분포 (안전한 처리)
                            st.markdown('###### 클러스터별 도시 분포')
                            if '클러스터' in cluster_data.columns:
                                unique_clusters = cluster_data['클러스터'].dropna().unique()
                                for c in sorted(unique_clusters):
                                    if c >= 0:  # 유효한 클러스터만
                                        cities_in_cluster = cluster_data[cluster_data['클러스터'] == c]['도시'].tolist()
                                        if cities_in_cluster:
                                            st.markdown(f'- **클러스터 {int(c)+1}**: {", ".join(cities_in_cluster)}')
                        else:
                            st.warning("클러스터링 결과 표시를 위한 충분한 컬럼이 없습니다.")
                    else:
                        st.warning("클러스터링을 수행할 수 없습니다.")
                else:
                    st.warning("상관분석을 위한 충분한 데이터가 없습니다.")
            else:
                st.warning("건강등급과 감염병 데이터를 병합할 수 없습니다.")
        except Exception as e:
            st.error(f"상관관계 분석 중 오류: {e}")
            st.write(f"오류 상세: {str(e)}")

# 5. Streamlit 대시보드 함수
def create_city_dashboard():
    """메인 대시보드 생성"""
    st.header('🏙️ 17개 도시별 군인 건강/감염병/면제율 분석')
    st.markdown('**도시별 주요 지표를 막대그래프, 상관도, 클러스터링으로 분석합니다.**')
    
    # 현재 작업 디렉토리 확인
    # st.info(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # 데이터 폴더 구조 확인
    data_dir = 'data'
    if not os.path.exists(data_dir):
        st.error("data 폴더가 존재하지 않습니다!")
        return
    
    # 데이터 파일 경로 (실제 파일명에 맞춤)
    filepath_health = 'data/mma/mma_health_grade.csv'
    filepath_infect = 'data/kdca/kdca_infections.csv'  # 정확한 파일명 사용
    
    # 파일 존재 여부 미리 확인
    if not os.path.exists(filepath_health) or not os.path.exists(filepath_infect):
        st.error('필요한 데이터 파일이 없습니다. 데이터를 확인해주세요.')
        return
    
    # 데이터 로드
    df_health = load_city_data(filepath_health) if os.path.exists(filepath_health) else pd.DataFrame()
    df_infect = load_city_data(filepath_infect) if os.path.exists(filepath_infect) else pd.DataFrame()
    
    if df_health.empty and df_infect.empty:
        st.error('분석할 데이터가 없습니다. 데이터 파일을 확인해주세요.')
        
        # 대안 제시
        st.markdown("### 📋 해결 방법")
        st.markdown("""
        1. **파일 경로 확인**: 위에 표시된 경로에 CSV 파일이 있는지 확인
        2. **파일명 확인**: 실제 파일명과 코드의 파일명이 일치하는지 확인
        3. **샘플 데이터 생성**: 테스트용 샘플 데이터를 생성할 수 있습니다
        """)
        
        # 샘플 데이터 생성 옵션
        if st.button("🎲 샘플 데이터로 테스트"):
            df_health_sample = create_sample_health_data()
            df_infect_sample = create_sample_infect_data()
            
            # 샘플 데이터로 분석 실행
            analyze_data(df_health_sample, df_infect_sample, "샘플")
        
        return
    
    # 실제 데이터로 분석 실행
    analyze_data(df_health, df_infect, "실제")
    
    # 정책 제안
    st.markdown('---')
    st.markdown('### 🎯 정책 제안')
    st.markdown('''
    **데이터 분석 결과를 바탕으로 한 맞춤형 정책 제안:**
    
    1. **고위험 지역 집중 관리**: 건강등급이 낮은 도시들을 대상으로 한 집중적인 건강관리 프로그램
    2. **예방 중심 정책**: 감염병 발생률이 높은 지역의 예방 체계 강화
    3. **지역별 맞춤 솔루션**: 클러스터링 결과를 바탕으로 한 지역 특성별 차별화된 접근
    4. **AI 기반 모니터링**: 실시간 건강 지표 모니터링 시스템 구축
    ''')

if __name__ == "__main__":
    create_city_dashboard()