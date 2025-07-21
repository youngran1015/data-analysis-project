import plotly.graph_objects as go
import streamlit as st

def plot_donut_chart(labels, values, title, colors=None):
    """
    Plotly 도넛그래프 생성 함수
    labels: 리스트(str) - 각 섹터 이름
    values: 리스트(float/int) - 각 섹터 값
    title: str - 그래프 제목
    colors: 리스트(str) - hex 색상코드(선택)
    """
    if colors is None:
        # 오렌지, 진청, 회색 계열 기본 팔레트
        colors = ['#F97316', '#0F172A', '#1E293B', '#F3F4F6', '#6B7280', '#D1D5DB']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='percent+label',
        insidetextorientation='radial',
        pull=[0.05 if v==max(values) else 0 for v in values],
        sort=False
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=22, color='#0F172A', family='Noto Sans KR'), x=0.5),
        showlegend=True,
        legend=dict(font=dict(size=14)),
        margin=dict(t=60, b=20, l=20, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Noto Sans KR', size=16, color='#0F172A')
    )
    return fig

# Streamlit에서 바로 사용 예시:
# labels = ['국산화', '해외조달']
# values = [70, 30]
# fig = plot_donut_chart(labels, values, '국산화율 vs 해외조달 비중')
# st.plotly_chart(fig) 