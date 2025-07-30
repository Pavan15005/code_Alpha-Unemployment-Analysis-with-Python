"""
Streamlit web application for unemployment analysis dashboard.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append('src')

from data_loader import UnemploymentDataLoader
from analysis import UnemploymentAnalyzer
from visualizations import UnemploymentVisualizer

# Configure page
st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f4e79;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache unemployment data."""
    loader = UnemploymentDataLoader('data/unemployment_data.csv')
    data = loader.preprocess_data()
    return data, loader

@st.cache_data
def get_analysis_results(data):
    """Get and cache analysis results."""
    analyzer = UnemploymentAnalyzer(data)
    
    trend_results = analyzer.trend_analysis()
    seasonal_results = analyzer.seasonal_analysis()
    covid_results = analyzer.covid_impact_analysis()
    policy_results = analyzer.policy_insights()
    volatility_results = analyzer.volatility_analysis()
    
    return {
        'trend': trend_results,
        'seasonal': seasonal_results,
        'covid': covid_results,
        'policy': policy_results,
        'volatility': volatility_results
    }

def create_main_chart(data):
    """Create the main unemployment trend chart."""
    fig = go.Figure()
    
    # Main trend line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Unemployment_Rate'],
        mode='lines',
        name='Unemployment Rate',
        line=dict(color='#1f4e79', width=2),
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>'
    ))
    
    # Highlight COVID period
    covid_data = data[data['COVID_Period']]
    fig.add_trace(go.Scatter(
        x=covid_data['Date'],
        y=covid_data['Unemployment_Rate'],
        mode='lines',
        name='COVID-19 Period',
        line=dict(color='#dc143c', width=3),
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.1f}%<br><i>COVID Period</i><extra></extra>'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA_6'],
        mode='lines',
        name='6-Month MA',
        line=dict(color='#ff8c00', width=1, dash='dash'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Unemployment Rate Trends (2018-2024)',
        xaxis_title='Date',
        yaxis_title='Unemployment Rate (%)',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_covid_impact_chart(data):
    """Create COVID-19 impact comparison chart."""
    # Calculate period averages
    pre_covid_avg = data[data['Pre_COVID']]['Unemployment_Rate'].mean()
    covid_avg = data[data['COVID_Period']]['Unemployment_Rate'].mean()
    post_covid_avg = data[data['Post_COVID']]['Unemployment_Rate'].mean()
    covid_peak = data[data['COVID_Period']]['Unemployment_Rate'].max()
    
    fig = go.Figure()
    
    # Bar chart
    periods = ['Pre-COVID<br>Average', 'COVID<br>Peak', 'COVID<br>Average', 'Post-COVID<br>Average']
    values = [pre_covid_avg, covid_peak, covid_avg, post_covid_avg]
    colors = ['#2e8b57', '#dc143c', '#ff6347', '#4169e1']
    
    fig.add_trace(go.Bar(
        x=periods,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Rate: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='COVID-19 Impact on Unemployment Rates',
        yaxis_title='Unemployment Rate (%)',
        height=400
    )
    
    return fig

def create_seasonal_chart(data):
    """Create seasonal analysis chart."""
    # Exclude COVID period
    normal_data = data[~data['COVID_Period']]
    monthly_avg = normal_data.groupby('Month')['Unemployment_Rate'].mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=month_names,
        y=monthly_avg.values,
        marker_color='#ff8c00',
        text=[f'{v:.2f}%' for v in monthly_avg.values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Average Rate: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Unemployment by Month (Excluding COVID Period)',
        xaxis_title='Month',
        yaxis_title='Average Unemployment Rate (%)',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Unemployment Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides comprehensive analysis of unemployment rate data from 2018-2024, 
    focusing on trends, COVID-19 impact, seasonal patterns, and policy insights.
    """)
    
    # Load data
    try:
        data, loader = load_data()
        analysis_results = get_analysis_results(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Overview", "Trend Analysis", "COVID-19 Impact", "Seasonal Patterns", "Policy Insights"]
    )
    
    # Data summary in sidebar
    st.sidebar.header("Data Summary")
    summary = loader.get_data_summary()
    st.sidebar.metric("Total Records", summary['total_records'])
    st.sidebar.metric("Current Rate", f"{summary['unemployment_stats']['current']:.1f}%")
    st.sidebar.metric("Historical Average", f"{summary['unemployment_stats']['mean']:.1f}%")
    st.sidebar.metric("COVID Peak", f"{summary['covid_impact']['covid_peak']:.1f}%")
    
    # Main content based on selected page
    if page == "Overview":
        st.header("📈 Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Rate",
                f"{data['Unemployment_Rate'].iloc[-1]:.1f}%",
                f"{data['Rate_Change'].iloc[-1]:+.1f}pp"
            )
        
        with col2:
            st.metric(
                "Historical Average",
                f"{data['Unemployment_Rate'].mean():.1f}%"
            )
        
        with col3:
            covid_peak = data[data['COVID_Period']]['Unemployment_Rate'].max()
            st.metric(
                "COVID Peak",
                f"{covid_peak:.1f}%"
            )
        
        with col4:
            pre_covid_avg = data[data['Pre_COVID']]['Unemployment_Rate'].mean()
            current_rate = data['Unemployment_Rate'].iloc[-1]
            recovery_status = "✅ Recovered" if current_rate <= pre_covid_avg + 0.5 else "⚠️ Recovering"
            st.metric(
                "Recovery Status",
                recovery_status
            )
        
        # Main chart
        st.plotly_chart(create_main_chart(data), use_container_width=True)
        
        # Key insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔍 Key Insights")
        
        insights = analysis_results['policy']['insights']
        for insight in insights[:3]:  # Show top 3 insights
            st.write(f"**{insight['category']}**: {insight['finding']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Trend Analysis":
        st.header("📊 Trend Analysis")
        
        trend_results = analysis_results['trend']
        
        # Overall trend metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Trend")
            direction = trend_results['overall_trend']['direction']
            slope = trend_results['overall_trend']['linear_slope']
            r2 = trend_results['overall_trend']['linear_r2']
            
            st.write(f"**Direction**: {direction.title()}")
            st.write(f"**Slope**: {slope:.6f} per month")
            st.write(f"**R-squared**: {r2:.4f}")
        
        with col2:
            st.subheader("Period Comparison")
            for period, stats in trend_results['period_trends'].items():
                period_name = period.replace('_', ' ').title()
                st.write(f"**{period_name}**: {stats['change']:+.2f}pp total change")
        
        # Trend chart
        st.plotly_chart(create_main_chart(data), use_container_width=True)
        
        # Volatility analysis
        st.subheader("📈 Volatility Analysis")
        volatility_results = analysis_results['volatility']
        
        vol_col1, vol_col2, vol_col3 = st.columns(3)
        
        for i, (period, stats) in enumerate(volatility_results['period_volatility'].items()):
            period_name = period.replace('_', ' ').title()
            col = [vol_col1, vol_col2, vol_col3][i]
            
            with col:
                st.metric(
                    f"{period_name} Volatility",
                    f"{stats['standard_deviation']:.2f}%",
                    f"Range: {stats['range']:.1f}pp"
                )
    
    elif page == "COVID-19 Impact":
        st.header("🦠 COVID-19 Impact Analysis")
        
        covid_results = analysis_results['covid']
        
        # Impact metrics
        col1, col2, col3, col4 = st.columns(4)
        
        impact_metrics = covid_results['impact_metrics']
        
        with col1:
            st.metric(
                "Peak Impact",
                f"+{impact_metrics['peak_impact_absolute']:.1f}pp"
            )
        
        with col2:
            st.metric(
                "Percentage Increase",
                f"+{impact_metrics['peak_impact_percentage']:.0f}%"
            )
        
        with col3:
            st.metric(
                "Recovery Period",
                f"{impact_metrics['recovery_months']} months"
            )
        
        with col4:
            recovery_status = "✅ Complete" if impact_metrics['full_recovery'] else "⚠️ Partial"
            st.metric(
                "Recovery Status",
                recovery_status
            )
        
        # COVID impact chart
        st.plotly_chart(create_covid_impact_chart(data), use_container_width=True)
        
        # Period statistics
        st.subheader("📊 Period Statistics")
        
        period_stats = covid_results['period_statistics']
        
        stats_df = pd.DataFrame({
            'Period': ['Pre-COVID', 'COVID Period', 'Post-COVID'],
            'Mean (%)': [
                period_stats['pre_covid']['mean'],
                period_stats['covid_period']['mean'],
                period_stats['post_covid']['mean']
            ],
            'Std Dev (%)': [
                period_stats['pre_covid']['std'],
                period_stats['covid_period']['std'],
                period_stats['post_covid']['std']
            ],
            'Min (%)': [
                period_stats['pre_covid']['min'],
                period_stats['covid_period']['min'],
                period_stats['post_covid']['min']
            ],
            'Max (%)': [
                period_stats['pre_covid']['max'],
                period_stats['covid_period']['max'],
                period_stats['post_covid']['max']
            ]
        })
        
        st.dataframe(stats_df.round(2), use_container_width=True)
    
    elif page == "Seasonal Patterns":
        st.header("🗓️ Seasonal Analysis")
        
        seasonal_results = analysis_results['seasonal']
        
        # Seasonal summary
        summary = seasonal_results['seasonal_summary']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Seasonal Range",
                f"{summary['seasonal_range']:.2f}pp"
            )
        
        with col2:
            st.metric(
                "Peak Month",
                summary['peak_month']['name']
            )
        
        with col3:
            st.metric(
                "Low Month",
                summary['low_month']['name']
            )
        
        # Seasonal chart
        st.plotly_chart(create_seasonal_chart(data), use_container_width=True)
        
        # Seasonality test
        seasonality_test = summary['seasonality_test']
        
        st.subheader("🧪 Statistical Test for Seasonality")
        
        test_col1, test_col2 = st.columns(2)
        
        with test_col1:
            st.metric(
                "Kruskal-Wallis Statistic",
                f"{seasonality_test['kruskal_statistic']:.4f}"
            )
        
        with test_col2:
            significance = "✅ Significant" if seasonality_test['is_seasonal'] else "❌ Not Significant"
            st.metric(
                "Seasonality",
                significance,
                f"p-value: {seasonality_test['p_value']:.6f}"
            )
        
        # Monthly statistics table
        st.subheader("📅 Monthly Statistics")
        
        monthly_stats = seasonal_results['monthly_statistics']
        monthly_df = pd.DataFrame(monthly_stats).T
        monthly_df['Month_Name'] = monthly_df['Month_Name']
        monthly_df = monthly_df[['Month_Name', 'mean', 'std', 'min', 'max', 'count']]
        monthly_df.columns = ['Month', 'Mean (%)', 'Std Dev (%)', 'Min (%)', 'Max (%)', 'Count']
        
        st.dataframe(monthly_df.round(2), use_container_width=True)
    
    elif page == "Policy Insights":
        st.header("💡 Policy Insights & Recommendations")
        
        policy_results = analysis_results['policy']
        
        # Summary metrics
        st.subheader("📊 Summary Metrics")
        
        metrics = policy_results['summary_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Rate", f"{metrics['current_rate']:.1f}%")
        
        with col2:
            st.metric("Historical Average", f"{metrics['historical_average']:.1f}%")
        
        with col3:
            st.metric("COVID Peak", f"{metrics['covid_peak']:.1f}%")
        
        with col4:
            st.metric("Recovery Status", metrics['recovery_status'])
        
        # Policy insights
        st.subheader("🎯 Key Insights & Recommendations")
        
        insights = policy_results['insights']
        
        for i, insight in enumerate(insights, 1):
            with st.expander(f"{i}. {insight['category']} (Priority: {insight['priority']})"):
                st.write(f"**Finding**: {insight['finding']}")
                st.write(f"**Recommendation**: {insight['recommendation']}")
        
        # Additional recommendations
        st.subheader("📋 Policy Recommendations")
        
        st.markdown("""
        ### Short-term Measures
        - Maintain flexible unemployment insurance systems
        - Prepare rapid-response economic stimulus frameworks
        - Monitor leading indicators for early intervention
        - Support job retraining and reskilling programs
        
        ### Long-term Strategies
        - Invest in labor market adaptability and resilience
        - Develop countercyclical fiscal policy frameworks
        - Strengthen social safety nets for future crises
        - Focus on sustainable employment growth
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Source**: Simulated unemployment data based on U.S. Bureau of Labor Statistics methodology  
    **Analysis Period**: January 2018 - June 2024  
    **Last Updated**: 2024
    """)

if __name__ == "__main__":
    main()