"""
Core analysis functions for unemployment data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class UnemploymentAnalyzer:
    """Class for performing various analyses on unemployment data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with processed data.
        
        Args:
            data (pd.DataFrame): Processed unemployment data
        """
        self.data = data
        
    def trend_analysis(self) -> Dict[str, Any]:
        """
        Perform trend analysis on unemployment data.
        
        Returns:
            Dict[str, Any]: Trend analysis results
        """
        df = self.data.copy()
        
        # Overall trend
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Unemployment_Rate'].values
        
        # Linear trend
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_trend = linear_model.coef_[0]
        
        # Polynomial trend (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        
        # Calculate R-squared for both models
        linear_r2 = linear_model.score(X, y)
        poly_r2 = poly_model.score(X_poly, y)
        
        # Period-specific trends
        periods = ['Pre_COVID', 'COVID_Period', 'Post_COVID']
        period_trends = {}
        
        for period in periods:
            period_data = df[df[period]]
            if len(period_data) > 2:
                X_period = np.arange(len(period_data)).reshape(-1, 1)
                y_period = period_data['Unemployment_Rate'].values
                
                period_model = LinearRegression()
                period_model.fit(X_period, y_period)
                
                period_trends[period] = {
                    'slope': period_model.coef_[0],
                    'r_squared': period_model.score(X_period, y_period),
                    'start_rate': y_period[0],
                    'end_rate': y_period[-1],
                    'change': y_period[-1] - y_period[0]
                }
        
        return {
            'overall_trend': {
                'linear_slope': linear_trend,
                'linear_r2': linear_r2,
                'polynomial_r2': poly_r2,
                'direction': 'increasing' if linear_trend > 0 else 'decreasing'
            },
            'period_trends': period_trends
        }
    
    def seasonal_analysis(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in unemployment data.
        
        Returns:
            Dict[str, Any]: Seasonal analysis results
        """
        # Exclude COVID period for seasonal analysis
        normal_data = self.data[~self.data['COVID_Period']].copy()
        
        # Monthly averages
        monthly_stats = normal_data.groupby('Month')['Unemployment_Rate'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).round(2)
        
        # Add month names
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                      5: 'May', 6: 'June', 7: 'July', 8: 'August',
                      9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        monthly_stats['Month_Name'] = monthly_stats.index.map(month_names)
        
        # Seasonal statistics
        seasonal_range = monthly_stats['mean'].max() - monthly_stats['mean'].min()
        peak_month = monthly_stats['mean'].idxmax()
        low_month = monthly_stats['mean'].idxmin()
        
        # Quarterly analysis
        quarterly_stats = normal_data.groupby('Quarter')['Unemployment_Rate'].agg([
            'mean', 'std', 'count'
        ]).round(2)
        
        # Statistical test for seasonality (Kruskal-Wallis)
        month_groups = [normal_data[normal_data['Month'] == month]['Unemployment_Rate'].values 
                       for month in range(1, 13)]
        kruskal_stat, kruskal_p = stats.kruskal(*month_groups)
        
        return {
            'monthly_statistics': monthly_stats.to_dict('index'),
            'seasonal_summary': {
                'seasonal_range': seasonal_range,
                'peak_month': {'number': peak_month, 'name': month_names[peak_month]},
                'low_month': {'number': low_month, 'name': month_names[low_month]},
                'seasonality_test': {
                    'kruskal_statistic': kruskal_stat,
                    'p_value': kruskal_p,
                    'is_seasonal': kruskal_p < 0.05
                }
            },
            'quarterly_statistics': quarterly_stats.to_dict('index')
        }
    
    def covid_impact_analysis(self) -> Dict[str, Any]:
        """
        Analyze the impact of COVID-19 on unemployment rates.
        
        Returns:
            Dict[str, Any]: COVID-19 impact analysis results
        """
        pre_covid = self.data[self.data['Pre_COVID']]['Unemployment_Rate']
        covid_period = self.data[self.data['COVID_Period']]['Unemployment_Rate']
        post_covid = self.data[self.data['Post_COVID']]['Unemployment_Rate']
        
        # Basic statistics
        pre_covid_stats = {
            'mean': pre_covid.mean(),
            'std': pre_covid.std(),
            'min': pre_covid.min(),
            'max': pre_covid.max()
        }
        
        covid_stats = {
            'mean': covid_period.mean(),
            'std': covid_period.std(),
            'min': covid_period.min(),
            'max': covid_period.max(),
            'peak_date': self.data[self.data['COVID_Period'] & 
                                 (self.data['Unemployment_Rate'] == covid_period.max())]['Date'].iloc[0]
        }
        
        post_covid_stats = {
            'mean': post_covid.mean(),
            'std': post_covid.std(),
            'min': post_covid.min(),
            'max': post_covid.max()
        }
        
        # Impact calculations
        peak_impact = covid_period.max() - pre_covid.mean()
        peak_impact_pct = (peak_impact / pre_covid.mean()) * 100
        
        # Recovery analysis
        covid_data = self.data[self.data['COVID_Period']].copy()
        recovery_months = len(covid_data[covid_data['Unemployment_Rate'] > pre_covid.mean()])
        
        # Statistical tests
        # T-test between pre-COVID and post-COVID periods
        ttest_stat, ttest_p = stats.ttest_ind(pre_covid, post_covid)
        
        return {
            'period_statistics': {
                'pre_covid': pre_covid_stats,
                'covid_period': covid_stats,
                'post_covid': post_covid_stats
            },
            'impact_metrics': {
                'peak_impact_absolute': peak_impact,
                'peak_impact_percentage': peak_impact_pct,
                'recovery_months': recovery_months,
                'full_recovery': post_covid.mean() <= pre_covid.mean() + pre_covid.std()
            },
            'statistical_tests': {
                'pre_vs_post_ttest': {
                    'statistic': ttest_stat,
                    'p_value': ttest_p,
                    'significant_difference': ttest_p < 0.05
                }
            }
        }
    
    def volatility_analysis(self) -> Dict[str, Any]:
        """
        Analyze volatility in unemployment rates across different periods.
        
        Returns:
            Dict[str, Any]: Volatility analysis results
        """
        # Calculate rolling volatility (standard deviation)
        self.data['Rolling_Volatility_6'] = self.data['Unemployment_Rate'].rolling(window=6).std()
        self.data['Rolling_Volatility_12'] = self.data['Unemployment_Rate'].rolling(window=12).std()
        
        # Period-specific volatility
        periods = ['Pre_COVID', 'COVID_Period', 'Post_COVID']
        volatility_by_period = {}
        
        for period in periods:
            period_data = self.data[self.data[period]]['Unemployment_Rate']
            if len(period_data) > 1:
                volatility_by_period[period] = {
                    'standard_deviation': period_data.std(),
                    'coefficient_of_variation': period_data.std() / period_data.mean(),
                    'range': period_data.max() - period_data.min(),
                    'interquartile_range': period_data.quantile(0.75) - period_data.quantile(0.25)
                }
        
        return {
            'period_volatility': volatility_by_period,
            'overall_volatility': {
                'total_range': self.data['Unemployment_Rate'].max() - self.data['Unemployment_Rate'].min(),
                'overall_std': self.data['Unemployment_Rate'].std(),
                'coefficient_of_variation': self.data['Unemployment_Rate'].std() / self.data['Unemployment_Rate'].mean()
            }
        }
    
    def policy_insights(self) -> Dict[str, Any]:
        """
        Generate policy insights based on the analysis.
        
        Returns:
            Dict[str, Any]: Policy insights and recommendations
        """
        # Get analysis results
        trend_results = self.trend_analysis()
        seasonal_results = self.seasonal_analysis()
        covid_results = self.covid_impact_analysis()
        volatility_results = self.volatility_analysis()
        
        insights = []
        
        # COVID-19 insights
        if covid_results['impact_metrics']['peak_impact_percentage'] > 200:
            insights.append({
                'category': 'Crisis Response',
                'priority': 'High',
                'finding': f"COVID-19 caused a {covid_results['impact_metrics']['peak_impact_percentage']:.0f}% increase in unemployment",
                'recommendation': "Develop rapid-response economic stimulus frameworks for future crises"
            })
        
        # Recovery insights
        if covid_results['impact_metrics']['full_recovery']:
            insights.append({
                'category': 'Recovery',
                'priority': 'Medium',
                'finding': "Unemployment has returned to pre-pandemic levels",
                'recommendation': "Focus on maintaining current employment levels and preventing future spikes"
            })
        
        # Seasonal insights
        if seasonal_results['seasonal_summary']['seasonality_test']['is_seasonal']:
            peak_month = seasonal_results['seasonal_summary']['peak_month']['name']
            insights.append({
                'category': 'Seasonal Planning',
                'priority': 'Low',
                'finding': f"Unemployment typically peaks in {peak_month}",
                'recommendation': "Implement seasonal job programs and training during high-unemployment months"
            })
        
        # Volatility insights
        covid_volatility = volatility_results['period_volatility'].get('COVID_Period', {}).get('standard_deviation', 0)
        normal_volatility = volatility_results['period_volatility'].get('Pre_COVID', {}).get('standard_deviation', 0)
        
        if covid_volatility > normal_volatility * 2:
            insights.append({
                'category': 'Economic Stability',
                'priority': 'High',
                'finding': "COVID period showed extremely high volatility in unemployment",
                'recommendation': "Strengthen social safety nets and unemployment insurance systems"
            })
        
        return {
            'insights': insights,
            'summary_metrics': {
                'current_rate': self.data['Unemployment_Rate'].iloc[-1],
                'historical_average': self.data['Unemployment_Rate'].mean(),
                'covid_peak': covid_results['period_statistics']['covid_period']['max'],
                'recovery_status': 'Complete' if covid_results['impact_metrics']['full_recovery'] else 'Partial'
            }
        }

# Example usage
if __name__ == "__main__":
    from data_loader import UnemploymentDataLoader
    
    # Load data
    loader = UnemploymentDataLoader()
    data = loader.preprocess_data()
    
    # Initialize analyzer
    analyzer = UnemploymentAnalyzer(data)
    
    # Run analyses
    print("=== TREND ANALYSIS ===")
    trends = analyzer.trend_analysis()
    print(f"Overall trend: {trends['overall_trend']['direction']}")
    print(f"Linear slope: {trends['overall_trend']['linear_slope']:.4f}")
    
    print("\n=== SEASONAL ANALYSIS ===")
    seasonal = analyzer.seasonal_analysis()
    print(f"Seasonal range: {seasonal['seasonal_summary']['seasonal_range']:.2f}%")
    print(f"Peak month: {seasonal['seasonal_summary']['peak_month']['name']}")
    
    print("\n=== COVID IMPACT ===")
    covid = analyzer.covid_impact_analysis()
    print(f"Peak impact: {covid['impact_metrics']['peak_impact_percentage']:.1f}%")
    print(f"Recovery status: {'Complete' if covid['impact_metrics']['full_recovery'] else 'Partial'}")
    
    print("\n=== POLICY INSIGHTS ===")
    insights = analyzer.policy_insights()
    for insight in insights['insights']:
        print(f"- {insight['category']}: {insight['finding']}")