"""
Visualization functions for unemployment data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UnemploymentVisualizer:
    """Class for creating visualizations of unemployment data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with processed data.
        
        Args:
            data (pd.DataFrame): Processed unemployment data
        """
        self.data = data
        self.colors = {
            'pre_covid': '#2E8B57',  # Sea Green
            'covid': '#DC143C',      # Crimson
            'post_covid': '#4169E1', # Royal Blue
            'overall': '#1F4E79',    # Dark Blue
            'seasonal': '#FF8C00'    # Dark Orange
        }
    
    def plot_unemployment_trend(self, save_path: Optional[str] = None, 
                               interactive: bool = False) -> None:
        """
        Create a comprehensive unemployment trend plot.
        
        Args:
            save_path (Optional[str]): Path to save the plot
            interactive (bool): Whether to create interactive plot
        """
        if interactive:
            self._plot_interactive_trend()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Main trend plot
            ax1.plot(self.data['Date'], self.data['Unemployment_Rate'], 
                    linewidth=2, color=self.colors['overall'], alpha=0.8)
            
            # Highlight COVID period
            covid_data = self.data[self.data['COVID_Period']]
            ax1.plot(covid_data['Date'], covid_data['Unemployment_Rate'], 
                    linewidth=3, color=self.colors['covid'], label='COVID-19 Period')
            
            # Add moving averages
            ax1.plot(self.data['Date'], self.data['MA_6'], 
                    '--', alpha=0.7, label='6-Month MA')
            ax1.plot(self.data['Date'], self.data['MA_12'], 
                    '--', alpha=0.7, label='12-Month MA')
            
            ax1.set_title('Unemployment Rate Trends Over Time', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Unemployment Rate (%)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Rate changes plot
            ax2.bar(self.data['Date'], self.data['Rate_Change'], 
                   color=np.where(self.data['Rate_Change'] >= 0, 
                                self.colors['covid'], self.colors['pre_covid']),
                   alpha=0.7)
            
            ax2.set_title('Month-over-Month Rate Changes', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Rate Change (percentage points)', fontsize=12)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_interactive_trend(self) -> None:
        """Create interactive trend plot using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Unemployment Rate Trends', 'Month-over-Month Changes'),
            vertical_spacing=0.1
        )
        
        # Main trend
        fig.add_trace(
            go.Scatter(x=self.data['Date'], y=self.data['Unemployment_Rate'],
                      mode='lines', name='Unemployment Rate',
                      line=dict(color=self.colors['overall'], width=2)),
            row=1, col=1
        )
        
        # COVID period highlight
        covid_data = self.data[self.data['COVID_Period']]
        fig.add_trace(
            go.Scatter(x=covid_data['Date'], y=covid_data['Unemployment_Rate'],
                      mode='lines', name='COVID-19 Period',
                      line=dict(color=self.colors['covid'], width=3)),
            row=1, col=1
        )
        
        # Rate changes
        colors = ['red' if x >= 0 else 'green' for x in self.data['Rate_Change']]
        fig.add_trace(
            go.Bar(x=self.data['Date'], y=self.data['Rate_Change'],
                  name='Rate Change', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Unemployment Analysis Dashboard")
        fig.show()
    
    def plot_covid_impact(self, save_path: Optional[str] = None) -> None:
        """
        Create COVID-19 impact visualization.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Period comparison
        periods = ['Pre_COVID', 'COVID_Period', 'Post_COVID']
        period_means = [self.data[self.data[period]]['Unemployment_Rate'].mean() 
                       for period in periods]
        period_labels = ['Pre-COVID', 'COVID Period', 'Post-COVID']
        
        bars = ax1.bar(period_labels, period_means, 
                      color=[self.colors['pre_covid'], self.colors['covid'], 
                            self.colors['post_covid']])
        ax1.set_title('Average Unemployment by Period', fontweight='bold')
        ax1.set_ylabel('Average Unemployment Rate (%)')
        
        # Add value labels on bars
        for bar, value in zip(bars, period_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # COVID timeline
        covid_data = self.data[self.data['COVID_Period']].copy()
        ax2.plot(covid_data['Date'], covid_data['Unemployment_Rate'], 
                marker='o', linewidth=2, markersize=6, color=self.colors['covid'])
        ax2.set_title('COVID-19 Period Timeline', fontweight='bold')
        ax2.set_ylabel('Unemployment Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Recovery analysis
        recovery_data = self.data[self.data['Date'] >= '2020-04-01'].copy()
        ax3.plot(recovery_data['Date'], recovery_data['Unemployment_Rate'], 
                linewidth=2, color=self.colors['covid'])
        
        # Add pre-COVID average line
        pre_covid_avg = self.data[self.data['Pre_COVID']]['Unemployment_Rate'].mean()
        ax3.axhline(y=pre_covid_avg, color=self.colors['pre_covid'], 
                   linestyle='--', linewidth=2, label=f'Pre-COVID Avg ({pre_covid_avg:.1f}%)')
        
        ax3.set_title('Recovery Timeline', fontweight='bold')
        ax3.set_ylabel('Unemployment Rate (%)')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Impact statistics
        stats_data = {
            'Pre-COVID Avg': self.data[self.data['Pre_COVID']]['Unemployment_Rate'].mean(),
            'COVID Peak': self.data[self.data['COVID_Period']]['Unemployment_Rate'].max(),
            'Current Rate': self.data['Unemployment_Rate'].iloc[-1]
        }
        
        ax4.bar(stats_data.keys(), stats_data.values(), 
               color=[self.colors['pre_covid'], self.colors['covid'], self.colors['post_covid']])
        ax4.set_title('Key Statistics', fontweight='bold')
        ax4.set_ylabel('Unemployment Rate (%)')
        
        # Add value labels
        for i, (key, value) in enumerate(stats_data.items()):
            ax4.text(i, value + 0.2, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create seasonal analysis visualization.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        # Exclude COVID period for seasonal analysis
        normal_data = self.data[~self.data['COVID_Period']].copy()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly averages
        monthly_avg = normal_data.groupby('Month')['Unemployment_Rate'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        bars = ax1.bar(month_names, monthly_avg.values, color=self.colors['seasonal'], alpha=0.7)
        ax1.set_title('Average Unemployment by Month (Excluding COVID)', fontweight='bold')
        ax1.set_ylabel('Average Unemployment Rate (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Highlight peak and trough
        peak_idx = monthly_avg.idxmax() - 1
        trough_idx = monthly_avg.idxmin() - 1
        bars[peak_idx].set_color('red')
        bars[trough_idx].set_color('green')
        
        # Box plot by month
        monthly_data = [normal_data[normal_data['Month'] == month]['Unemployment_Rate'].values 
                       for month in range(1, 13)]
        
        box_plot = ax2.boxplot(monthly_data, labels=month_names, patch_artist=True)
        for patch in box_plot['boxes']:
            patch.set_facecolor(self.colors['seasonal'])
            patch.set_alpha(0.7)
        
        ax2.set_title('Monthly Unemployment Distribution', fontweight='bold')
        ax2.set_ylabel('Unemployment Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Quarterly analysis
        quarterly_avg = normal_data.groupby('Quarter')['Unemployment_Rate'].mean()
        quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        ax3.bar(quarter_labels, quarterly_avg.values, color=self.colors['pre_covid'], alpha=0.7)
        ax3.set_title('Average Unemployment by Quarter', fontweight='bold')
        ax3.set_ylabel('Average Unemployment Rate (%)')
        
        # Year-over-year comparison
        yearly_avg = normal_data.groupby('Year')['Unemployment_Rate'].mean()
        ax4.plot(yearly_avg.index, yearly_avg.values, marker='o', 
                linewidth=2, markersize=8, color=self.colors['overall'])
        ax4.set_title('Annual Average Unemployment (Excluding COVID)', fontweight='bold')
        ax4.set_ylabel('Average Unemployment Rate (%)')
        ax4.set_xlabel('Year')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all key visualizations.
        
        Args:
            save_path (Optional[str]): Path to save the dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Main trend (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data['Date'], self.data['Unemployment_Rate'], 
                linewidth=2, color=self.colors['overall'])
        
        # Highlight COVID period
        covid_data = self.data[self.data['COVID_Period']]
        ax1.plot(covid_data['Date'], covid_data['Unemployment_Rate'], 
                linewidth=3, color=self.colors['covid'], label='COVID-19 Period')
        
        ax1.set_title('Unemployment Rate Trends (2018-2024)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Unemployment Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Period comparison
        ax2 = fig.add_subplot(gs[1, 0])
        periods = ['Pre_COVID', 'COVID_Period', 'Post_COVID']
        period_means = [self.data[self.data[period]]['Unemployment_Rate'].mean() 
                       for period in periods]
        period_labels = ['Pre-COVID', 'COVID', 'Post-COVID']
        
        bars = ax2.bar(period_labels, period_means, 
                      color=[self.colors['pre_covid'], self.colors['covid'], 
                            self.colors['post_covid']])
        ax2.set_title('Period Averages', fontweight='bold')
        ax2.set_ylabel('Avg Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Monthly seasonality
        ax3 = fig.add_subplot(gs[1, 1])
        normal_data = self.data[~self.data['COVID_Period']]
        monthly_avg = normal_data.groupby('Month')['Unemployment_Rate'].mean()
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        ax3.bar(month_names, monthly_avg.values, color=self.colors['seasonal'], alpha=0.7)
        ax3.set_title('Seasonal Pattern', fontweight='bold')
        ax3.set_ylabel('Avg Rate (%)')
        
        # Volatility
        ax4 = fig.add_subplot(gs[1, 2])
        volatility_data = {
            'Pre-COVID': self.data[self.data['Pre_COVID']]['Unemployment_Rate'].std(),
            'COVID': self.data[self.data['COVID_Period']]['Unemployment_Rate'].std(),
            'Post-COVID': self.data[self.data['Post_COVID']]['Unemployment_Rate'].std()
        }
        
        ax4.bar(volatility_data.keys(), volatility_data.values(),
               color=[self.colors['pre_covid'], self.colors['covid'], self.colors['post_covid']])
        ax4.set_title('Volatility by Period', fontweight='bold')
        ax4.set_ylabel('Std Dev')
        ax4.tick_params(axis='x', rotation=45)
        
        # Rate changes
        ax5 = fig.add_subplot(gs[2, :])
        colors = np.where(self.data['Rate_Change'] >= 0, 
                         self.colors['covid'], self.colors['pre_covid'])
        ax5.bar(self.data['Date'], self.data['Rate_Change'], color=colors, alpha=0.7)
        ax5.set_title('Month-over-Month Rate Changes', fontweight='bold')
        ax5.set_ylabel('Rate Change (pp)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax5.grid(True, alpha=0.3)
        
        # Key statistics
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Calculate key stats
        current_rate = self.data['Unemployment_Rate'].iloc[-1]
        historical_avg = self.data['Unemployment_Rate'].mean()
        covid_peak = self.data[self.data['COVID_Period']]['Unemployment_Rate'].max()
        pre_covid_avg = self.data[self.data['Pre_COVID']]['Unemployment_Rate'].mean()
        
        stats_text = f"""
        KEY STATISTICS:
        Current Rate: {current_rate:.1f}%  |  Historical Average: {historical_avg:.1f}%  |  COVID Peak: {covid_peak:.1f}%  |  Pre-COVID Average: {pre_covid_avg:.1f}%
        
        INSIGHTS:
        • COVID-19 caused unemployment to spike from {pre_covid_avg:.1f}% to {covid_peak:.1f}% (a {((covid_peak/pre_covid_avg-1)*100):.0f}% increase)
        • Recovery took approximately {len(self.data[self.data['COVID_Period']])} months to return to pre-pandemic levels
        • Current unemployment rate is {'above' if current_rate > pre_covid_avg else 'at or below'} pre-COVID levels
        """
        
        ax6.text(0.05, 0.5, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightgray", alpha=0.8))
        
        plt.suptitle('Unemployment Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    from data_loader import UnemploymentDataLoader
    
    # Load data
    loader = UnemploymentDataLoader()
    data = loader.preprocess_data()
    
    # Create visualizer
    viz = UnemploymentVisualizer(data)
    
    # Create visualizations
    print("Creating unemployment trend plot...")
    viz.plot_unemployment_trend()
    
    print("Creating COVID impact analysis...")
    viz.plot_covid_impact()
    
    print("Creating seasonal analysis...")
    viz.plot_seasonal_analysis()
    
    print("Creating comprehensive dashboard...")
    viz.create_dashboard()