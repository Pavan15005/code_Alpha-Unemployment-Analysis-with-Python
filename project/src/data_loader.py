"""
Data loading and preprocessing module for unemployment analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class UnemploymentDataLoader:
    """Class to handle loading and preprocessing of unemployment data."""
    
    def __init__(self, data_path: str = 'data/unemployment_data.csv'):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the unemployment data CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load unemployment data from CSV file.
        
        Returns:
            pd.DataFrame: Raw unemployment data
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.raw_data)} records from {self.data_path}")
            return self.raw_data
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the unemployment data.
        
        Returns:
            pd.DataFrame: Processed unemployment data
        """
        if self.raw_data is None:
            self.load_data()
        
        # Create a copy for processing
        df = self.raw_data.copy()
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Add additional time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Month_Name'] = df['Date'].dt.strftime('%B')
        
        # Create COVID-19 period indicators
        df['Pre_COVID'] = df['Date'] < '2020-03-01'
        df['COVID_Period'] = (df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-12-31')
        df['Post_COVID'] = df['Date'] > '2021-12-31'
        
        # Calculate rate changes
        df['Rate_Change'] = df['Unemployment_Rate'].diff()
        df['Rate_Change_Pct'] = df['Unemployment_Rate'].pct_change() * 100
        
        # Calculate moving averages
        df['MA_3'] = df['Unemployment_Rate'].rolling(window=3).mean()
        df['MA_6'] = df['Unemployment_Rate'].rolling(window=6).mean()
        df['MA_12'] = df['Unemployment_Rate'].rolling(window=12).mean()
        
        self.processed_data = df
        print("Data preprocessing completed successfully.")
        return df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the unemployment data.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m'),
                'end': df['Date'].max().strftime('%Y-%m')
            },
            'unemployment_stats': {
                'mean': df['Unemployment_Rate'].mean(),
                'median': df['Unemployment_Rate'].median(),
                'std': df['Unemployment_Rate'].std(),
                'min': df['Unemployment_Rate'].min(),
                'max': df['Unemployment_Rate'].max(),
                'current': df['Unemployment_Rate'].iloc[-1]
            },
            'covid_impact': {
                'pre_covid_avg': df[df['Pre_COVID']]['Unemployment_Rate'].mean(),
                'covid_peak': df[df['COVID_Period']]['Unemployment_Rate'].max(),
                'post_covid_avg': df[df['Post_COVID']]['Unemployment_Rate'].mean()
            }
        }
        
        return summary
    
    def get_period_data(self, period: str) -> pd.DataFrame:
        """
        Get data for a specific period.
        
        Args:
            period (str): 'pre_covid', 'covid', 'post_covid', or 'all'
            
        Returns:
            pd.DataFrame: Filtered data for the specified period
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        
        if period == 'pre_covid':
            return df[df['Pre_COVID']].copy()
        elif period == 'covid':
            return df[df['COVID_Period']].copy()
        elif period == 'post_covid':
            return df[df['Post_COVID']].copy()
        else:
            return df.copy()

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = UnemploymentDataLoader()
    
    # Load and process data
    data = loader.preprocess_data()
    
    # Display summary
    summary = loader.get_data_summary()
    print("\n=== DATA SUMMARY ===")
    print(f"Total Records: {summary['total_records']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Current Unemployment Rate: {summary['unemployment_stats']['current']:.1f}%")
    print(f"Historical Average: {summary['unemployment_stats']['mean']:.1f}%")
    print(f"COVID Peak: {summary['covid_impact']['covid_peak']:.1f}%")
    
    # Display first few rows
    print("\n=== SAMPLE DATA ===")
    print(data[['Date', 'Unemployment_Rate', 'Year', 'Month', 'COVID_Period']].head(10))