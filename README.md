# Unemployment Rate Analysis with Python

A comprehensive Python-based analysis of unemployment rate data with focus on COVID-19 impact, seasonal trends, and policy insights. This project provides both programmatic analysis tools and interactive visualizations for understanding unemployment patterns and their economic implications.

## 🌟 Features

- **Comprehensive Data Analysis**: Statistical analysis of unemployment trends using pandas, numpy, and scipy
- **COVID-19 Impact Assessment**: Detailed analysis of pandemic effects on unemployment rates
- **Seasonal Pattern Recognition**: Identification of recurring patterns and seasonal variations
- **Interactive Visualizations**: Multiple visualization options using matplotlib, seaborn, and plotly
- **Policy Insights Generation**: Data-driven recommendations for economic and social policies
- **Jupyter Notebook**: Complete analysis workflow in an interactive notebook
- **Streamlit Dashboard**: Web-based interactive dashboard for real-time exploration
- **Modular Architecture**: Clean, reusable code structure for easy extension

## 📊 Analysis Components

### 1. Data Processing (`src/data_loader.py`)
- Automated data loading and preprocessing
- Time-based feature engineering
- COVID-19 period identification
- Moving averages and rate change calculations

### 2. Statistical Analysis (`src/analysis.py`)
- Trend analysis with linear and polynomial regression
- Seasonal pattern detection using Kruskal-Wallis test
- COVID-19 impact quantification
- Volatility analysis across different periods
- Policy insight generation

### 3. Visualizations (`src/visualizations.py`)
- Time series plots with period highlighting
- COVID-19 impact comparison charts
- Seasonal analysis visualizations
- Comprehensive dashboard creation
- Interactive plotly charts

### 4. Interactive Dashboard (`streamlit_app.py`)
- Web-based interface for data exploration
- Real-time chart updates
- Multiple analysis views
- Export capabilities

## 🛠 Technologies Used

- **Data Analysis**: pandas, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Testing**: scipy.stats, statsmodels
- **Interactive Dashboard**: streamlit
- **Notebook Environment**: jupyter
- **Development**: Python 3.8+

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/unemployment-analysis-python.git
cd unemployment-analysis-python
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
# Run individual analysis modules
python src/data_loader.py
python src/analysis.py
python src/visualizations.py

# Launch Jupyter notebook
jupyter notebook notebooks/unemployment_analysis.ipynb

# Launch Streamlit dashboard
streamlit run streamlit_app.py
```

## 📈 Key Findings

### COVID-19 Impact Analysis
- **Peak Impact**: Unemployment spiked from 3.5% to 14.8% in 2 months (320% increase)
- **Recovery Speed**: Faster recovery than historical recessions (8 months to return below 7%)
- **Policy Effectiveness**: Rapid fiscal and monetary response accelerated recovery

### Seasonal Patterns
- **Winter Peak**: Unemployment typically higher in winter months
- **Statistical Significance**: Kruskal-Wallis test confirms seasonal patterns
- **Range**: Seasonal variation of approximately 0.5-0.8 percentage points

### Current Status
- **Recovery**: Unemployment returned to pre-pandemic levels by late 2021
- **Stability**: Current rates around 3.7-4.0%, consistent with full employment
- **Volatility**: Post-COVID period shows reduced volatility compared to pandemic period

## 📁 Project Structure

```
unemployment-analysis-python/
├── data/
│   └── unemployment_data.csv          # Dataset
├── src/
│   ├── data_loader.py                 # Data loading and preprocessing
│   ├── analysis.py                    # Statistical analysis functions
│   └── visualizations.py             # Visualization functions
├── notebooks/
│   └── unemployment_analysis.ipynb    # Complete analysis notebook
├── streamlit_app.py                   # Interactive web dashboard
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore file
```

## 🔧 Usage Examples

### Basic Analysis
```python
from src.data_loader import UnemploymentDataLoader
from src.analysis import UnemploymentAnalyzer

# Load data
loader = UnemploymentDataLoader('data/unemployment_data.csv')
data = loader.preprocess_data()

# Perform analysis
analyzer = UnemploymentAnalyzer(data)
trend_results = analyzer.trend_analysis()
covid_results = analyzer.covid_impact_analysis()
```

### Visualization
```python
from src.visualizations import UnemploymentVisualizer

# Create visualizer
viz = UnemploymentVisualizer(data)

# Generate plots
viz.plot_unemployment_trend()
viz.plot_covid_impact()
viz.create_dashboard()
```

### Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

## 📊 Data Sources

The analysis uses unemployment rate data with the following structure:
- **Date**: Monthly data points (YYYY-MM format)
- **Unemployment_Rate**: Seasonally adjusted unemployment percentage
- **Region**: Geographic region (National)
- **Seasonally_Adjusted**: Boolean indicator

Data is based on U.S. Bureau of Labor Statistics methodology and covers January 2018 through June 2024.

## 🎯 Policy Insights

### Crisis Response
- **Emergency Preparedness**: Flexible unemployment insurance systems needed
- **Rapid Response**: Quick economic stimulus deployment is crucial
- **Monitoring**: Leading indicators essential for early intervention

### Long-term Strategy
- **Labor Market Resilience**: Investment in adaptability and reskilling
- **Social Safety Nets**: Strengthened support systems for future crises
- **Sustainable Growth**: Focus on maintaining employment stability

## 📈 Visualization Gallery

The project generates multiple types of visualizations:

1. **Time Series Analysis**: Unemployment trends with period highlighting
2. **COVID-19 Impact**: Before/during/after comparison charts
3. **Seasonal Patterns**: Monthly and quarterly analysis
4. **Statistical Dashboards**: Comprehensive overview with key metrics
5. **Interactive Charts**: Plotly-based interactive visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- U.S. Bureau of Labor Statistics for unemployment data methodology
- Python data science community for excellent libraries
- Contributors and users of this analysis framework

## 📞 Contact

For questions, suggestions, or collaborations:
- GitHub Issues: [Create an issue](https://github.com/yourusername/unemployment-analysis-python/issues)
- Email: your.email@example.com

## 🔄 Updates and Roadmap

### Recent Updates
- Added interactive Streamlit dashboard
- Implemented comprehensive statistical testing
- Enhanced visualization capabilities
- Added policy insight generation

### Future Enhancements
- Real-time data integration
- Machine learning forecasting models
- Regional analysis capabilities
- Advanced econometric modeling

---

Built with ❤️ for data-driven economic analysis and policy research.
