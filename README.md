# Smart Portfolio

A sophisticated portfolio optimization tool built with Python, featuring multiple optimization algorithms and a modern Streamlit interface.

## 🚀 Features

- **Data Management**: Fetch and process financial data using Yahoo Finance
- **Portfolio Optimization**: Multiple optimization strategies including SGD-based approaches
- **Performance Metrics**: Comprehensive portfolio analysis and risk metrics
- **Interactive UI**: Modern web interface built with Streamlit
- **Testing Suite**: Comprehensive test coverage with pytest

## 📁 Project Structure

```
smart-portfolio/
├── .github/
│   └── workflows/
│       └── python-app.yml      # CI/CD pipeline
├── portfolio/
│   ├── __init__.py            # Package initialization
│   ├── data.py                # Data fetching and processing
│   ├── metrics.py             # Portfolio performance metrics
│   ├── optimize.py            # Traditional optimization methods
│   ├── optimize_sgd.py        # SGD-based optimization
│   └── ui.py                  # Streamlit user interface
├── tests/
│   ├── test_data.py           # Data module tests
│   ├── test_metrics.py        # Metrics module tests
│   └── test_optimize_sgd.py   # SGD optimization tests
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd smart-portfolio
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Application
```bash
streamlit run portfolio/ui.py
```

### Running Tests
```bash
pytest tests/
```

### Running Tests with Coverage
```bash
pytest tests/ --cov=portfolio
```

## 📊 Modules

### `portfolio.data`
- Fetches financial data from Yahoo Finance
- Processes and cleans market data
- Handles data validation and error checking

### `portfolio.metrics`
- Calculates portfolio performance metrics
- Risk analysis and Sharpe ratio calculations
- Drawdown analysis and volatility measures

### `portfolio.optimize`
- Traditional portfolio optimization methods
- Mean-variance optimization
- Risk parity and other allocation strategies

### `portfolio.optimize_sgd`
- Stochastic Gradient Descent optimization
- Custom loss functions for portfolio optimization
- Advanced optimization techniques

### `portfolio.ui`
- Streamlit-based web interface
- Interactive charts and visualizations
- Real-time portfolio analysis dashboard

## 🧪 Testing

The project includes comprehensive tests for all modules:

- **Data Tests**: Validate data fetching and processing
- **Metrics Tests**: Ensure accurate performance calculations
- **Optimization Tests**: Verify optimization algorithm correctness

## 🔄 CI/CD

Automated testing and deployment using GitHub Actions:
- Runs on Python 3.8+
- Executes full test suite
- Checks code quality and coverage

## 📈 Getting Started

1. **Install the package**: Follow the installation steps above
2. **Run the app**: `streamlit run portfolio/ui.py`
3. **Input your portfolio**: Enter stock symbols and weights
4. **Analyze**: View optimization results and metrics
5. **Optimize**: Use built-in optimization algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Yahoo Finance data fetching
- **PyPortfolioOpt**: Portfolio optimization library
- **pytest**: Testing framework 