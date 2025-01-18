**Overview**

This project aims to explore the volatility of the MAKR. What is MAKR (Maker) and what is its Connection to DAI:   
MAKR is the native governance token of the MakerDAO ecosystem, which is responsible for the decentralized protocol behind the stablecoin DAI. MakerDAO allows users to create DAI by locking collateral in smart contracts, and MAKR holders participate in decision-making processes within the ecosystem. DAI is a popular decentralized stablecoin pegged to the US dollar, and its stability is crucial for various decentralized finance (DeFi) applications.   

Why Predicting MKR Can Be Valuable:   
Predicting the price movements of MKR is valuable because it serves as both a governance token and a key asset in the MakerDAO ecosystem. As MKR influences the health of DAI and MakerDAO's protocol, accurate predictions of its price could offer valuable insights into broader market sentiment, governance decisions, and the potential stability of DAI, benefiting DeFi participants and investors. 


**Features**

Crypto Data Fetcher: Retrieves OHLC data for selected cryptocurrencies and stablecoins using the Binance and Kraken API, with additional derived metrics and timezone conversion.    
Stock Data Fetcher: Fetches hourly stock data for predefined tickers using Yahoo Finance, enriching the data with calculated metrics.  
Feature Engeneering: Various technical features and custom calculations are created and the Tal-Lib library is used.  
MKRUSDT Analysis: Focuses on analyzing the governance token, examining factors influencing its price growth, and using machine learning models to close (y).  
Machine Learning Models: Implements models like Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost to predict price trends.
Flask API: A Flask-based API is included to interact with the data programmatically (optional, for deployment).
Docker Support: A Dockerfile is provided for easy deployment in containerized environments.  


**Datasets**     
Stablecoin Data:
[Link to Cryptocurrencies Dataset](https://drive.google.com/file/d/18IzkQYiodTNiIxmnG7lGrrdb-akB0C-l/view?usp=sharing)

Stock Data:
[Link to Stocks Dataset](https://drive.google.com/file/d/1d4PRGApTcuQaCAj16dOc9k79P3M2PaYF/view?usp=sharing)

Merged Data with Features:
[Link to merged Dataset](https://drive.google.com/file/d/1aImaDFQWnDEN1wliP5KTh2MwfqFSktEi/view?usp=sharing)

  
   

```bash
stable_coin/  
├── images/                                     # Contains the images that are generated through EDA     
│   ├── boxplot_mkr.png      
│   ├── correlation_matrix_mkr.png   
│   ├── distribution_price_change.png            
│   ├── timeseries_mkrusdt.png      
│   ├── timeseries_eur.png    
│   ├── price_change_correlation_with_volume.png 
├── README.md                      
├── notebook.ipynb/       
│   ├── get_coins                               # Fetches and processes cryptocurrency data     
│   ├── get_stocks                              # Fetches and processes stock data      
│   ├── feature engineering                     # Adds derived metrics for ML models      
│   ├── model evaluation and tuning             # Compares models and saves the best as a pickle file      
├── train.py                                    # Trains the best model            
├── predict.py                                  # Flask application for making predictions       
├── requirements.txt                            # List of required Python packages       
├── environment.yml                             # Conda environment file     
├── LICENSE      
├── Dockerfile                                  # For containerized deployment    
```

## Data Exploration:

MKRUSDT (Maker):  
Number of data points: 1862   
Close Price:   
    Mean: 1566.29  
    Min: 1063.00  
    Max: 2411.00  
    Standard Deviation: 298.14  
      
Price Change:  
    Mean: 0.50%  
    Max: 6.62%  
    Min: -4.03%  
       
Volume:      
    Mean: 572.94  
    Min: 16.25  
    Max: 8915.15  

7-Day Moving Average (7d_ma):
    Mean: 1564.15  
    Min: 201.77  
    Max: 2374.29  

7-Day Volatility:  
    Mean: 17.30%  
    Max: 751.05%  
    
DAIUSD (DAI Stablecoin):    
Number of data points: 613   
Maximum price change: 0.47%      


**Correlation for MKRUSDT** 

Key observations: 
Strong correlations: `7d_ma` and `30d_ma` show high correlation with `close` and `open`, indicating their importance for price trends.
`atr` (Average True Range) is moderately correlated with price indicators, highlighting its role in volatility analysis.
   
Indicators like `adx` and `rsi` show weak correlations with price-related variables but can provide additional signals.
`volume` and `volume_change` exhibit moderate correlations with certain price metrics, making them valuable for demand-supply analysis.  
`growth_future_1h` and `growth_future_24h` have weak correlations with other features, suggesting they could be challenging targets to predict directly.  
A combination of moving averages (`7d_ma`, `30d_ma`), volatility (`atr`), and volume-based features can provide a strong basis for predicting MKR price trends.   

![Correlation Matrix](images/correlation_matrix_mkr.png) 
 

**Boxplot for Closing Prices for MKRUSDT** 

To better understand the data distribution and identify potential outliers, a boxplot of the closing prices for MKRUSDT was generated:

![Boxplot](images/boxplot_mkr.png)   
         

Key observations:       
Distribution shows closing prices primarily clustered between 1400-1600 range. The Box plot indicates median price around 1500, with several outliers visible around 2200, suggesting occasional price spikes. The data appears to have moderate spread within the core trading range. Lower whisker extends to around 1000, indicating historical support level. The box (IQR) shows the middle 50% of price activity is relatively concentrated. The overall pattern suggests a somewhat stable trading range with occasional upside volatility.    
        
             
**Timeseries for MKRUSDT and DAIUSD** 

Key observations for MKRUSDT:  
Started around $1200, with initial sideways movement until early November.
Strong upward trend from November to early December, with a major price spike in early December reaching ~$2400. Significant volatility in December with multiple peaks above $2000 followed by a gradual downward trend since mid-December, currently showing bearish momentum, trading around $1400. Overall range: $1000-2400, with most activity between $1400-2000. Pattern suggests a completed pump and distribution phase.     
   

![Timeseries](images/timeseries_mkrusdt.png)   
         

Key observations for DAIUSD:       
Stable price action around $1.00 as expected for a stablecoin, with minimal volatility with most fluctuations staying within $0.999-1.001 range. Brief spike to $1.005 around January 9th and small spike to $1.002 on January 1st. Overall maintains excellent peg stability. Recent days (Jan 9-13) show slightly increased volatility but still within acceptable ranges.
   
![Timeseries](images/timeseries_daiusd.png)   
   
  
**Distribution of Price Change for MKRUSDT** 
       
Key observations:    
Distribution appears normal (bell-shaped) and is centered around 0, indicating balanced price movements. Most frequent changes are small (between -1 and +1). A few extreme outliers, especially on positive side are visible (up to +6). The Distribution tails extend from roughly -4 to +6. Peak frequency around 250 occurrences for smallest changes.

![Distribution of Price Change](images/distribution_price_change.png) 

    

### Machine Learning Models
  
**Linear Regression (LR)**    
Features with a higher accuracy drop: 'ppo', 'trix', 'atr'. Various features have no influence on the accuracy and could be considered for removal.  
    
![Distribution of Predicted Values for Linear Regression](images/predicted_values_distribution_lr.png)   
Analysis:   
Most predictions are centered around 0, with a sharp peak and minimal spread. This indicates that the model is predicting a narrow range of values, which could suggest underfitting or that the target variable has a limited variance.    
      
   
**Decision Trees (DT)**   
![Cross-Validation MSE Heatmap for Decision Tree](images/cross_validation_mse_vs_max_depth__dt.png)       
Validation Set Performance:   
Accuracy: 53.0%  
ROC-AUC: 0.530  
Confusion Matrix:  
True Negatives: 2506, False Positives: 2167  
False Negatives: 2093, True Positives: 2298  
F1-Score: 0.52  
Key Observations:  
#### 



**Random Forest (RF)**   
![Auc vs. Number of Trees for Random Forest](images/auc_vs_num_trees_diff_max_depth_rf.png)    
Validation Set Performance:   
AUC: 0.56 (indicating slightly better-than-random performance).  
Marginal improvements observed as n_estimators increase, with diminishing returns beyond 150 trees.  
Key Observations:  
Lower min_samples_leaf (1-3) values lead to overfitting and poor generalization.
Higher min_samples_leaf (≥50) oversimplifies the model, reducing performance.
Moderate max_depth (10) avoids overfitting while capturing relevant patterns.
The model achieves optimal performance by balancing flexibility (min_samples_leaf = 5) with ensemble size.   


**XGBoost**     
![Feature Importance For XGBOOST](images/feature_importance_xgboost.png)    
Evaluation:  
Time-based features (month, day, hour) suggest the model leverages seasonal patterns. Volatility indicates the model uses market fluctuations for predictions.  
Stock tickers such as DOCN (DigitalOcean), GRAB (Grab Holdings), and SMCI (Super Micro Computer) highlight a connection between traditional finance and crypto.  
DOCN: As a cloud infrastructure company, its stock volatility or market sentiment could reflect broader financial trends influencing crypto markets.
GRAB: A Southeast Asian tech company, its market performance may signal regional economic trends affecting investor sentiment, including in crypto.
SMCI: Specializing in high-performance computing, its stock could correlate with tech market trends, influencing investor confidence and risk sentiment in both traditional and crypto markets.  


### Installation
1. Clone the repository
bash
git clone https://github.com/your-repo.git
cd your-repo
2. Set up the environment
Using Conda:
bash
conda env create -f environment.yml
conda activate your-environment-name
Using pip:
bash
pip install -r requirements.txt

### How to Use
- Jupyter Notebook (notebook.ipynb):
Fetch cryptocurrency and stock market data:
Fetches data for cryptocurrencies and stablecoins defined in the coins list.
Processes the data and adds derived metrics (e.g., price change).
Saves the final dataset as stable_coins.csv.
Fetches hourly stock data for predefined tickers, adds derived metrics, and formats timestamps.
Combines all stock data into a single DataFrame and saves it as a CSV.
Logs missing or delisted stocks/cryptos as warnings or errors.
Perform feature engineering and derive metrics.
Evaluate multiple machine learning models.
Save the best models as .pkl files.
Train the Model:
Use train.py to train the best-performing model (default: XGBoost) on the processed data.
Save the trained model as a .pkl file.
Deploy the Model with Flask:
Use predict.py to deploy the model and provide predictions via a Flask API.

### Flask API
The repository includes a Flask API (`predict.py`) to interact with the trained XGBoost model. The API allows users to predict whether the price of USDC/USDT will grow positively within the next hour.

#### Steps to Use
1. **Start the Flask Server**  
   Ensure the conda environment is active and run:
   ```bash
   python predict.py
The API will start locally at http://0.0.0.0:8000.

2. **Make Predictions**
Send an HTTP POST request with the input features as JSON to the /predict endpoint.
EInput Example:
{
    "30d_ma": 102.5,
    "7d_ma": 101.0,
    "ticker=BTCUSDT": 1,
    "ticker=AAPL": 0
}

Output Example:
{
    "is_positive_growth_1h_future_probability": 0.67,
    "is_positive_growth_1h_future": true
}


### Run with Docker
To simplify deployment, a Dockerfile is provided. To build and run the Docker container:

## Build the Docker image:
docker build -t crypto-stock-analysis .
Run the container:
docker run -p 5000:5000 crypto-stock-analysis


License
This project is open-source and licensed under the MIT License.