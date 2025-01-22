# Overview

**This project focuses on analyzing the price volatility of MKR, the governance token of MakerDAO, and predicting its price changes using machine learning models.**


MKR is the native governance token of the MakerDAO ecosystem, which is responsible for the decentralized protocol behind the stablecoin DAI. MakerDAO allows users to create DAI by locking collateral in smart contracts, and MKR holders participate in decision-making processes within the ecosystem. DAI is a popular decentralized stablecoin pegged to the US dollar, and its stability is crucial for various decentralized finance (DeFi) applications.   

**Why Predicting MKR Can Be Valuable:**       
Predicting the price movements of MKR is valuable because it serves as both a governance token and a key asset in the MakerDAO ecosystem. As MKR influences the health of DAI and MakerDAO's protocol, accurate predictions of its price could offer valuable insights into broader market sentiment, governance decisions, and the potential stability of DAI, benefiting DeFi participants and investors. 

In this project y represents the percentage change in the closing price of MKR over consecutive time periods. It is calculated as:
```bash
y = (close - close_lag_1) / close_lag_1
```
   
close:   
The closing price of MKR at the current time step.   
close_lag_1:    
The closing price of MKR at the previous time step.   
Why Use Percentage Change:     
Because it normalizes the price movements, making the model less sensitive to the absolute price levels of MKR. This approach ensures that the model captures relative price movements, which are more relevant for understanding volatility and predicting market trends, especially in the context of a highly volatile asset like MKR.   
y is a continuous variable representing the relative change in the price of MKR. Positive values indicate an increase in price, while negative values represent a decrease. The values are expressed as decimals (e.g., 0.05 = 5% increase, -0.03 = 3% decrease).   
Analyzing and predicting the percentage change in MKR's price helps uncover patterns in its volatility and behavior. This is particularly valuable because MKR's price dynamics have a direct impact on the stability of DAI and the MakerDAO ecosystem, which are critical for decentralized finance (DeFi) applications. By focusing on percentage changes, this project aims to provide insights that are actionable for both investors and participants in the DeFi space.






# Features

Crypto Data Fetcher: Retrieves OHLC data for selected cryptocurrencies and stablecoins using the Binance and Kraken API, with additional derived metrics and timezone conversion.    
Stock Data Fetcher: Fetches hourly stock data for predefined tickers using Yahoo Finance, enriching the data with calculated metrics.  
Feature Engeneering: Various technical features and custom calculations are created and the Tal-Lib library is used.  
MKRUSDT Analysis: Focuses on analyzing the governance token, examining factors influencing its price growth, and using machine learning models to close (y).  
Machine Learning Models: Implements models like Linear Regression (LR), Decision Trees (DT), Random Forest (RF), and XGBoost to predict price trends.
Flask: Flask is included to interact with the data programmatically (optional, for deployment).
Docker Support: A Dockerfile is provided for easy deployment in containerized environments.  


# Datasets  
Crypto Data:      
[Link to Cryptocurrencies Binance Dataset](https://drive.google.com/file/d/1voYH8gYeAXWd2MIM7w4720hSbXBrOpdc/view?usp=sharing)    

[Link to Cryptocurrencies Kraken Dataset](https://drive.google.com/file/d/1ha7QAT9VKI43hVwTo3mZA23tKC6hQ0c9/view?usp=sharing)   

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
`7d_ma` and `30d_ma` show high correlation with `close` and `open`, indicating their importance for price trends.
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

    

# Machine Learning Models
  
  
Target Variable Analysis: y
Mean (y): 0.0557  
Standard Deviation (y): 5.3460  
![Histogram of y](images/y_histogram.png) 
Description:
The histogram illustrates the frequency distribution of the target variable y across the Train, Validation, and Test datasets. 
The majority of values are concentrated near 0. Extreme outliers exist, with some values exceeding 5000. The distribution is highly skewed, with most values clustered in a small range and a few values significantly larger.
Interpretation:  
The extreme outliers can adversely affect the model by increasing error and reducing prediction accuracy.
The skewness indicates potential difficulty for the model in correctly predicting y.  
   
![Boxplot of y](images/y_boxplot.png) 
Description:
The boxplot highlights the distribution of y across Train, Validation, and Test datasets, as well as the presence of outliers. The Interquartile Range (IQR) is small, suggesting that most data points are closely clustered.
Numerous strong outliers exceed 1000.
This aligns with the histogram: the majority of values are small, with a few extreme values.
These outliers can significantly distort metrics like MSE and RMSE during training and validation.
To address this, further analysis is required to decide whether to remove or transform the outliers.
  
*Transformation of y*   
To address the skewness and extreme outliers, a logarithmic transformation was applied to y:
```bash
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)
```
![Histogram of y](images/y_log_histogram.png) 
![Boxplot of y](images/y_log_boxplot.png) 



  
**Linear Regression (LR)**    
Features with a higher accuracy drop: 'ppo', 'trix', 'atr'. Various features have no influence on the accuracy and could be considered for removal.  
    
![Distribution of Predicted Values for Linear Regression](images/predicted_values_distribution_lr.png)   
Analysis:   
Most predictions are centered around 0, with a sharp peak and minimal spread. This indicates that the model is predicting a narrow range of values, which could suggest underfitting or that the target variable has a limited variance.    
      
   
**Decision Trees (DT)**   
       
![Cross-Validation MSE Heatmap for Decision Tree](images/cross_validation_mse_vs_max_depth__dt.png)    

Key observations:  
For min_samples_leaf, higher values (13-15) yield better results with MSE ~12.9, depth has minimal impact (MSE stable across depths) except for min_samples_leaf=1 hich shows significant deterioration at depth=4 (MSE spikes to ~34).   
Optimal configuration found: max_depth=4, min_samples_leaf=13, achieving MSE=12.9. This suggests the model benefits from higher leaf sample restrictions to prevent overfitting.   
   
 
**Random Forest (RF)**    

![MSE vs Number of Trees for Different Minimum Sample Leafs Random Forest](images/mse_vs_num_trees_diff_min_samples_rf.png)    
Key observation: 
The best number of trees (n_estimators): 180, achieving an MSE of ~2900.731. The best max_depth: 10, striking a balance between underfitting (depth=5) and overfitting (depth=15). The best min_samples_leaf: 1, enabling the model to capture detailed patterns.
Final model performance:   
MSE: 2870.841, RMSE: 53.580, R² Score: 0.085.
Despite reasonable RMSE, the low R² score (8.5%) suggests limited explanatory power.    
      
![Residual Plot For Random Forest](images/residuals_rf.png)
Key observation:
The residual plot shows the residuals (log scale) against the predicted values. Most residuals are clustered around zero, indicating that the model predictions in the log scale are fairly accurate. There are a few residuals that deviate from zero, suggesting areas where the model struggles to predict accurately. No clear pattern in the residuals suggests that the model is well-calibrated in the log scale. However, the low R² is not good.    
     

**XGBoost**      
   
![Scatterplot Actual vs Predicted Values XGBOOST](images/scatter_actual_vs_predicted_rf.png)    
Key observations:
The majority of points cluster around the diagonal line (pink dashed line), indicating that the model's predictions generally align with actual values. There's a strong linear relationship between predicted and actual values, suggesting the model captures the underlying patterns well.
Most data points are concentrated around the 0 value on both axes. The data spans from approximately -2 to 6 on both scales. Sparse data points in higher value ranges (4-6). Some outliers visible, particularly around (-2,0) and (6,0). Slight tendency to underpredict at extreme values. The sparsity of points at higher values might indicate less reliable predictions in these ranges. Generating more training data for these extreme ranges would be valuable.   
   


![Feature Importance For XGBOOST](images/feature_importance_xgboost.png)    
Key observation:    
Technical indicators ('trix' and 'roc', 'ppo', 'cmo', 'cci', 'bop') dominate the top features, suggesting strong predictive power
of technical analysis for 1h predictions.  
Time-based features (hour, day, month, year) show very low importance, suggests price movements are more technical than time-dependent.
Most cryptocurrency tickers (BTCUSDT, ETHUSDT, etc.) have minimal impact, limited cross-crypto correlation in 1h predictions.
Traditional market indicators (^SPX, ^VIX) show low importance, limited correlation with traditional markets.
Focus on technical indicators proved valuable, Fibonacci levels show surprisingly low importance across all timeframes.
   
So I considered model simplification by focusing on top 10-15 features and looked at SHAP
   
![SHAP For XGBOOST](images/shap_beeswarm_plot_xgboost.png)   

SHAP provides detailed insight into how individual feature values influence the predictions. Each dot represents a single data point, with its color indicating the feature value (blue = low, red = high). For "price_change," high values (red dots) positively push predictions, while low values (blue dots) negatively push them. Features like "cci" and "roc" exhibit a mix of positive and negative effects, showing non-linear relationships with the target variable. Features with narrow distributions of SHAP values, such as "day" or "ln_volume," have a limited effect on predictions across the dataset.

Retraining the model with the most important features surprisingly had lower scores. 

**The best model for this project right now is:**   
XGBoost MSE on the Validation Set: 2.1524  
XGBoost MAE on the Validation Set: 0.0163   
XGBoost R² Score on the Validation Set: 0.9247   
Best Hyperparameters:   
eta                 0.250000   
max_depth           5.000000   
min_child_weight    1.000000   
rmse                1.467122   

    


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

* **Jupyter Notebook (notebook.ipynb):**
  * Fetch cryptocurrency and stock market data:
    * Fetches data for cryptocurrencies and stablecoins defined in the coins list.
    * Processes the data and adds derived metrics (e.g., price change).
    * Saves the final dataset as `stable_coins.csv`.
  * Fetch hourly stock data for predefined tickers:
    * Adds derived metrics.
    * Formats timestamps.
    * Combines all stock data into a single DataFrame and saves it as a CSV.
    * Logs missing or delisted stocks/cryptos as warnings or errors.
  * Perform feature engineering and derive metrics.
  * Evaluate multiple machine learning models.
  * Save the best models as `.pkl` files.

* **Train the Model:**
  * Use `train.py` to train the best-performing model (default: XGBoost) on the processed data.
  * Save the trained model as a `.pkl` file.

* **Deploy the Model with Flask:**
  * Use `predict.py` to deploy the model and provide predictions via Flask.



    

### Flask
* The repository includes a Flask (`predict.py`) to interact with the trained XGBoost model. The API allows users to predict whether the price of USDC/USDT will grow positively within the next hour.

* **Steps to Use**
1. **Start the Flask Server**  
   Ensure the conda environment is active and run:
   ```bash python predict.py ```
  Server runs at http://0.0.0.0:8000    


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
```bash
docker build -t crypto-stock-analysis .
```

Run the container:
```bash
docker run -p 5000:5000 crypto-stock-analysis
```

    
License
This project is open-source and licensed under the MIT License.