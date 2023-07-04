# Time-Series Data Analysis
- Statistical model: 
    - ARIMA(AutoRegressive Integrated Moving Average), SARIMA (Seasonal ARIMA), 
        - [YouTube](https://www.youtube.com/watch?v=8FCDpFhd1zk)
        - [Time Series Analysis Using ARIMA From Statsmodels](https://www.nbshare.io/notebook/136553745/Time-Series-Analysis-Using-ARIMA-From-StatsModels/)
        - [Forecasting web traffic with machine learning and Python](https://www.cienciadedatos.net/documentos/py37-forecasting-web-traffic-machine-learning#Tuning-the-hyperparameters)
- Regression Algorithm: 
    - XGBoost
        - [How to Select a Model For Your Time Series Prediction Task [Guide]](https://neptune.ai/blog/select-model-for-time-series-prediction-task)
        - [Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM, Scikit-learn y CatBoost](https://www.cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost)

## Bike Sharing Forecasting
- evaluation index: RMSE (Root Mean Square Error)
### Libraries
- numpy
- pandas
- matplotlib
- statsmodels
- pmdarima
- xgboost
- sklearn
- skforecast
### Data Preparation
- ARIMA/SARIMA: day.csv
- XGBoost: hour.csv
- exongenous features: weekday, weather, holiday
### Model Results
- Statistical model
    - ARIMA: 
        - order = (1, 1, 1) with auto_arima
        - AIC: 9927.707, RMSE: 845.579
    - SARIMA: 
        - orser = (1, 1, 1), (1, 1, 1, 14) by grid search
        - AIC: 9743.914, RMSE: 855.621
    - SARIMA with exogenous features:
        - orser = (1, 1, 1), (1, 1, 1, 14)
        - AIC: 9582.199, RMSE: 741.800
- Regression Algorithm: 
    - XGBoost with backtesting:
        - parameter: 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500 by grid_search_forecaster
        - RMSE: 100.357
    - XGBoost with backtesting and exogenous features:
        - parameter: 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500 by grid_search_forecaster
        - RMSE: 83.086