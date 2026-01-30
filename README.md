Netflix Stock Price Prediction using Attention-Based LSTM
Project Overview

This project focuses on building an end-to-end time series forecasting pipeline to predict Netflix (NFLX) stock prices using both traditional statistical models and advanced deep learning techniques. The goal is to compare model performance and demonstrate how Attention mechanisms improve prediction accuracy in volatile financial markets.

Dataset

Source: Real-world historical stock market data
Company: Netflix (NFLX)
Time Period: 2015 to Present
Frequency: Daily closing prices

Data Preprocessing

The following preprocessing steps were applied to enhance model learning:
Min–Max Scaling for normalization
Lag Features (previous day’s price)
Time-Based Features (day of the week)

Models Implemented

Three different modeling approaches were developed and compared:
SARIMA
Traditional statistical model
Used as a baseline for comparison
LSTM (Long Short-Term Memory)

Deep learning model capable of capturing long-term dependencies
Trained on 60-day rolling sequences
Attention-Based LSTM
Enhanced LSTM model with a custom Attention layer
Assigns importance weights to historical time steps

Results & Key Observations

SARIMA captured the general trend but struggled with sudden price fluctuations and volatility.
LSTM significantly outperformed SARIMA by learning price momentum and recent patterns, resulting in lower prediction error.

Attention-Based LSTM produced the most stable and accurate forecasts by focusing on historically important price movements while reducing the impact of noise.

Conclusion

This project demonstrates that Attention mechanisms significantly enhance the performance of Recurrent Neural Networks (RNNs) in stock price forecasting tasks.
Unlike traditional models that treat all past data equally, the Attention-based model selectively emphasized critical historical periods—such as earnings announcements—leading to more accurate and stable predictions. This confirms the effectiveness of Attention-based deep learning models for complex and volatile time series data like financial markets.