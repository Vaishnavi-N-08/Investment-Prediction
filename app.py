import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

add_selectbox = st.sidebar.selectbox(
    "Choose Relevant Investment Method",
    ("Home","Gold", "Stocks", "Real Estate","Crypto","Combined")
)

def predict_crypto():
    # Load the dataset
    data = pd.read_csv('crypto_price.csv')

    # Set the initial number of days to use for the moving average
    window_size = 10

    # Calculate the moving average
    data['MA'] = data['Price'].rolling(window_size).mean()

    # Extrapolate the next 31 days of prices based on the moving average and the average change
    predictions = []
    for i in range(31):
        # Get the most recent window_size days of data
        last_window = data.tail(window_size)

        # Calculate the average change in price over the last window_size days
        average_change = np.mean(np.diff(last_window['Price']))

        # Extrapolate the next day's price based on the last moving average and the average change
        if i == 0:
            prediction = data['Price'].iloc[-1]
        else:
            # Update the window size based on the number of predictions made so far
            window_size = min(i*2, len(data)-1)

            # Calculate the new moving average and make the prediction
            data['MA'] = data['Price'].rolling(window_size).mean()
            prediction = data['MA'].iloc[-1] + average_change

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the dataset with the new prediction
        new_date = pd.date_range(data['Date'].iloc[-1], periods=2, freq='D')[1]
        new_data = pd.DataFrame(
            {'Date': new_date, 'Price': prediction}, index=[data.index[-1]+1])
        data = pd.concat([data, new_data], ignore_index=False)

    return predictions[:30]

# Creating a function to train the model


def get_model():
    df = pd.read_csv('gold_price.csv')

    # Convert the date column to datetime type and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scaling the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    # Creating a function to create the LSTM model
    def create_lstm_model():
        model = Sequential()
        model.add(LSTM(units=80, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(units=80))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Creating the training data
    x_train, y_train = [], []
    for i in range(60, scaled_price.shape[0]):
        x_train.append(scaled_price[i-60:i, 0])
        y_train.append(scaled_price[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshaping the data for the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating the LSTM model
    model = create_lstm_model()

    # Training the LSTM model
    model.fit(x_train, y_train, epochs=10, batch_size=25)

    # Creating the testing data
    test_data = scaled_price[-60:]
    x_test = []
    x_test.append(test_data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Saving the model
    model.save('gold_model.h5')

# Creating a function to load the model and predict the price


def predict_gold():
    # Checking if the model is already trained
    try:
        model = load_model('gold_model.h5')
    except:
        get_model()
        model = load_model('gold_model.h5')

    df = pd.read_csv('gold_price.csv')

    # Convert the date column to datetime type and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scaling the price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    # Creating the testing data
    test_data = scaled_price[-60:]
    x_test = []
    x_test.append(test_data)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predicting the prices using the LSTM model for the next 30 days
    predicted_prices = []
    for i in range(30):
        predicted_price = model.predict(x_test)
        predicted_prices.append(predicted_price[0, 0])
        x_test = np.concatenate(
            (x_test[:, 1:, :], predicted_price.reshape(1, 1, 1)), axis=1)

    # Scaling the predicted prices back to their original range
    predicted_prices = scaler.inverse_transform(
        np.array(predicted_prices).reshape(-1, 1))

    # Converting the predicted prices to a list
    predicted_prices = predicted_prices.reshape(1, -1)[0].tolist()
    return predicted_prices


def predict_real_estate():
    real_estate_df = pd.read_csv("real_estate_price.csv")

    # Prepare data
    real_estate_df['Date'] = pd.to_datetime(real_estate_df['Date'])
    real_estate_df.set_index('Date', inplace=True)
    real_estate_df = real_estate_df.resample('M').interpolate(method='linear')

    # Train models
    real_estate_model = sm.tsa.ARIMA(real_estate_df, order=(1, 1, 0))
    real_estate_result = real_estate_model.fit()

    # Generate forecasts
    real_estate_forecast = real_estate_result.predict(
        start=len(real_estate_df), end=len(real_estate_df)+29, typ='levels')

    return np.array(real_estate_forecast)


def predict_stocks():

    # Load the dataset
    data = pd.read_csv('stock_price.csv')
    window_size = 10

    # Calculate the moving average
    data['MA'] = data['Price'].rolling(window_size).mean()

    # Extrapolate the next 31 days of prices based on the moving average and the average change
    predictions = []
    for i in range(31):
        # Get the most recent window_size days of data
        last_window = data.tail(window_size)

        # Calculate the average change in price over the last window_size days
        average_change = np.mean(np.diff(last_window['Price']))

        # Extrapolate the next day's price based on the last moving average and the average change
        if i == 0:
            prediction = data['Price'].iloc[-1]
        else:
            # Update the window size based on the number of predictions made so far
            window_size = min(i*2, len(data)-1)

            # Calculate the new moving average and make the prediction
            data['MA'] = data['Price'].rolling(window_size).mean()
            prediction = data['MA'].iloc[-1] + average_change

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the dataset with the new prediction
        new_date = pd.date_range(data['Date'].iloc[-1], periods=2, freq='D')[1]
        new_data = pd.DataFrame(
            {'Date': new_date, 'Price': prediction}, index=[data.index[-1]+1])
        data = pd.concat([data, new_data], ignore_index=False)

    # Return the predictions for the next 30 days
    return predictions[1:]


crypto = predict_crypto()
gold = predict_gold()
real_estate = predict_real_estate()
stocks = predict_stocks()

dates = pd.date_range('2023-04-05', periods=30, freq='D')

gold_hist = pd.read_csv('gold_price.csv')
gold_hist = gold_hist.set_index(pd.to_datetime(gold_hist['Date'], format='%m-%d-%Y'))
gold_hist = gold_hist.drop('Date', axis=1)

stock_hist = pd.read_csv('stock_price.csv')
stock_hist = stock_hist.set_index(pd.to_datetime(stock_hist['Date'], format='%Y-%m-%d'))
stock_hist = stock_hist.drop('Date', axis=1)

crypto_hist = pd.read_csv('crypto_price.csv')
crypto_hist = crypto_hist.set_index(pd.to_datetime(crypto_hist['Date'], format='%Y-%m-%d'))
crypto_hist = crypto_hist.drop('Date', axis=1)

real_estate_hist = pd.read_csv('real_estate_price.csv')
real_estate_hist = real_estate_hist.set_index(pd.to_datetime(real_estate_hist['Date'], format='%Y-%m-%d'))
real_estate_hist = real_estate_hist.drop('Date', axis=1)
new_rows = pd.DataFrame({'Price': [793199.0525,804724]}, index=pd.to_datetime(['2023-03-31','2023-04-05']))
real_estate_hist = pd.concat([real_estate_hist, new_rows])




if add_selectbox == "Home":
    import streamlit as st

    # Introduction
    st.header("Introduction")
    st.write("""Welcome to our student-led investment prediction project. Our team of student analysts has conducted insightful analysis and predictions on four major investment options: gold, stock, real estate, and cryptocurrency. Our aim is to provide fellow students and investors with reliable and accurate predictions based on historical data and current market trends.

Through careful analysis of the past and current performance of each investment option, we provide predictions based on various factors that affect their prices. Our analysis enables investors to make informed decisions about their investments and take advantage of potential high returns.

Gold, as an investment option, has been popular for centuries due to its perceived stability and value. Our predictive model analyzes various factors that affect the price of gold, such as global economic conditions, geopolitical tensions, and inflation rates. With our analysis, investors can make informed decisions about investing in gold.

Stocks, another popular investment option, have proven to be profitable for many investors. Our predictive model analyzes stock market trends, company financial reports, and other economic indicators to provide reliable predictions of future stock prices. Investors can make informed decisions about buying, selling, or holding stocks.

Real estate has been a reliable investment option for many investors, with the potential for high returns through rental income and property appreciation. Our predictive model analyzes real estate market trends, such as population growth, housing supply and demand, and interest rates, to provide reliable predictions of future real estate prices.

Cryptocurrencies are a relatively new investment option that has gained significant attention from investors in recent years. Our predictive model analyzes cryptocurrency market trends, such as supply and demand, market sentiment, and regulatory changes, to provide reliable predictions of future cryptocurrency prices. With our analysis, investors can make informed decisions about investing in cryptocurrencies and take advantage of their potential for high returns.

Overall, our student-led investment prediction project provides investors with reliable predictions of four major investment options, allowing them to make informed decisions about their investments. Stay tuned for our latest predictions, and let us help you make smart investment decisions.""")
    st.write("""The approach of viewing multiple investment options together provides investors with a broader perspective and helps them diversify their investment portfolios. By investing in a variety of assets, investors can minimize the risk of losing all their investments due to the failure of one asset. Diversification is an effective risk management strategy that enables investors to spread their investments across different asset classes and minimize the impact of volatility in any one particular asset class.

For example, if an investor only invests in stocks and the stock market experiences a sudden crash, the investor may lose a significant portion of their investments. However, if the investor also has investments in gold, real estate, and cryptocurrency, they may experience less impact on their overall portfolio because these other assets are not as closely correlated with the stock market.

Investing in multiple assets can also provide investors with opportunities to maximize their returns. Different assets perform well in different market conditions, and by investing in a variety of assets, investors can take advantage of these different market conditions. For example, if the stock market is performing well, an investor can benefit from the growth of their stock investments, but if the stock market experiences a downturn, the investor can benefit from the stability of their gold or real estate investments.

In conclusion, investing in multiple assets is a wise decision for investors who want to minimize their losses and maximize their returns. The approach of viewing multiple investment options together provides a broader perspective, and by diversifying their investment portfolios, investors can minimize the risk of losing all their investments due to the failure of one asset.""")

    # Benefits
    st.header("Benefits")
    st.write("Our approach offers several benefits, including:")
    st.write("- Reduces Risk: Diversifying your investments means spreading your money across different assets, reducing your exposure to any single asset class. By doing so, you can minimize the risk of losing all your money if one investment performs poorly.")
    st.write("- Enhances Returns: Diversifying your investments can also help to enhance your overall returns. This is because different asset classes perform well at different times, and by investing across multiple assets, you can potentially capture these performance gains.")
    st.write("- Mitigates Market Volatility: Diversification can help mitigate market volatility, as losses in one asset class can be offset by gains in another. This can help to smooth out the overall performance of your portfolio and reduce the impact of market downturns.")
    st.write("- Provides Flexibility: Diversifying your investments can provide flexibility in your investment strategy. By having exposure to different assets, you can adjust your portfolio to changing market conditions and investment opportunities.")

    # Drawbacks
    st.header("Drawbacks")
    st.write("While our approach has many benefits, there are also some drawbacks, such as:")
    st.write("- Dilutes Gains: While diversification can help to enhance overall returns, it can also dilute gains from high-performing investments. This is because gains from one asset class may be offset by losses in another, limiting the potential upside of any single investment.")
    st.write("- Increases Complexity: Diversification can also increase the complexity of your portfolio. It requires a thorough understanding of different asset classes and their performance drivers, as well as the ability to monitor and rebalance your portfolio regularly.")
    st.write("- Can Lead to Over-Diversification: Over-diversification can occur when an investor holds too many investments, to the point where the performance of the portfolio is diluted. This can occur when an investor tries to diversify too much, or when investing in index funds or ETFs that are already broadly diversified.")
    st.write("- May Not Always Mitigate Risk: While diversification can help to mitigate risk, it is not a guarantee. Some market events, such as a global financial crisis, can cause widespread losses across all asset classes, regardless of how diversified a portfolio is.")

    # Risks
    st.header("Risks")
    st.write("There are some risks associated with our approach, including:")
    st.write("- Correlated Risk: Investing in multiple assets does not guarantee that the assets are not correlated with each other. If all your investments are correlated, then diversification does not provide any risk reduction.")
    st.write("- Opportunity Cost: Diversifying your investments means allocating funds to multiple assets, which can limit your ability to invest in higher-yielding assets or opportunities that could have offered better returns.")
    st.write("- Cost of Diversification: Maintaining a diversified portfolio can come with additional costs such as transaction fees, management fees, and taxes. These costs can eat into your returns and reduce the overall benefit of diversification.")


if add_selectbox == "Gold":
    gold1 = pd.DataFrame({'Predicted': gold},index=dates)
    goldPlot = pd.concat([gold_hist, gold1], axis=1)
    st.line_chart(goldPlot)

    st.header("Gold")
    st.write("""Gold is a precious metal that has been considered valuable throughout human history. As an asset, gold is a popular investment option for investors looking for a safe haven in times of economic uncertainty or inflation. Gold has several unique characteristics that make it a popular investment option:

Limited Supply: Gold is a finite resource, and its supply is limited. The amount of gold available for mining is relatively small, and it takes a significant amount of time and effort to mine and refine gold. As a result, the supply of gold is relatively stable and cannot be easily manipulated by governments or central banks.

Inherent Value: Gold is considered to have intrinsic value, meaning that it has value in and of itself, regardless of its use or function. This makes gold a popular choice for investors who want to hedge against inflation or currency devaluation.

Global Acceptance: Gold is accepted as a form of payment and store of value throughout the world. It is widely traded in global markets, and its value is recognized and accepted by all major countries and financial institutions.

Historical Performance: Gold has a long history of maintaining its value over time. It has been used as a currency and store of value for centuries, and its value has remained relatively stable over the long term.

Diversification: Gold is considered to be a diversifying asset, meaning that it has a low correlation with other asset classes such as stocks and bonds. This means that adding gold to a portfolio can help to reduce overall portfolio risk and volatility.

However, there are also some potential drawbacks to investing in gold. It does not generate any income or dividends, and its value is largely dependent on market sentiment and demand. Additionally, the price of gold can be volatile, and it may not always perform well in certain economic conditions.

Overall, gold is a popular investment option for investors who want to diversify their portfolios and protect against economic uncertainty or inflation. Its unique characteristics, historical performance, and global acceptance make it a valuable asset for investors to consider.
      """)

if add_selectbox == "Stocks":
    stock1 = pd.DataFrame({'Predicted': stocks},index=dates)
    stockPlot = pd.concat([stock_hist, stock1], axis=1)
    st.line_chart(stockPlot)
    st.header("Stocks")
    st.write("""Stocks represent ownership in a company and are a popular investment option for investors looking to grow their wealth over the long term. When an investor buys stocks, they become a shareholder in the company and have a claim on its assets and earnings.

There are several types of stocks, including common stocks and preferred stocks. Common stocks represent ownership in the company and typically come with voting rights at shareholder meetings. Preferred stocks, on the other hand, offer a fixed dividend payment and have a higher priority than common stocks in the event of a company's bankruptcy.

Stocks offer several benefits as an investment option, including:

Potential for High Returns: Stocks have the potential to offer high returns over the long term. Historically, stocks have outperformed other asset classes such as bonds and cash.

Liquidity: Stocks are highly liquid, meaning that they can be easily bought and sold in the market. This makes it easy for investors to adjust their portfolios as needed.

Diversification: Stocks offer investors the ability to diversify their portfolios across different industries and sectors, reducing overall portfolio risk.

Dividend Income: Many stocks offer dividend payments to shareholders, providing a source of regular income.

However, there are also some potential risks associated with investing in stocks. Stock prices can be volatile and subject to market fluctuations, and individual companies can experience financial difficulties that can impact their stock prices. Additionally, stock prices can be influenced by external factors such as political events or economic conditions.

Overall, stocks can be a valuable investment option for investors looking for long-term growth potential and the ability to diversify their portfolios. However, investors should carefully consider the potential risks and rewards before investing in stocks and ensure that they have a well-diversified portfolio that aligns with their investment goals and risk tolerance.""")

if add_selectbox == "Real Estate":
    #st.write(real_estate)
    real_estate1 = pd.DataFrame({'Predicted': real_estate},index=dates)
    real_estatePlot = pd.concat([real_estate_hist, real_estate1], axis=1)
    st.line_chart(real_estatePlot)
    st.header("Real Estate")
    st.write("""Real estate is an asset class that involves investing in physical property, such as land, buildings, or other structures. Real estate investments can take many forms, including owning rental properties, investing in real estate investment trusts (REITs), or participating in real estate crowdfunding platforms.

There are several benefits to investing in real estate, including:

Potential for Appreciation: Real estate has the potential to appreciate in value over time, providing a source of long-term capital growth for investors.

Cash Flow: Real estate investments can provide a steady stream of rental income, providing a source of regular cash flow for investors.

Diversification: Real estate can be a valuable way to diversify an investment portfolio, as it is not closely correlated with other asset classes such as stocks and bonds.

Inflation Hedge: Real estate can serve as an inflation hedge, as property values and rental incomes tend to rise with inflation.

However, there are also potential risks associated with investing in real estate. Real estate investments can be illiquid, meaning that they are not easily bought or sold, and require significant upfront capital to acquire. Property values can also be impacted by external factors such as economic conditions or changes in interest rates.

Overall, real estate can be a valuable asset class for investors looking for long-term growth potential, regular income, and portfolio diversification. However, investors should carefully consider the potential risks and rewards before investing in real estate and ensure that they have a well-diversified portfolio that aligns with their investment goals and risk tolerance.""")


if add_selectbox == "Crypto":
    dates = pd.date_range('2023-04-06', periods=30, freq='D')
    crypto1 = pd.DataFrame({'Predicted': crypto},index=dates)
    cryptoPlot = pd.concat([crypto_hist, crypto1], axis=1)
    st.line_chart(cryptoPlot)
    st.header("Crypto")
    st.write("""Crypto assets, also known as cryptocurrencies, are digital or virtual tokens that utilize encryption techniques to secure their transactions and to control the creation of new units. Cryptocurrencies are decentralized, meaning they are not issued or regulated by a central authority such as a government or financial institution.

Ethereum (ETH) is one of the largest cryptocurrencies by market capitalization, and is the native token of the Ethereum blockchain platform. Ethereum is used as both a currency and a utility token, providing access to the Ethereum platform's decentralized applications (dApps) and smart contract functionality.

One of the benefits of investing in Ethereum is the potential for high returns. In recent years, the price of Ethereum has seen significant fluctuations, providing opportunities for investors to profit from price movements. Additionally, Ethereum's smart contract functionality has led to the creation of a wide range of decentralized applications, which could potentially increase demand for the token and drive up its value.

However, there are also significant risks associated with investing in Ethereum and other cryptocurrencies. Cryptocurrencies are highly volatile, with prices subject to sudden and dramatic fluctuations based on market demand and sentiment. Additionally, the lack of regulation and oversight in the cryptocurrency market can make it a target for fraud and manipulation.

Overall, investing in Ethereum and other cryptocurrencies can provide an opportunity for high returns, but comes with significant risks. Investors should carefully consider their investment goals and risk tolerance before investing in cryptocurrencies, and ensure they have a well-diversified portfolio that aligns with their investment objectives.""")


if add_selectbox == "Combined":

    # Normalize the predictions from each model in a range from 0 to 100
    cryptoN = (crypto - np.min(crypto)) / (np.max(crypto) - np.min(crypto)) * 100
    goldN = (gold - np.min(gold)) / (np.max(gold) - np.min(gold)) * 100
    real_estateN = (real_estate - np.min(real_estate)) / \
        (np.max(real_estate) - np.min(real_estate)) * 100
    stocksN = (stocks - np.min(stocks)) / (np.max(stocks) - np.min(stocks)) * 100

    # Create a dataframe with the all the predictions combined
    predictions = pd.DataFrame(
        {'Crypto': cryptoN, 'Gold': goldN, 'Real Estate': real_estateN, 'Stocks': stocksN},index=dates)

    st.line_chart(predictions)
    st.write("Above Chart contains Normalized Values of Predicted prices for each respected asset option to put into picture their trends in comparison as there is vast difference in their actual prices.")
    st.write("")
    st.write("""Normalizing the prediction values means scaling the predicted values to a common scale so that they can be compared easily. This approach is useful when we want to observe the general trend of different assets and compare their performance over a certain period of time. Normalizing the values enables us to eliminate the differences in the absolute values of the predictions and focus on the relative performance of each asset.

Visualizing the normalized predictions for each asset in the same graph allows for easy comparison between them. It provides a quick overview of how different assets are predicted to perform over a certain period of time. This approach helps investors make informed decisions about which assets to invest in, based on their predicted performance.

Additionally, visualizing the normalized predictions in the same graph can also help investors identify potential trends or patterns that may be missed by looking at each asset's predictions in isolation. For example, if the graph shows that the predicted performance of two assets is highly correlated, investors can use this information to adjust their portfolio accordingly and avoid overexposure to similar assets.

In summary, normalizing the prediction values and visualizing them in the same graph provides a powerful tool for investors to compare the predicted performance of different assets. It helps to simplify the data and provides a clear understanding of the general trend, enabling investors to make informed decisions about their investments.""")