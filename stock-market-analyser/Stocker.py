# Importing necessary libraries
import pandas as pd
import plotly.offline as py
import datetime as dt
import yahoo_fin.stock_info as yf
import plotly.graph_objs as go

Markets = ("NASDAQ", "DOW", "SP500")

Market = input("Select Market Index: ")

while Market not in Markets:
    print("Please Select between DOW, NASDAQ or SP500")
    Market = input("Select Market Index: ")


if Market == "DOW":
    Market = yf.tickers_dow()
    print("Fetching DOW JONES tickers")
elif Market == "SP500":
    Market = yf.tickers_sp500()
    print("Fetching S&P500 tickers")
elif Market == "NASDAQ":
    Market = yf.tickers_nasdaq()
    print("Fetching NASDAQ tickers")
else:
    print("Market Data currently unavailabe")
    Market = input("Select Market Index: ")

Ticker = input("Select Stock: ")

if Ticker in Market:
    print("Fetching Required Data for " + str(Ticker) + " Please Follow Forward.")
if Ticker not in Market:
    print("Ticker Not Found. Please select a valid Ticker from the market")
    Ticker = input("Select Stock: ")

SDate = input("Enter a start date in YYYY-MM-DD format")
year, month, day = map(int, SDate.split("-"))
StartDate = dt.date(year, month, day)

EDate = input("Enter an end date in YYYY-MM-DD format")
year, month, day = map(int, EDate.split("-"))
EndDate = dt.date(year, month, day)

if StartDate > EndDate:
    print("Specified Range is incorrect")
    StartDate = input("Please select a start date: ")
    EndDate = input("Please mention a final date: ")
if StartDate >= dt.date.today():
    print("Specified Range is incorrect")
    StartDate = input("Please select a start date: ")
    EndDate = input("Please mention a final date: ")


def Stocker(Ticker, StartDate, EndDate):
    df = pd.DataFrame()
    StockInfo = yf.get_data(Ticker, start_date=StartDate, end_date=EndDate)
    df = df.append(StockInfo)
    return df


StockData = Stocker(Ticker, StartDate, EndDate)
StockData["FDayMA"] = StockData["close"].rolling(window=50, min_periods=0).mean()
StockData["TDayMA"] = StockData["close"].rolling(window=200, min_periods=0).mean()

print((StockData.head()))

# ----------------------------Chart 1: Time Series for Selected Stock--------------------------------------------------

df = StockData
trace_high = go.Scatter(x=list(df.index), y=list(df.high), name="High", fillcolor="red")
trace_open = go.Scatter(x=list(df.index), y=list(df.open), name="Opening Price")
trace_low = go.Scatter(
    x=list(df.index), y=list(df.adjclose), name="Closing Price", fillcolor="blue"
)
trace_200DMA = go.Scatter(x=list(df.index), y=list(df.FDayMA), name="200 Day MA")
trace_50DMA = go.Scatter(x=list(df.index), y=list(df.TDayMA), name="50 Day MA")
data = [trace_open, trace_high, trace_low, trace_200DMA, trace_50DMA]
layout = dict(
    title="Stock Performance for " + str(Ticker),
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                    dict(
                        label="Live Price (in Dollars): "
                        + str((yf.get_live_price(Ticker)))
                    ),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    ),
)

fig = dict(data=data, layout=layout)
py.plot(fig)
