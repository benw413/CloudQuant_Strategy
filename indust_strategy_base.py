# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK


INIT_CAP = 1000000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170301'  # backtesting end

PERIOD = 30  # the period used to calculate win/lose
UP_BAND = 0.6  # the buy signal band
DOWN_BAND = 0.25  # the sell signal band
FACTORS = ["LZ_GPA_VAL_PB",
           "LZ_GPA_FIN_IND_ARTURNDAYS",
           "LZ_GPA_FIN_IND_DEBTTOASSETS",
           "LZ_GPA_FIN_IND_OPTODEBT",
           "LZ_GPA_FIN_IND_PROFITTOGR",
           "LZ_GPA_FIN_IND_QFA_CGRGR",
           # 其他因子
           "LZ_GPA_VAL_TURN",# turn over
           "LZ_GPA_DERI_LnFloatCap"]
config = {
    'username': 'zhouyusheng2016',
    'password': '6403583love',
    'rootpath': 'd:/cStrategy/',  # client root path
    'initCapitalStock': INIT_CAP,
    'startDate': START_DATE,
    'endDate': END_DATE,
    'cycle': 1,  # backtesting freq
    'executeMode': 'D',
    'feeRate': 0.001,
    'feeLimit': 5,
    'strategyName': 'Strategy_base_selcted',  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'STOCK'
}

def initial(sdk):
    # data prepare -- industry index, industry classification ZX
    #              -- quote information
    sdk.prepareData(["LZ_GPA_CMFTR_CUM_FACTOR",
                     "LZ_GPA_QUOTE_TCLOSE",
                     "LZ_GPA_INDXQUOTE_CLOSE",
                     "LZ_GPA_INDU_ZX",
                     "LZ_GPA_SLCIND_STOP_FLAG"])
    sdk.prepareData(FACTORS)
    global dayCounter
    dayCounter = 0
def initPerDay(sdk):
    global dayCounter
    if dayCounter % PERIOD == 0:
        currentHolding = [i.code for i in sdk.getPositions()]
        buy, sell = getSignal(getURatio(sdk))
        buy.extend(sell)
        stocks = []
        for i in buy:
            stocks.extend(getStocksForIndustry(sdk, i))
        pool = list(set(stocks) | set(currentHolding))
        sdk.setGlobal("POOL", pool)
    dayCounter += 1
def getStocksForIndustry(sdk, index):
    stockCodes = sdk.getStockList()
    industry = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_INDU_ZX", 1), columns=stockCodes)
    series = industry.iloc[0]
    list = series[series == index].index.tolist()
    return list
def getSignal(array):
    buyIndex = []
    sellIndex = []
    for index, i in enumerate(array):
        if i > UP_BAND:
            buyIndex.append(index+1)
        if i < DOWN_BAND:
            sellIndex.append(index+1)
    return buyIndex, sellIndex
def getURatio(sdk):

    stockCodes = sdk.getStockList()
    # adjustment for div and event like split
    cmftr = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_CMFTR_CUM_FACTOR", PERIOD),
                         columns=stockCodes)
    # close price of all stocks
    df = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TCLOSE", PERIOD),
                      columns=stockCodes)

    # backward adjusted stock price
    dt = cmftr*df

    # find trend in last period
    start = dt.loc[dt.index[0]]  # value at period start(n days before)
    end = dt.loc[dt.index[-1]]  # value at period end(today)
    upflag = end > start  # stock whose value are increasing during last PERIOD

    upRatio = []
    for i in range(1, 30):  # each industry in ZX
        codes = getStocksForIndustry(sdk, i)
        ratio = upflag.loc[codes].sum() / float(len(codes))
        upRatio.append(ratio)
    upRatio = np.array(upRatio)
    return upRatio
def strategy(sdk):
    sdk.sdklog(sdk.getNowDate(), 'now')
    global dayCounter
    if (dayCounter - 1) % PERIOD == 0:
        # the historical state
        lastState = sdk.getGlobal("STATE")
        # the current state
        buy, sell = getSignal(getURatio(sdk))
        newState = np.zeros(29, dtype=int)
        for _, j in enumerate(buy):
            newState[j-1] = 1
        for _, j in enumerate(sell):
            newState[j-1] = -1
        #  check if buy or sell signal came in

        industryIndex = np.array(range(1, 30))
        industToBuy = industryIndex[np.logical_and(newState == 1, np.logical_or(lastState == 0, lastState == -1))]
        industToSell = industryIndex[np.logical_and(newState == -1, np.logical_or(lastState == 0, lastState == 1))]

        stockToBuy = []
        for i in industToBuy:
            stockToBuy.extend(getStocksForIndustry(sdk, i))
        stockToSell = []
        for i in industToSell:
            stockToSell.extend(getStocksForIndustry(sdk, i))
        # get the latest price
        quotes = sdk.getQuotes(sdk.getGlobal("POOL"))
        if stockToSell:
            sellAllPositionInStocks(sdk, stockToSell, quotes)
        if stockToBuy:
            stockToBuy = selectBySomeMethod(sdk, stockToBuy)
            buyStocks(sdk, stockToBuy, quotes)
        # update the state
        sdk.setGlobal("STATE", newState)
#  returns a Series of stockcodes and its optimal weight
# market risk contronl method
# use only partlly of totoal capital
# select stocks within certain industry
def selectBySomeMethod(sdk, stockToBuy):
    buy = []
    stockCodes = sdk.getStockList()
    # the history volumne weighted ave price
    df = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TCLOSE", PERIOD), columns=stockCodes)[stockToBuy]
    # the daily returns
    rts = (df / df.shift(1) - 1)
    sharpe = rts.mean() / rts.std()
    good = sharpe[sharpe > sharpe.median()]
    return good.index.tolist()
# the buy stock methods, currently buy at open
def buyStocks(sdk, stockToBuy, quotes):
    quoteStocks = quotes.keys()
    stockToBuy = list(set(stockToBuy) & set(quoteStocks))
    asset = sdk.getAccountInfo()
    # 剩余现金作为购买预算 各支股票平均分配预算
    if stockToBuy and asset:
        budget = asset.availableCash / len(stockToBuy)
        orders = []
        for buyStock in stockToBuy:
            buyPrice = quotes[buyStock].open  # 购买价格为上分钟最高价
            buyAmount = int(np.round(budget/buyPrice, -2))  # 预算除购买价格作为购入量
            if buyPrice > 0 and buyAmount >= 100:
                orders.append([buyStock, buyPrice, buyAmount, 'BUY'])  # 委托购买
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'buy')  # 将购买计入日志
# the sell method
def sellAllPositionInStocks(sdk, stockToSell, quotes):
    # 滤除取不到盘口的股票
    quoteStocks = quotes.keys()
    stockToSell = list(set(stockToSell) & set(quoteStocks))
    # 卖出
    if stockToSell:
        orders = []
        positions = sdk.getPositions()  # 查持仓
        for pos in positions:
            if pos.code in stockToSell:
                sellPrice = quotes[pos.code].open  # 设置出售价格为上分钟最低价
                sellAmount = pos.optPosition
                if sellPrice > 0 and sellAmount > 100:
                    orders.append([pos.code, sellPrice, sellAmount, 'SELL'])  # 委托出售
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'sell')  # 将出售记入日志
# currently sell at open
def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()