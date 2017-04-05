# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK


INIT_CAP = 1000000000  # init capital
START_DATE = '20100101'  # backtesting start
END_DATE = '20170301'  # backtesting end

PERIOD = 30  # the period used to calculate win/lose
FACTORS = ["LZ_GPA_VAL_PB",
           "LZ_GPA_FIN_IND_ARTURNDAYS",
           "LZ_GPA_FIN_IND_DEBTTOASSETS",
           "LZ_GPA_FIN_IND_OPTODEBT",
           "LZ_GPA_FIN_IND_PROFITTOGR",
           "LZ_GPA_FIN_IND_QFA_CGRGR",
           # 其他因子
           "LZ_GPA_VAL_TURN",# turn over
           "LZ_GPA_VAL_A_TCAP"]
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
    'strategyName': 'industM_stockT_strategy',  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'STOCK'
}
global mt, up, down
def initial(sdk):
    global dayCounter
    dayCounter = 0
    stockCode = ["000001", "000002"]
    if sdk.getGlobal("MT") is None:
        mtdf = pd.DataFrame(data=np.zeros((1, len(stockCode))), columns=stockCode)
        up_bands = pd.DataFrame(data=np.zeros((1, len(stockCode))), columns=stockCode)
        down_bands = pd.DataFrame(data=np.zeros((1, len(stockCode))), columns=stockCode)
        sdk.setGlobal("MT", mtdf)
        sdk.setGlobal("UP_BAND", up_bands)
        sdk.setGlobal("DOWN_BAND", down_bands)
    if sdk.getGlobal("HM") is None:
        sdk.setGlobal("HM", set())
    if sdk.getGlobal("HT") is None:
        sdk.setGlobal("HT", set())
def initPerDay(sdk):
    global dayCounter
    dayCounter += 1
def updateGlobalMt(sdk, series):
    mt = sdk.getGlobal("MT")
    # dealing with the case a new stock come into stockCodeList
    mt.loc[len(mt.index)] = series
    sdk.setGlobal("MT", mt)
def updateBands(sdk, up, down):
    up_b = sdk.getGlobal("UP_BAND")
    down_b = sdk.getGlobal("DOWN_BAND")
    up_b.loc[len(up_b)] = up
    down_b .loc[len(down_b)] = down
    sdk.setGlobal("UP_BAND", up_b)
    sdk.setGlobal("DOWN_BAND", down_b)
def strategy(sdk):
    global dayCounter
    updateMtPeriod = 5
    UP_BAND = 0.2  # the initial up band
    DOWN_BAND = -0.2  # the initial down band
    stdObsers = 20  # the param
    stockCode = ["000001", "000002"]
    latestMt = sdk.getGlobal("MT").loc[sdk.getGlobal("MT").index[-1]]  # yesterday' signal
    # set stock individualized standard
    if len(sdk.getGlobal("MT")) > stdObsers:
        recentMts = sdk.getGlobal("MT").tail(stdObsers)
        mtStds = recentMts.std()
        mtMeans = recentMts.mean()
        UP_BAND = mtMeans + mtStds
        DOWN_BAND = mtMeans - mtStds
    # find stocks with Hight M and High T
    highMStocks = set(latestMt[latestMt > UP_BAND].index.tolist())
    highTStocks = set(latestMt[latestMt < DOWN_BAND].index.tolist())

    ''' if the stock lose its position in HM or HT, we should empty our position on this stock'''
    lastHighMStocks = sdk.getGlobal("HM")
    lastHighTStocks = sdk.getGlobal("HT")

    """classify the stocks"""
    newInHM = []
    leaveHM = []
    newInHT = []
    leaveHT = []
    if highMStocks - lastHighMStocks:
        newInHM = list(highMStocks - lastHighMStocks)
    elif lastHighMStocks - highMStocks:
        leaveHM = list(lastHighMStocks - highMStocks)
    elif highTStocks - lastHighTStocks:
        newInHT = list(highTStocks - lastHighTStocks)
    elif lastHighTStocks - highTStocks:
        leaveHM = list(lastHighTStocks - highTStocks)
    """check its recent behavior, and operate on it"""
    stockToBuy = []
    stockToSell = []
    # sell the leaving stock
    stockToSell.extend(leaveHM)
    stockToSell.extend(leaveHT)

    if newInHM:
        buy_1, sell_1 = checkLastPeriodPerformance(sdk, newInHM, updateMtPeriod)
        stockToSell.extend(sell_1)
        stockToBuy.extend(buy_1)
    if newInHT:
        sell_2, buy_2 = checkLastPeriodPerformance(sdk, newInHT, updateMtPeriod)
        stockToSell.extend(sell_2)
        stockToBuy.extend(buy_2)

    if stockToBuy or stockToSell:
        quotes = sdk.getQuotes(stockCode)
        if stockToBuy:
            buyStocks(sdk, stockToBuy, quotes)
        if stockToSell:
            sellAllPositionInStocks(sdk, stockToSell, quotes)

    if (dayCounter - 1) % updateMtPeriod == 0:
        # update the MT signal to todays price, should be done at end of strategy
        """stock pool must be filtered, since there may stop stocks in current holdings"""
        mtseries = stockBMT(sdk, stockCode, 20, updateMtPeriod)
        updateGlobalMt(sdk, mtseries)
        """set the new Hight M and Hight T stocks"""
        sdk.setGlobal("HM", highMStocks)
        sdk.setGlobal("HT", highTStocks)
        updateBands(sdk, UP_BAND, DOWN_BAND)
        global mt, up, down
        mt = sdk.getGlobal("MT")
        up = sdk.getGlobal("UP_BAND")
        down = sdk.getGlobal("DOWN_BAND")
def checkLastPeriodPerformance(sdk, stockCodes, period):
    priceAdjdf = pd.DataFrame(columns=stockCodes)
    for stock in stockCodes:
        priceadj = {i: item.close * item.factor
                    for i, item in enumerate(sdk.getLatest(code=stock, count=period, timefreq="1D"))}
        stockPriceAdj = pd.Series(data=priceadj.values(), index=priceadj.keys())
        priceAdjdf[stock] = stockPriceAdj
    p = priceAdjdf
    rts = p / p.shift(1) - 1
    rts.drop(rts.index[0], inplace=True)  # drop the first line NaN value of returns
    win = rts.mean() > 0
    lose = rts.mean() < 0
    return win[win].index.tolist(), lose[lose].index.tolist()
# compute the signal
# measures the effect of momentum, gives todays mt signal in a dataframe
def stockBMT(sdk, stockCodes, numOfObservatios, period=1):
    priceAdjdf = pd.DataFrame(columns=stockCodes)
    for stock in stockCodes:
        priceadj = {i: item.close * item.factor
                    for i, item in enumerate(sdk.getLatest(code=stock, count=period*numOfObservatios, timefreq="1D"))}
        stockPriceAdj = pd.Series(data=priceadj.values(), index=priceadj.keys())
        priceAdjdf[stock] = stockPriceAdj
    p = priceAdjdf.loc[::-period].loc[::-1]
    rts = p / p.shift(1) - 1
    rts.drop(rts.index[0], inplace=True)  # drop the NaN value of returns
    cluster = rts > 0  # the binary state of rts
    mt = cluster.apply(lambda x: binaryDet(x))
    return mt
# count the number of given state stays the same
def binaryDet(series):
    states = [False, True]
    same1 = 0
    same2 = 0
    transit1 = 0
    transit2 = 0
    for i, state in enumerate(series):
        if i == 0:
            continue
            # omit the first state
        if series.iloc[i-1] == states[0]:
            if state == states[0]:
                same1 += 1
            else:
                transit1 += 1
        if series.iloc[i-1] == states[1]:
            if state == states[1]:
                same2 += 1
            else:
                transit2 += 1

    if same1 + transit1 == 0:
        return 1  # the case all state are True, perfect pos momentum, 1 as perfect momentum
    else:
        p = float(same1) / float(same1 + transit1)
    if same2 + transit2 == 0:
        return 1  # the case all state are False, perfect neg momentum, 1 as perfect momentum
    else:
        q = float(same2) / float(same2 + transit2)
    return p*q-(1-p)*(1-q)
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
def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()

mt[mt.columns[0]].plot()
up[up.columns[0]].plot()
down[down.columns[0]].plot()

mt[mt.columns[1]].plot()
up[up.columns[1]].plot()
down[down.columns[1]].plot()
