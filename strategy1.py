# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import cvxopt as cvx
from CloudQuant import MiniSimulator  # 导入云宽客SDK
from joblib import Parallel, delayed

INIT_CAP = 80000000  # init capital
START_DATE = '20090101'  # backtesting start
END_DATE = '20091201'  # backtesting end

PERIOD = 20  # the period used to calculate win/lose
UP_BAND = 0.6  # the buy signal band
DOWN_BAND = 0.3  # the sell signal band

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
    'strategyName': 'Strategy_1',  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'FUND'
}

def initial(sdk):
    # data prepare -- industry index, industry classification ZX
    #              -- quote information
    sdk.prepareData(["LZ_GPA_CMFTR_CUM_FACTOR",
                     "LZ_GPA_QUOTE_TCLOSE",
                     "LZ_GPA_INDXQUOTE_CLOSE",
                     "LZ_GPA_INDU_ZX"])
    global dayCounter
    dayCounter = 0
def initPerDay(sdk):
    global dayCounter
    if dayCounter % PERIOD == 0:
        currentHolding = [i.code for i in sdk.getPositions()]
        print(currentHolding)
        buy, sell = getSignal(getURatio(sdk))
        buy.extend(sell)
        stocks = []
        for i in buy:
            stocks.extend(getStocksForIndustry(sdk, i))
        pool = list(set(stocks) | set(currentHolding))
        sdk.setGlobal("POOL", pool)

    if sdk.getGlobal("STATE") is None:
        sdk.setGlobal("STATE", np.zeros(29, dtype=int))

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
        industToBuy = industryIndex[np.logical_and(newState == 1, lastState != 1)]
        industToSell = industryIndex[np.logical_and(newState == -1, lastState != -1)]

        stockToBuy = []
        for i in industToBuy:
            stockToBuy.extend(getStocksForIndustry(sdk, i))
        stockToSell = []
        for i in industToSell:
            stockToSell.extend(getStocksForIndustry(sdk, i))

        # get the latest price
        quotes = sdk.getQuotes(sdk.getGlobal("POOL"))

        if stockToSell:
            sellStocks(sdk, stockToSell, quotes)

        # set optimal weight to in position aseests
        currentHolding = [i.code for i in sdk.getPositions()]
        # intend to hold these stocks
        #intend = list(set(currentHolding) | set(stockToBuy))
        #optWeight = getOptWeight(intend)

        if stockToBuy:
            buyStocks(sdk, stockToBuy, quotes)

        # update the state
        sdk.setGlobal("STATE", newState)
def getOptWeight(sdk, stockCodeList, FactorNames, exposurePeriod):
    ###        important notes                ###
    # Factor  files should be prepared in init()#
    # sdk.prepare(FactorNames)                  #
    #############################################
    stockCodes = sdk.getStockList()
    # get stock returns
    rts = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TCLOSE",
                                             (exposurePeriod+1)*PERIOD), columns=stockCodes)[stockCodeList]
    trials = range(0, len(rts), PERIOD)
    rts = rts.iloc[trials]
    rts = (rts / (rts.shift(1)) - 1).shift(-1).drop(rts.index[-1])# cacl the returns and drop the Nan row
    ### rts    rt 1 step    |rt 2 step .shift(1) |  rts 3 step.shift(-1) | rts 4 step .drop()
    #    days before  today |  factor Value               factor Value   |    factor Value          rt at time
    # 0       (P+1)         |       NaN                      v1          |         v1                P P-days before rt
    # 1         P           |       v1                       v2          |         v2                P-1
    # 2         .           |       v2                        .          |          .
    # 3         .           |       .                         .          |          .
    # .         .           |       .                         .          |　        .
    # P-1       2           |      vp-1                      vp          |          vp               1  yesterday rt
    # P        (1)          |       vp                       NaN         |        droped
    ####
    factorDataList = []
    for name in FactorNames:
        dt = pd.DataFrame(data=sdk.getFieldData(name, (exposurePeriod+1)*PERIOD), columns=stockCodes).iloc[trials]
        dt.drop(dt.index[-1])
        # save only stocks we are interested in
        factorDataList.append(dt[stockCodeList])
    ### factordatalist elment dataframe dt
    #  days before today  | factor value    |
    # 0         P+1       |     v0          |
    # 1         P         |     v1          |
    # .         .         |      .          |
    # .         .         |      .          |
    # P-1       2         |     vp-1        |
    # P         1         |     droped      |

    # 1. regression on time series to find factor exposure
    # construct the factor df for each stock
    stockDataList = []
    for stockCode in stockCodeList:
        stockFactor = pd.DataFrame(columns=FactorNames)
        for i, factor in enumerate(factorDataList):
            stockFactor[FactorNames[i]] = factor[stockCode]# this is a series
        stockDataList.append(stockFactor)
    # construct the exposure for each stock
    stockExposure = pd.DataFrame(index=FactorNames, columns=stockCodeList)

    # cacl exposure of individual stock to factor
    for i, stockCode in enumerate(stockCodeList):
        rts[stockCode]  # series of stock returns  : nx1
        stockDataList[i]  # index = time , columns = factors  : nxk

        # use numpy to calc exposure


















def buyStocks(sdk, stockToBuy, quotes):
    quoteStocks = quotes.keys()
    stockToBuy = list(set(stockToBuy) & set(quoteStocks))
    asset = sdk.getAccountInfo()
    # 剩余现金作为购买预算 各支股票平均分配预算
    if stockToBuy and asset:
        budget = asset.availableCash * 0.1 / len(stockToBuy)
        orders = []
        for buyStock in stockToBuy:
            buyPrice = quotes[buyStock].high  # 购买价格为上分钟最高价
            buyAmount = int(np.round(budget/buyPrice, -2))  # 预算除购买价格作为购入量
            if buyPrice > 0 and buyAmount >= 100:
                orders.append([buyStock, buyPrice, buyAmount, 'BUY'])  # 委托购买
        if orders:
            print(orders)
            sdk.makeOrders([orders[1]])
            sdk.sdklog(orders, 'buy')  # 将购买计入日志
# the sell method
def sellStocks(sdk, stockToSell, quotes):
    # 滤除取不到盘口的股票
    quoteStocks = quotes.keys()
    stockToSell = list(set(stockToSell) & set(quoteStocks))
    # 卖出
    if stockToSell:
        orders = []
        positions = sdk.getPositions()  # 查持仓
        for pos in positions:
            if pos.code in stockToSell:
                sellPrice = quotes[pos.code].low  # 设置出售价格为上分钟最低价
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
