# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK

"""this is the stock mt method using mean+std as band, and momentum only.
the method shows potential to be future studied"""
INIT_CAP = 1000000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170301'  # backtesting end

PERIOD = 30  # the period used to calculate win/lose
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
    'strategyName': 'stock_mt_monly_corr_strategy',  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'STOCK'
}
"""frame work methods"""
def initial(sdk):
    global dayCounter
    dayCounter = 0
    # data prepare -- industry index, industry classification ZX
    #              -- quote information
    sdk.prepareData(["LZ_GPA_QUOTE_TCLOSE",
                     "LZ_GPA_INDXQUOTE_CLOSE",
                     "LZ_GPA_INDU_ZX",
                     "LZ_GPA_VAL_A_TCAP",  # trading capital
                     "LZ_GPA_QUOTE_TVOLUME",  # trading volume in hands
                     "LZ_GPA_SLCIND_STOP_FLAG",  # stop trading
                     "LZ_GPA_SLCIND_ST_FLAG"])  # s.t. stocks
def initPerDay(sdk):
    stockCodeList = sdk.getStockList()
    stockCodeListFiltered1 = removeSTStocks(sdk, stockCodeList, 2*PERIOD)
    stockCodeListFiletered2 = removeIlliquidStocks(sdk, stockCodeListFiltered1, 2*PERIOD, 0.3)
    #stockCodeListFiletered3 = removeSmallCapStocks(sdk, stockCodeListFiletered2, 2*PERIOD, 0.5)
    stockCodeListFiletered4 = removeTodayInvalidStocks(sdk, stockCodeListFiletered2)


    currentStock = [i.code for i in sdk.getPositions()]
    sdk.setGlobal("Holdings", currentStock)
    pool = list(set(stockCodeListFiletered4) | set(currentStock))
    sdk.setGlobal("Pool", pool)

    # initialized the global parameter
    if sdk.getGlobal("DELAYSELLS") is None:
        sdk.setGlobal("DELAYSELLS", set())
    if sdk.getGlobal("MT") is None:
        stockCodeList = sdk.getStockList()
        mtdf = pd.DataFrame(data=np.zeros((1, len(stockCodeList))), columns=stockCodeList)
        sdk.setGlobal("MT", mtdf)
    if sdk.getGlobal("HM") is None:
        sdk.setGlobal("HM", set())
    global dayCounter
    dayCounter += 1
def strategy(sdk):
    global dayCounter
    updateMtPeriod = 1
    '''corrected'''
    UP_BAND = 0.2  # the initial up band
    stdObsers = 20  # the param
    # since we update signal using Close price, at the beginning of each trading day, we can only use
    # yesterday's signal as the latest signal
    if (dayCounter - 1) % updateMtPeriod == 0:
        latestMt = sdk.getGlobal("MT").loc[sdk.getGlobal("MT").index[-1]]  # yesterday' signal
        # set stock individualized standard
        if len(sdk.getGlobal("MT")) > stdObsers:
            recentMts = sdk.getGlobal("MT").tail(stdObsers)
            mtStds = recentMts.std()
            mtMeans = recentMts.mean()
            UP_BAND = mtMeans+mtStds

        # find stocks with Hight M and High T
        highMStocks = set(latestMt[latestMt > UP_BAND].index.tolist())
        ''' if the stock lose its position in HM or HT, we should empty our position on this stock'''
        lastHighMStocks = sdk.getGlobal("HM")
        """classify the stocks"""
        newInHM = []
        leaveHM = []
        if highMStocks - lastHighMStocks:
            newInHM = list(highMStocks - lastHighMStocks)
        if lastHighMStocks - highMStocks:
            leaveHM = list(lastHighMStocks - highMStocks)
        """check its recent behavior, and operate on it"""
        stockToBuy = []
        stockToSell = []
        # sell the leaving stock
        stockToSell.extend(leaveHM)
        if newInHM:
            buy_1, sell_1 = checkLastPeriodPerformance(sdk, newInHM, 5)
            stockToSell.extend(sell_1)
            stockToBuy.extend(buy_1)
        '''sell the delyaed sell stocks first, this section must come after def stockToBuy and stockToSell'''
        delayed = sdk.getGlobal("DELAYSELLS")  # is a set
        sellandUpdateDelayed(sdk, stockToBuy, stockToSell, list(delayed))
        '''buy sell stocks with signal'''
        if stockToBuy or stockToSell:
            quotes = sdk.getQuotes(sdk.getGlobal("Pool"))
            if stockToSell:
                unsuccessful = sellStockAndShowUnsuccess(sdk, stockToSell, quotes)
                sdk.setGlobal("DELAYSELLS", sdk.getGlobal("DELAYSELLS") | set(unsuccessful))
            if stockToBuy:
                buyStocks(sdk, stockToBuy, quotes)


        """update the signal"""
        # update the MT signal to todays price, should be done at end of strategy
        # using last 20 observations
        mtseries = stockBMT(sdk, sdk.getGlobal("Pool"), 20, updateMtPeriod)
        updateGlobalMt(sdk, mtseries)
        sdk.setGlobal("HM", highMStocks)
"""stock filtering methods"""
def removeSTStocks(sdk, stockCodes, period):
    stockCodeList = sdk.getStockList()
    st = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_SLCIND_ST_FLAG", period),
                      columns=stockCodeList)[stockCodes]
    # if st and stop the value should be 1, NaN present normally traded
    f2 = st.isnull().any()
    # list of stockcodes that have all nan in last period
    l2 = f2.index[f2].tolist()
    return l2
def removeStopStocks(sdk, stockCodes, period):
    # stop trading stocks
    list1 = []
    for stock in stockCodes:
        closeList = [i.close for i in sdk.getLatest(code=stock, count=period + 1, timefreq="1D")]
        if len(closeList) == period+1:
            list1.append(stock)
    return list1
def removeTodayInvalidStocks(sdk, stockCodes):
    dict1 = {i.code: i.open for i in sdk.getLatest(code=stockCodes, timefreq="1D")}
    dict2 = {i.code: i.close for i in sdk.getLatest(code=stockCodes, timefreq="1D")}
    valid = set(dict1.keys()) & set(dict2.keys())
    return list(valid)
def removeIlliquidStocks(sdk, stockCodes, period, quantile):
    # returns the stockcodes with larger the quantile trading volumes
    stockCodeList = sdk.getStockList()
    liquid = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TVOLUME", period),
                          columns=stockCodeList)[stockCodes]
    v = liquid.mean().quantile(quantile)
    return liquid.columns[liquid.mean() > v].tolist()
def removeSmallCapStocks(sdk, stockCodes, period, quantile):
    # returns the stockcodes with larger the quantile trading capital
    stockCodeList = sdk.getStockList()
    tradcap = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_VAL_A_TCAP", period),
                           columns=stockCodeList)[stockCodes]
    v = tradcap.mean().quantile(quantile)
    return tradcap.columns[tradcap.mean() > v].tolist()
"""operating methods"""
def updateGlobalMt(sdk, series):
    mt = sdk.getGlobal("MT")
    # dealing with the case a new stock come into stockCodeList
    dif = len(sdk.getStockList()) - len(mt.columns)
    if dif > 0:
        # extrating new listed stock codes
        # newCol[0] should be the first added and newCol[-1] should be the last added
        newCol = sdk.getStockList()[-dif:]
        for i, item in enumerate(newCol):
            mt[item] = np.nan
    mt.loc[len(mt.index)] = series
    sdk.setGlobal("MT", mt)
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
    """need to make sure that stockCodes has no stock with Nan value for last numberOfObser * period """
    priceAdjdf = pd.DataFrame(columns=stockCodes)
    for stock in stockCodes:
        priceadj = {i: item.close * item.factor
                    for i, item in enumerate(sdk.getLatest(code=stock, count=period * numOfObservatios, timefreq="1D"))}
        stockPriceAdj = pd.Series(data=priceadj.values(), index=priceadj.keys())
        priceAdjdf[stock] = stockPriceAdj

    p = priceAdjdf.loc[::-period].loc[::-1]
    rts = p / p.shift(1) - 1
    rts.drop(rts.index[0], inplace=True)  # drop the NaN value of returns
    cluster = rts > 0  # the binary state of rts
    mt = cluster.apply(lambda x: binaryDet(x))
    return mt
# use last period median return as measure of last period performance
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
    win = rts.median() > 0
    lose = rts.median() < 0
    return win[win].index.tolist(), lose[lose].index.tolist()
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
"""position adjusting methods"""
def unSuccessfulSell(sdk, sellStocks):
    remainStocks = [i.code for i in sdk.getPositions()]
    return list(set(remainStocks) & set(sellStocks))
# sell stock return unsuccessful sells
def sellStockAndShowUnsuccess(sdk, stockToSell,quotes):
    sellAllPositionInStocks(sdk, stockToSell, quotes)
    return unSuccessfulSell(sdk, stockToSell)
def constrianDelayedSell(stockToBuy, stockToSell, delayedSell):
    sell = set(delayedSell) - set(stockToBuy)
    sell = sell - set(stockToSell)
    return sell
def sellandUpdateDelayed(sdk, stockToBuy, stockToSell, delayed):
    if delayed:
        delayed = constrianDelayedSell(stockToBuy, stockToSell, delayed)
        quotes = sdk.getQuotes(list(delayed))
        unsuccessful = sellStockAndShowUnsuccess(sdk, list(delayed), quotes)
        sdk.setGlobal("DELAYSELLS", set(unsuccessful))
# the buy stock methods, currently buy at open
def buyStocksWithCap(sdk, stockToBuyWithCap, quotes):
    quoteStocks = quotes.keys()
    stockToBuy = list(set(stockToBuyWithCap.index.tolist()) & set(quoteStocks))
    asset = sdk.getAccountInfo()
    if stockToBuy and asset:
        orders = []
        for stock in stockToBuy:
            buyPrice = quotes[stock].open
            buyAmount = int(np.round(stockToBuyWithCap.loc[stock]/buyPrice, -2))
            if buyPrice > 0 and buyAmount >= 100:
                orders.append([stock, buyPrice, buyAmount, "BUY"])
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'buy')
def buyStocks(sdk, stockToBuy, quotes):
    quoteStocks = quotes.keys()
    stockToBuy = list(set(stockToBuy) & set(quoteStocks))
    asset = sdk.getAccountInfo()
    # 剩余现金作为购买预算 各支股票平均分配预算
    if stockToBuy and asset:
        budget = asset.availableCash * 0.1 / len(stockToBuy)
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
def sellStocksWithCap(sdk, stockToSellWithCap, quotes):
    quoteStocks = quotes.keys()
    stockToSell = list(set(stockToSellWithCap.index.tolist()) & set(quoteStocks))
    if stockToSell:
        orders = []
        for stock in stockToSell:
            sellPrice = quotes[stock].open
            sellAmount = int(np.round(-stockToSellWithCap.loc[stock]/sellPrice, -2))
            if sellPrice > 0 and sellAmount >= 100:
                orders.append([stock, sellPrice, sellAmount, "SELL"])
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'sell')
"""pick up self method"""
def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()
