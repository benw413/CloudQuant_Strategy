# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import talib
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK

"""this is the stock mt method using mean+std as band, and momentum only.
the method shows potential to be future studied"""
INIT_CAP = 1000000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170301'  # backtesting end

PERIOD = 30  # the period used to calculate win/lose
UP_BAND = 0.1  # the initial up band
DOWN_BAND = -0.2  # the initial down band
stdObsers = 20  # the param
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
    'strategyName': 'stock_mt_monly_strategy',  # strategy name
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
    #sdk.prepareData(FACTORS)
def initPerDay(sdk):
    stockCodeList = sdk.getStockList()
    stockCodeListFiltered1 = removeSTStocks(sdk, stockCodeList, 2*PERIOD)
    stockCodeListFiletered2 = removeIlliquidStocks(sdk, stockCodeListFiltered1, 2*PERIOD, 0.5)
    stockCodeListFiletered3 = removeSmallCapStocks(sdk, stockCodeListFiletered2, 2*PERIOD, 0.5)
    stockCodeListFiletered4 = removeTodayInvalidStocks(sdk, stockCodeListFiletered3)

    currentStock = [i.code for i in sdk.getPositions()]
    sdk.setGlobal("Holdings", currentStock)
    pool = list(set(stockCodeListFiletered4) | set(currentStock))
    sdk.setGlobal("Pool", pool)
    # initialized the global parameter
    if sdk.getGlobal("MACD") is None:
        stockCodeList = sdk.getStockList()
        df = pd.DataFrame(np.zeros((1, len(stockCodeList))), columns=stockCodeList)
        sdk.setGlobal("MACD", df)
        sdk.setGlobal("DEA", df)
        sdk.setGlobal("HIST", df)
    global dayCounter
    dayCounter += 1
def strategy(sdk):
    global dayCounter
    updatePeriod = 5
    if (dayCounter-1)%updatePeriod == 0:
        print(MACD(sdk, sdk.getGlobal("Pool"), 40, 5))
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
def updateGlobalMACD(sdk, series):
    mt = sdk.getGlobal("MACD")
    # dealing with the case a new stock come into stockCodeList
    dif = len(sdk.getStockList()) - len(mt.columns)
    if dif > 0:
        # extrating new listed stock codes
        # newCol[0] should be the first added and newCol[-1] should be the last added
        newCol = sdk.getStockList()[-dif:]
        for i, item in enumerate(newCol):
            mt[item] = np.nan
    mt.loc[len(mt.index)] = series
    sdk.setGlobal("MACD", mt)
def MACD(sdk, stockCodes, numOfObservations, period=1):
    Macd = pd.DataFrame(columns=stockCodes)
    DEA = pd.DataFrame(columns=stockCodes)
    Hist = pd.DataFrame(columns=stockCodes)
    priceAdjdf = pd.DataFrame(columns=stockCodes)
    for stock in stockCodes:
        priceadj = {i: item.close * item.factor
                    for i, item in enumerate(sdk.getLatest(code=stock, count=numOfObservations*period, timefreq="1D"))}
        stockPriceAdj = pd.Series(data=priceadj.values(), index=priceadj.keys())
        priceAdjdf[stock] = stockPriceAdj
    p = priceAdjdf.loc[::-period].loc[::-1]
    for stock in p.columns:
        print(p[stock].values)
        macd, dea, hist = talib.MACD(np.array(p[stock].values), fastperiod=12, slowperiod=26, signalperiod=9)
        Macd[stock] = macd[-1]
        DEA[stock] = dea[-1]
        Hist[stock] = hist[-1]
    return Macd, DEA, Hist
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