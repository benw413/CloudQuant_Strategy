# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK


INIT_CAP = 100000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170301'  # backtesting end

PERIOD = 20  # the period used to calculate win/lose
UP_BAND = 0.6  # the buy signal band
DOWN_BAND = 0.25  # the sell signal band
FACTORS = ["LZ_GPA_VAL_PB",
           "LZ_GPA_FIN_IND_PROFITTOGR",
           "LZ_GPA_FIN_IND_QFA_CGRGR",
           "LZ_GPA_DERI_LnFloatCap",
           "LZ_GPA_QUOTE_TVOLUME"]

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
    'strategyName': 'Strategy_weightedContrast',  # strategy name
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
        # set optimal weight to in position aseests
        #currentHolding = [i.code for i in sdk.getPositions()]
        # intend to hold these stocks
        #intend = list(set(currentHolding) | set(stockToBuy))
        #if intend:
         #   optWeight = getOptWeight(sdk, intend, FACTORS, 12, "000985")
          #  adjustPosition(sdk, optWeight, quotes)  # adjust portfolio with weight
        if stockToBuy:
            buyStocks(sdk, stockToBuy, quotes)
        # update the state
        sdk.setGlobal("STATE", newState)
#  returns a Series of stockcodes and its optimal weight
def getOptWeight(sdk, stockCodeList, FactorNames, exposurePeriod, bencmarkIndexCode):
    ###        important notes                ###
    # Factor  files should be prepared in init()#
    # sdk.prepare(FactorNames)                  #
    #############################################
    numOfFactor = len(FactorNames)
    stockCodes = sdk.getStockList()
    # get stock returns use adjustment for div and split
    rts = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TCLOSE",
                                             (exposurePeriod)*PERIOD), columns=stockCodes)[stockCodeList]
    cumFac = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_CMFTR_CUM_FACTOR",
                                                (exposurePeriod)*PERIOD), columns=stockCodes)[stockCodeList]
    # set the stock index as benchmark
    stock_index =sdk.getFieldData("LZ_GPA_TMP_INDEX")
    stock_index_price = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_INDXQUOTE_CLOSE",
                                                      (exposurePeriod) * PERIOD), columns=stock_index)[bencmarkIndexCode]
    trials = np.array(range(0, (exposurePeriod+1)*PERIOD, PERIOD))-1
    trials[0] = 0
    rts = (rts*cumFac).iloc[trials]  # adj for div and split
    stock_index_price = stock_index_price.iloc[trials]
    rts = (rts / (rts.shift(1)) - 1).shift(-1).drop(rts.index[-1])# cacl the returns and drop the Nan row
    # index returns
    stock_index_price = (stock_index_price / stock_index_price.shift(1) - 1).shift(-1).drop(stock_index_price.index[-1])
    # get the excess returns
    rts = rts.apply(lambda x: x-stock_index_price)

    '''1st handling nan values in rts, reset stockCodeList'''
    # should drop nan values of rts that is too much to calc OLS and should drop it from label   ##
    drop_label = rts.columns[rts.isnull().sum() > (exposurePeriod-numOfFactor)].tolist()               ##
    stockCodeList = list(set(stockCodeList) - set(drop_label))
    ###############################################################################################
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
    drop_label = []
    ### factordatalist elment dataframe dt
    #  days before today  | factor value    |
    # 0         P+1       |     v0          |
    # 1         P         |     v1          |
    # .         .         |      .          |
    # .         .         |      .          |
    # P-1       2         |     vp-1        |
    # P         1         |     droped      |
    #########################################
    for name in FactorNames:
        dt = pd.DataFrame(data=sdk.getFieldData(name, (exposurePeriod)*PERIOD), columns=stockCodes)[stockCodeList].iloc[trials]
        # normlise the factors
        dt = dt.apply(lambda x: (x-x.mean()) / x.std())
        drop_labels = dt.columns[dt.isnull().sum() > (exposurePeriod-numOfFactor)].tolist()
        drop_label.extend(drop_labels)
        # save only stocks we are interested in
        factorDataList.append(dt.drop(dt.index[-1]))
    #############################################################
    '''2nd handling nan values in Factors, reset stockCodeList'''
    stockCodeList = list(set(stockCodeList) - set(drop_label))
    #############################################################
    ''' from now on the stockCodeList has fixed order'''
    # construct the factor df for each stock
    stockDataList = []
    for stockCode in stockCodeList:
        stockFactor = pd.DataFrame(columns=FactorNames)
        for i, factor in enumerate(factorDataList):
            stockFactor[FactorNames[i]] = factor[stockCode]# this is a series
        stockDataList.append(stockFactor)
    # 1. regression on time series for each asset to find factor exposure
    stockExposure = pd.DataFrame(columns=FactorNames, index=stockCodeList)
    #        example of stockExposure  #Asset X #Factors  ####
    #  code  | LZ_GPA_VAL_PE      | LZ_GPA_DERI_LnFloatCap   #
    # 000005 | -0.0374334         |        1.19216           #
    # 600601 |  0.00104023        |       -0.0295672         #
    # 600602 | -0.000732552       |        0.138838          #
    # 600651 | -0.000860048       |        0.159073          #
    # 600652 | -0.000919313       |        0.278275          #
    ##########################################################
    # cacl exposure of individual stock to factor
    for i, stockCode in enumerate(stockCodeList):
        df = stockDataList[i]  # index = time , columns = factors  : nxk
        df[stockCode] = rts[stockCode]  # series of stock returns  : nx1
        X = sm.add_constant(df[FactorNames].astype(float))
        result = sm.OLS(df[stockCode].astype(float), X, missing="drop").fit()
        exposure = result.params  # a series contains regression parameters
        stockExposure.loc[stockCode] = exposure
    # 2. regression on cross-sectional data to find the factor returns
    # get the cross_sectional returns and store it as pd.Series & the residual values
    factor_return_residual = pd.DataFrame(index=rts.index, columns=["residual"])
    factor_return_df = pd.DataFrame(index=rts.index, columns=FactorNames)
    ##    LZ_GPA_VAL_PE    |    LZ_GPA_DERI_LnFloatCap
    # 0        -5.73738    |          -0.234322
    # 19       -0.496935    |          0.123082
    # 39        4.15772    |           0.0787571
    # 59        0.755598    |          0.124334
    # 79       -2.79853    |          -0.0216826
    # 99        4.60543    |           0.0540049
    # 119       1.48025    |           0.0763404
    # 139       0.918471    |          0.053799
    # 159      -6.90497    |          -0.259223
    # 179       4.62786    |           0.0687939
    for index in rts.index:
        cross_sectional_rts = pd.Series(index=stockCodeList, name=index, data=rts[stockCodeList].loc[index].values)
        X = sm.add_constant(stockExposure.astype(float))
        result = sm.OLS(cross_sectional_rts.astype(float), X, missing="drop").fit()
        factor_returns = result.params
        factor_return_df.loc[index] = factor_returns
        factor_return_residual.loc[index] = factor_returns.loc["const"]

    # 3. decide weight on asset to optimize
    # 3.1 should be diagonalized specific risk variance factor_return_residual
    speciRisk = np.identity(n=len(stockCodeList), dtype=float)*factor_return_residual.std().values**2
    F = np.dot(factor_return_df.values.transpose(), factor_return_df.values)
    facRisk = np.dot(np.dot(stockExposure.values, F), stockExposure.values.transpose())

    # 3.2 solving the QP
    P =matrix((facRisk + speciRisk).tolist())  # ---  #Assest X #Asset # the quadratics term
    q = matrix(np.zeros((1, len(stockCodeList))).tolist())  # first oreder term
    # now minimize the totoal risk with respect to certain weight
    # import cvxopt to solve
    # the constrains
    A = matrix(np.ones((1, len(stockCodeList))), (1, len(stockCodeList)))
    b = matrix(1.0)

    Garray = np.identity(len(stockCodeList), dtype=float)*(-1)
    G = matrix(Garray.tolist())
    h = matrix(np.zeros((1, len(stockCodeList))).tolist())
    # solving the QP
    sol = solvers.qp(P, q, G, h, A, b)
    soldf = pd.Series(index=stockCodeList, data=np.array(sol["x"]).reshape(1, len(stockCodeList))[0])
    return soldf
def getAccountCapital(sdk, quotes):
    dict_position = {i.code: i.optPosition for i in sdk.getPositions()}
    dict_price = {i: quotes[i].open for i in dict_position.keys()}
    dict_cap = {i: dict_position[i]*dict_price[i] for i in dict_position.keys()}

    cap = sdk.getAccountInfo().availableCash
    for i in dict_cap.keys():
        cap += dict_cap[i]
    return cap
# adjusting position in stocks
def adjustPosition(sdk, stockWithWeight,quotes):
    dict_position = {i.code: i.optPosition for i in sdk.getPositions()}
    dict_price = {i: quotes[i].open for i in dict_position.keys()}
    dict_cap = {i: dict_position[i] * dict_price[i] for i in dict_position.keys()}
    # captial in account
    cap = sdk.getAccountInfo().availableCash
    for i in dict_cap.keys():
        cap += dict_cap[i]
    # target portfolio
    portfolio = stockWithWeight * cap
    # current portfolio
    current_port = pd.Series(data=dict_cap)
    # distance to adjust
    distance = pd.Series()
    if dict_cap:
        for i in portfolio.index:
            if i in current_port.index:
                distance.loc[i] = portfolio.loc[i] - current_port.loc[i]
            else:
                distance.loc[i] = portfolio.loc[i]
    else:
        distance = portfolio
    # first sell the under weighted asset to required weight
    tosell = distance[distance < 0]
    print(tosell)
    sellStocksWithCap(sdk, tosell, quotes)
    # second buy the over weighted asset to required weight
    tobuy = distance[distance > 0]
    print(tobuy)
    buyStocksWithCap(sdk, tobuy, quotes)
# the buy stock methods
def buyStocksWithCap(sdk, stockToBuyWithCap, quotes):
    quoteStocks = quotes.keys()
    stockToBuy = list(set(stockToBuyWithCap.index.tolist()) & set(quoteStocks))
    asset = sdk.getAccountInfo()
    if stockToBuy and asset:
        orders = []
        for stock in stockToBuy:
            buyPrice = quotes[stock].high
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
        budget = asset.availableCash / len(stockToBuy)
        orders = []
        for buyStock in stockToBuy:
            buyPrice = quotes[buyStock].high  # 购买价格为上分钟最高价
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
                sellPrice = quotes[pos.code].low  # 设置出售价格为上分钟最低价
                sellAmount = pos.optPosition
                if sellPrice > 0 and sellAmount > 100:
                    orders.append([pos.code, sellPrice, sellAmount, 'SELL'])  # 委托出售
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'sell')  # 将出售记入日志
def sellStocksWithCap(sdk, stockToSellWithCap, quotes):
    quoteStocks = quotes.keys()
    stockToSell = list(set(stockToSellWithCap.index.tolist()) & set(quoteStocks))
    if stockToSell:
        orders = []
        for stock in stockToSell:
            sellPrice = quotes[stock].low
            sellAmount = int(np.round(-stockToSellWithCap.loc[stock]/sellPrice, -2))
            if sellPrice > 0 and sellAmount >= 100:
                orders.append([stock, sellPrice, sellAmount, "SELL"])
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'sell')

def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()