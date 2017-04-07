# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator

"""1. stock filtering methods"""
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

"""2. measuring last period performance"""
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
"""3. tech signals"""
# mt signals
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

"""4. optimization buying stocks"""
# this function should avoid to use todays data, len(FactorNames) < numObsers
def getOptWeight(sdk, stockCodeList, FactorNames, obsers, period, bencmarkIndexCode):
    ###        important notes                ###
    # Factor  files should be prepared in init()#
    # sdk.prepare(FactorNames)                  #
    #############################################
    numOfFactor = len(FactorNames)
    stockCodes = sdk.getStockList()
    # get stock returns use adjustment for div and split
    rts = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_QUOTE_TCLOSE",
                                             obsers*period), columns=stockCodes)[stockCodeList]
    cumFac = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_CMFTR_CUM_FACTOR",
                                                obsers * period), columns=stockCodes)[stockCodeList]
    # set the stock index as benchmark
    stock_index = sdk.getFieldData("LZ_GPA_TMP_INDEX")
    stock_index_price = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_INDXQUOTE_CLOSE",
                                                           obsers * period), columns=stock_index)[bencmarkIndexCode]
    trials = np.array(range(0, (obsers+1)*period, period))-1
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
    drop_label = rts.columns[rts.isnull().sum() > (obsers-numOfFactor)].tolist()               ##
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
        dt = pd.DataFrame(data=sdk.getFieldData(name, obsers*period), columns=stockCodes)[stockCodeList].iloc[trials]
        # normlise the factors
        dt = dt.apply(lambda x: (x-x.mean()) / x.std())
        drop_labels = dt.columns[dt.isnull().sum() > (obsers-numOfFactor)].tolist()
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
    speciRisk = matrix(np.identity(n=len(stockCodeList), dtype=float)*(factor_return_residual.std().values**2),
                       (len(stockCodeList), len(stockCodeList)))

    f = matrix(factor_return_df.values.tolist(), (len(rts.index), len(FactorNames)))
    F = f.T * f
    exp = matrix(stockExposure.values.tolist(), (len(stockCodeList), len(FactorNames)))
    facRisk = exp * F * exp.T
    # the expected returns
    exprts = -1 * rts[stockCodeList].mean().values
    # 3.2 solving the quadratic cone min Z'SigZ to maximize sharpe ratio
    P = facRisk + speciRisk  # ---  #Assest X #Asset # the quadratics term
    q = matrix(np.zeros((1, len(stockCodeList))), (len(stockCodeList), 1))  # first order term zeros
    # now minimize the totoal risk with respect to certain weight
    # import cvxopt to solve
    # the constrains
    # u'z = 1
    A = matrix(exprts, (1, len(stockCodeList)))
    b = matrix(1.0)
    # the inequaility constraint
    d = matrix(np.identity(len(stockCodeList)))
    e = matrix(1, (1, len(stockCodeList)))
    G = -1 * matrix([d, e])
    h = matrix(np.zeros((len(stockCodeList)+1, 1)))
    # solving the QP
    # settings of the solvers
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    z = matrix(sol["x"])
    k = e*z
    soldf = pd.Series(index=stockCodeList, data=np.array(z/k).reshape(1, len(stockCodeList))[0])
    return soldf

"""5. buy or sell methods"""
'''5.1 in dealing with unsuccessful sell'''
def unSuccessfulSell(sdk, sellStocks):
    remainStocks = [i.code for i in sdk.getPositions()]
    return list(set(remainStocks) & set(sellStocks))
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
'''5.2 in dealing with unsuccessful buy'''
def unSuccessfulBuy(sdk, sellStocks):
    remainStocks = [i.code for i in sdk.getPositions()]
    return list(set(remainStocks) & set(sellStocks))
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

"""6. sdk, ultility functions"""
def updateGlobal(sdk, series, name):
    mt = sdk.getGlobal(name)
    # dealing with the case a new stock come into stockCodeList
    dif = len(sdk.getStockList()) - len(mt.columns)
    if dif > 0:
        # extrating new listed stock codes
        # newCol[0] should be the first added and newCol[-1] should be the last added
        newCol = sdk.getStockList()[-dif:]
        for i, item in enumerate(newCol):
            mt[item] = np.nan
    mt.loc[len(mt.index)] = series
    sdk.setGlobal(name, mt)
# stock index dict function
def indexDict(sdk):
    array = sdk.getFieldData("LZ_GPA_TMP_INDEX")
    dict = {code: index for index, code in enumerate(array)}
    return dict
# get stock index close given indexCode
def getIndexClose(sdk, indexCode, count):
    index = indexDict(sdk)[indexCode]
    indexClose = pd.Series(data=sdk.getFieldData("LZ_GPA_INDXQUOTE_CLOSE", count)[:, index], name=indexCode)
    return indexClose