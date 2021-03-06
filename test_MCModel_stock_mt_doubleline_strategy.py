# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK

"""this is the stock double line stock mt method, the method looks fine and showed potential in future study"""
'''this strategy avioded using diverted conditions in real time market as far as possible, checked 2017/04013'''
''' there is still rooms to optimize without overfitting '''
INIT_CAP = 100000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170101'  # backtesting end
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
    'strategyName': 'stock_mt_doubleline_strategy',  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'STOCK'
}
'''double line method seems to be robust on dayline timing, with good potential to be examed in futrue'''
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
    stockCodeListFiletered2 = removeIlliquidStocks(sdk, stockCodeListFiltered1, 2*PERIOD, 0.5)
    stockCodeListFiletered3 = removeSmallCapStocks(sdk, stockCodeListFiletered2, 2*PERIOD, 0.5)
    stockCodeListFiletered4 = removeTodayInvalidStocks(sdk, stockCodeListFiletered3)

    currentStock = [i.code for i in sdk.getPositions()]
    sdk.setGlobal("Holdings", currentStock)
    pool = list(set(stockCodeListFiletered4) | set(currentStock))
    sdk.setGlobal("Pool", pool)
    # initialized the global parameter
    if sdk.getGlobal("State") is None:
        stockCodeList = sdk.getStockList()
        mtdf = pd.DataFrame(data=np.zeros((1, len(stockCodeList))), columns=stockCodeList)
        sdk.setGlobal("State", mtdf)
    # initialized the global parameter
    if sdk.getGlobal("DELAYSELLS") is None:
        sdk.setGlobal("DELAYSELLS", set())
    global dayCounter
    dayCounter += 1
def strategy(sdk):
    global dayCounter
    updateMtPeriod = 1

    if dayCounter > 2:
        lastState = sdk.getGlobal("State").loc[sdk.getGlobal("State").index[-1]]
        last2State = sdk.getGlobal("State").loc[sdk.getGlobal("State").index[-2]]

        highM = []
        lowM = []
        ''' there are situations that lastState == lastState2'''
        for i, index in enumerate(lastState.index):
            if lastState.loc[index] == 1 and last2State.loc[index] == -1:
                highM.append(index)  # up crossing
            elif lastState.loc[index] == -1 and last2State.loc[index] == 1:
                lowM.append(index)  # down crossing
        stockToBuy = []
        stockToSell = []
        stockToSell.extend(lowM)
        if highM:
            buy, sell = checkLastPeriodPerformanceMean(sdk, highM, 5)
            buy2, sell2 = checkLastPeriodPerformanceMedian(sdk, highM, 5)

            buy3 = list(set(buy) & set(buy2))
            sell3 = list(set(sell) & set(sell2))

            stockToBuy.extend(buy3)
            stockToSell.extend(sell3)
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

    '''update the MT signal to todays price, should be done at end of strategy'''
    # using last 20 observations
    mtLseries = stockBMT(sdk, sdk.getGlobal("Pool"), 20, updateMtPeriod)
    mtSseries = stockBMT(sdk, sdk.getGlobal("Pool"), 10, updateMtPeriod)

    mtUpState = mtSseries - mtLseries > 0
    '''< =  is important'''
    mtDownState = mtSseries - mtLseries <= 0

    State = pd.Series()
    for i, index in enumerate(mtLseries.index):
        if mtUpState.loc[index] == True:
            State.loc[index] = 1
        elif mtDownState.loc[index] == True:
            State.loc[index] = -1
    updateGlobalMt(sdk, State, "State")
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
def updateGlobalMt(sdk, series,name):
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
def getStocksForIndustry(sdk, index):
    stockCodes = sdk.getStockList()
    industry = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_INDU_ZX", 1), columns=stockCodes)
    series = industry.iloc[0]
    list = series[series == index].index.tolist()
    return list
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
def checkLastPeriodPerformanceMedian(sdk, stockCodes, period):
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
def checkLastPeriodPerformanceMean(sdk, stockCodes, period):
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
    drop_label = rts.columns[rts.isnull().sum() > (exposurePeriod-numOfFactor-1)].tolist()               ##
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
        drop_labels = dt.columns[dt.isnull().sum() > (exposurePeriod-numOfFactor-1)].tolist()
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
    #F = np.dot(factor_return_df.values.transpose(), factor_return_df.values)
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
"""position adjusting methods"""
# adjusting position in stocks
def adjustPosition(sdk, stockWithWeight, quotes, totalCap):
    dict_position = {i.code: i.optPosition for i in sdk.getPositions()}
    dict_price = {i: quotes[i].open for i in dict_position.keys()}
    dict_cap = {i: dict_position[i] * dict_price[i] for i in dict_position.keys()}
    # target portfolio
    portfolio = stockWithWeight * totalCap
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
    sellStocksWithCap(sdk, tosell, quotes)
    # second buy the over weighted asset to required weight
    tobuy = distance[distance > 0]
    buyStocksWithCap(sdk, tobuy, quotes)
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
    if stockToBuy and asset:
        budget = asset.availableCash / len(stockToBuy)
        orders = []
        for buyStock in stockToBuy:
            buyPrice = quotes[buyStock].open
            buyAmount = int(np.round(budget/buyPrice, -2))
            if buyPrice > 0 and buyAmount >= 100:
                orders.append([buyStock, buyPrice, buyAmount, 'BUY'])
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'buy')
# the sell method
def sellAllPositionInStocks(sdk, stockToSell, quotes):
    quoteStocks = quotes.keys()
    stockToSell = list(set(stockToSell) & set(quoteStocks))
    if stockToSell:
        orders = []
        positions = sdk.getPositions()
        for pos in positions:
            if pos.code in stockToSell:
                sellPrice = quotes[pos.code].open
                sellAmount = pos.optPosition
                if sellPrice > 0 and sellAmount > 100:
                    orders.append([pos.code, sellPrice, sellAmount, 'SELL'])
        if orders:
            sdk.makeOrders(orders)
            sdk.sdklog(orders, 'sell')
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
