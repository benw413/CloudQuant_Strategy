# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import cvxopt as cvx
import statsmodels.api as sm
from CloudQuant import MiniSimulator  # 导入云宽客SDK
from joblib import Parallel, delayed

# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from CloudQuant import MiniSimulator  # 导入云宽客SDK
from joblib import Parallel, delayed

INIT_CAP = 80000000  # init capital
START_DATE = '20160101'  # backtesting start
END_DATE = '20160109'  # backtesting end

PERIOD = 20  # the period used to calculate win/lose
UP_BAND = 0.6  # the buy signal band
DOWN_BAND = 0.3  # the sell signal band
FACTORS = ["LZ_GPA_VAL_PE", "LZ_GPA_DERI_LnFloatCap", "LZ_GPA_QUOTE_TVOLUME"]
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
    'assetType': 'STOCK'
}

def initial(sdk):
    sdk.prepareData(["LZ_GPA_CMFTR_CUM_FACTOR",
                     "LZ_GPA_QUOTE_TCLOSE",
                     "LZ_GPA_INDXQUOTE_CLOSE",
                     "LZ_GPA_TMP_INDEX",
                     "LZ_GPA_INDU_ZX"])
    sdk.prepareData(FACTORS)
    global day
    day = 0
def initPerDay(sdk):
    global day
    #if day % PERIOD == 0:
     #   stockCodeList = sdk.getStockList()[0:6]
      #  getOptWeight(sdk, stockCodeList, FACTORS, 10, "000985")
    #day += 1
    exposurePeriod = 12
    PERIOD = 20
    trials = np.array(range(0, (exposurePeriod + 1) * PERIOD, PERIOD)) - 1
    trials[0] = 0
    a = pd.DataFrame(data=sdk.getFieldData("LZ_GPA_VAL_PE", 12*20),
                     columns=sdk.getStockList())["002173"].iloc[trials]
    print(a.isnull().sum())
def strategy(sdk):
    pass
def getOptWeight(sdk, stockCodeList, FactorNames, exposurePeriod, bencmarkIndexCode):
    ###        important notes                ###
    # Factor  files should be prepared in init()#
    # sdk.prepare(FactorNames)                  #
    #############################################
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
        # save only stocks we are interested in
        factorDataList.append(dt.drop(dt.index[-1]))
    # construct the factor df for each stock
    stockDataList = []
    for stockCode in stockCodeList:
        stockFactor = pd.DataFrame(columns=FactorNames)
        for i, factor in enumerate(factorDataList):
            stockFactor[FactorNames[i]] = factor[stockCode]# this is a series
        # normlise the factors
        stockFactor = stockFactor.apply(lambda x: (x - x.mean()) / x.std())
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
        cross_sectional_rts = pd.Series(index=stockCodeList, name=index, data=rts.loc[index].values)
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
    soldf = pd.DataFrame(columns=stockCodeList)
    soldf.loc["sol-x"] = np.array(sol["x"]).reshape(1, len(stockCodeList))[0]
    return soldf

def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()
