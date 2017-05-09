import numpy as np
import pandas as pd
import Tools_strategy_function as tsf
from CloudQuant import MiniSimulator

'''sdk backtest settings'''
NAME = 'LZ_TEST_BPSSB_HA_001035'
INIT_CAP = 1000000000  # init capital
START_DATE = '20120101'  # backtesting start
END_DATE = '20170101'  # backtesting end
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
    'strategyName': NAME,  # strategy name
    "logfile": "maday",
    'dealByVolume': True,
    "memorySize": 5,
    'assetType': 'STOCK'
}
''' sdk framework'''
def initial(sdk):
    pass
def initPerDay(sdk):
    pass
def strategy(sdk):
    pass
'''strategy '''

'''pick up self methods'''
def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    MiniSimulator(**config).run()
if __name__ == "__main__":
    main()
