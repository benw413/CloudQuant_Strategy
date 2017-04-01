# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
period = 20
filepath = 'D:\cStrategy\Factor\LZ_GPA_QUOTE_TCLOSE.csv'
filepath2 = "D:\cStrategy\Factor\LZ_GPA_CMFTR_CUM_FACTOR.csv"
df = pd.read_csv(filepath_or_buffer=filepath).iloc[range(0, 2970)]
index = df[df.columns[0]].loc[period+2:]
cum = pd.read_csv(filepath_or_buffer=filepath2).iloc[range(0, 2970)]
df = df * cum
df = df.loc[::5]
df.dropna(axis=1, inplace=True)
date = pd.to_datetime(index.astype(str))
df = df / df.shift(1) - 1
df.drop(df.index[0], inplace=True)
a = df > 0
print(df.columns)
def countBinary(series):
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
c = pd.DataFrame(columns=df.columns)
for i, trial in enumerate(a.index):
    if i > period:
        d = a.loc[:i].tail(period)
        c.loc[i-period] = d.apply(lambda x: countBinary(x))
result = c.set_index(date)
z = np.zeros((1, len(result.index)))
ii = 1
zero = pd.Series(data=z[0], index=result.index)
sig = result[result.columns[ii]]
mean = result[result.columns[ii]].mean()
std = result[result.columns[ii]].std()

m1s = (mean+std)*pd.Series(data=np.ones((1, len(result.index)))[0], index=result.index)
m_1s = (mean-std)*pd.Series(data=np.ones((1, len(result.index)))[0], index=result.index)

sig.plot()
m1s.plot()
m_1s.plot()


