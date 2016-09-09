# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:29:38 2016

@author: yiran.zhou
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import taifook.taifook as tf
import zigzag_c as zz
import pylab as pl


def plot_pivots(X, pivots):
    pl.xlim(0, len(X))
    pl.ylim(X.min()*0.99, X.max()*1.01)
    pl.plot(np.arange(len(X)), X, 'k:', alpha=0.8)
    pl.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    pl.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    pl.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')


if __name__ == '__main__':
    
    # 读入数据
    pos = pd.read_excel('INPUT COT.xls', sheetname = 'Sheet1 (2)')
    pos.set_index('Date', inplace = True)
    price = pd.read_excel('INPUT COT.xls', sheetname = 'Sheet2')
    label = price.columns[0]
    price.set_index(label, inplace = True)

#    posDate = pos.index #统计pos的日期是周二，但是release是周末
#    priceDate = posDate.shift(6, 'D') #所以下个周一（6天后）是第一个公布数据后的交易日
#    selDatePrice = price.ix[priceDate].copy()
#
#    ust = selDatePrice.ix[-50:-1, 'PX_LAST']
    
    ust = price.ix[:, 'PX_LAST']
    
#    ust = np.float64(ust)
    
    pivots = zz.peak_valley_pivots(ust, 0.015, -0.015)  
    plot_pivots(ust, pivots)
    
    price['pivots'] = pivots
    price.to_excel('price with pivots.xls')
    selPrice = price[price.pivots == 1]
    nextMon = tf.findNearbyDate(selPrice.index, 7, 'W-MON')
    prevMon = tf.findNearbyDate(selPrice.index, -7, 'W-MON')
    mon = pd.DataFrame()
    mon['nextMon'] = nextMon
    mon['prevMon'] = prevMon
    mon.to_excel('peak Mon.xls')
    
    selPrice = price[price.pivots == -1]
    nextMon = tf.findNearbyDate(selPrice.index, 7, 'W-MON')
    prevMon = tf.findNearbyDate(selPrice.index, -7, 'W-MON')
    mon = pd.DataFrame()
    mon['nextMon'] = nextMon
    mon['prevMon'] = prevMon
    mon.to_excel('valley Mon.xls')
    
    
    
    
#    pivots = zz.peak_valley_pivots(ust, 0.03, -0.03)
#    plot_pivots(ust, pivots)
    