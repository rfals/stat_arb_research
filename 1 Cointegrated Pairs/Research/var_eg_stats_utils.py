import seaborn as sns; sns.set()

import numpy as np
import pandas as pd

import scipy.stats as s


def profit_loss(x):
	'''% P&L Return or Cumulative Return'''
	return 100*(x.cumsum().apply(np.exp)[-1]-1)

def pl_CAGR(x):
	'''% P&L CAGR Return or Annualised Return'''
	y = float((x.index[-1]-x.index[0]).days)/(365) # assuming data is in hours
	return 100*(x.cumsum().apply(np.exp)[-1]**(1/y)-1)

def an_vol(x, freq):
    if freq == 'daily':
        an = 365
    elif freq == 'weekly':
        an = 52
    elif freq == 'monthly':
        an = 12
    elif freq == 'quarterly':
        an = 4
    elif freq == 'hourly':
        an = 24 * 365
    elif freq == 'minute':
        an = 60 * 24 * 365
    else:
        an = 1
    return 100 * (np.std(x) * np.sqrt(an))

def an_down_vol(x, freq):
    if freq == 'daily':
        an = 365
    elif freq == 'weekly':
        an = 52
    elif freq == 'monthly':
        an = 12
    elif freq == 'quarterly':
        an = 4
    elif freq == 'hourly':
        an = 24 * 365
    elif freq == 'minute':
        an = 60 * 24 * 365
    else:
        an = 1
    return 100 * (np.std(x.loc[x < 0]) * np.sqrt(an))

def sharpe(x, freq, rf):
    if freq == 'daily':
        an = 365
    elif freq == 'weekly':
        an = 52
    elif freq == 'monthly':
        an = 12
    elif freq == 'quarterly':
        an = 4
    elif freq == 'hourly':
        an = 24 * 365
    elif freq == 'minute':
        an = 60 * 24 * 365
    else:
        an = 1
    return (pl_CAGR(x) - rf) / (an_vol(x, freq))

def sortino(x, freq, rf):
    if freq == 'daily':
        an = 365
    elif freq == 'weekly':
        an = 52
    elif freq == 'monthly':
        an = 12
    elif freq == 'quarterly':
        an = 4
    elif freq == 'hourly':
        an = 24 * 365
    elif freq == 'minute':
        an = 60 * 24 * 365
    else:
        an = 1
    return (pl_CAGR(x) - rf) / (an_down_vol(x, freq))

def info_ratio(x, y, freq):
    if freq == 'daily':
        an = 365
    elif freq == 'weekly':
        an = 52
    elif freq == 'monthly':
        an = 12
    elif freq == 'quarterly':
        an = 4
    elif freq == 'hourly':
        an = 24 * 365
    elif freq == 'minute':
        an = 60 * 24 * 365
    else:
        an = 1
    return (pl_CAGR(x) - pl_CAGR(y)) / an_vol(x - y, freq)

def positive_per(x):
    ''' Percentage of positive periods'''
    return round(100*float(len(x.loc[x>0]))/len(x),2)

def max_draw(x):
    ''' Maximum/Worst Drawdown'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()
    drawdown = abs(cumret/cummax-1).dropna()
    return 100*(drawdown.max())

def max_dd_duration(x):
    ''' Maximum/Worst Drawdown Duration'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()
    drawdown = abs(cumret/cummax-1).dropna()
    temp = drawdown[drawdown.values == 0]
    periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
    return periods.max()

def worst3_draw_avg(x):
    ''' Average 3 Worst Drawdown'''
    cumret = x.cumsum().apply(np.exp)
    cummax = cumret.cummax()	
    drawdown = abs(cumret/cummax-1).dropna()
    return 100*(np.mean(sorted(drawdown, reverse=True)[0:3]))

def num_trades(pos):
    ''' Number of Trades calculated as number of time '''
    return len(pos)-sum((pos.diff().dropna()==0))

def num_trades_to_periods(pos):
    ''' Number of trades as % of total bars/periods'''
    return round(float(num_trades(pos))/len(pos),4)

def quick_perf_st(x, y, freq, rf):
    ''' Comprehensive Performance Analysis after running run_strategy() class method'''
    dfx = x.dropna()
    dfy = y.dropna()
    stats = {'Statistics': ['P&L', 'CAGR','Anual_Vol', '%_Positive',
                            'Skew','Kurtosis','Kurtosis PV','Downside_Vol', 'Worst',
                            'Sharpe_Ratio', 'Sortino_Ratio', 'Information_Ratio', 
                            'Max_Drawdown','Worst_3_drawdown_avg', 'Max_DD_Duration'], 
            'Strategy': [profit_loss(dfx), pl_CAGR(dfx), an_vol(dfx,freq), positive_per(dfx),
                             s.skew(dfx),s.kurtosis(dfx), s.kurtosistest(dfx)[1], an_down_vol(dfx,freq), dfx.min(),
                            sharpe(dfx,freq,rf), sortino(dfx,freq,rf), info_ratio(dfx,dfy,freq),
                            max_draw(dfx), worst3_draw_avg(dfx),max_dd_duration(dfx)],
            'Benchmark': [profit_loss(dfy), pl_CAGR(dfy), an_vol(dfy,freq), positive_per(dfy),
                             s.skew(dfy),s.kurtosis(dfy), s.kurtosistest(dfy)[1], an_down_vol(dfy,freq), dfy.min(),
                             sharpe(dfy,freq,rf), sortino(dfy,freq,rf), info_ratio(dfy,dfy,freq),
                             max_draw(dfy), worst3_draw_avg(dfy), max_dd_duration(dfy)] 
                             }

    df = pd.DataFrame(stats)
    df.set_index('Statistics', inplace=True)
    return df.round(4)
