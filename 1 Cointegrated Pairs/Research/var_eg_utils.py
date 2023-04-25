import pandas as pd
import numpy as np
import itertools
import quandl as q

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.optimize import brute
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.regression.linear_model import OLS

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class cointeg_pairs:
    '''
    The cointeg_pairs class analyzes time series data for cointegration and performs various
    statistical tests on the data.

    Attributes:
    -----------
    df : DataFrame
        Input dataframe containing time series data.

    Methods:
    --------
    advaced_describe(alpha=0.05) -> DataFrame:
        Returns a DataFrame with descriptive statistics for each time series in the input data, including kurtosis, skewness, and normality tests.
        
    adf_test() -> DataFrame:
        Returns a DataFrame with augmented Dickey-Fuller (ADF) test results for stationarity of each time series in the input data.
        
    p_VAR(p, constant=True) -> DataFrame:
        Returns a DataFrame with Vector Autoregressive (VAR) model coefficients for the input data.
        
    acf_plot(ts, lags_=40) -> None:
        Creates an autocorrelation plot for a given time series.
        
    coint_test_bulk(start, end, alpha=0.01, max_d=4, max_lag=None) -> DataFrame:
        Returns a DataFrame with Engle-Granger cointegration test results for all pairs of variables in the input data.

    '''
    def __init__(self, df):
        self.df = df

    def advaced_describe(self, alpha=0.05):
        '''
        pandas describe with kurtosis and skewness
        parameters
        ---------
        alpha = significance level for normality test
        normality test based on D'Agostino and Pearson's  test that combines skew and kurtosis to produce an omnibus test of normality.'
        '''
        df_ = self.df.dropna()
        df_des = df_.describe().round(5)
        des = stats.describe(df_)
        df_des.loc['skew'] = des.skewness
        df_des.loc['kurt'] = des.kurtosis
        pvals = np.round(stats.normaltest(df_)[1] > alpha, decimals=0)
        df_des.loc['normal?'] = ['Yes' if x == True else "No" for x in pvals]
        print('normal test based on D\'Agostino and Pearson\'s  test that combines skew and kurtosis to produce an omnibus test of normality. ')
        return df_des
    
    def adf_test(self):
        '''
        Returns dataframe with ADF (augmented Dickey-Fuller. Ho: unit root = Non-Stationary) stationarity test p-value
        Parameters
        -----------
        df = Dataframe or Series.
        '''
        nobs = self.df.shape[0]
        max_l = int(round((12 * nobs / 100)**(0.25), 0))  # Schwert (1989) optimal max_lag ADF = (12*T/100)^0.25
        df_ = self.df.dropna()
        df_pv_list = []
        
        try:
            for i in df_.columns:
                temp = sm.tsa.stattools.adfuller(df_[i], maxlag=max_l, regression='n')
                df_pv_list.append(pd.DataFrame({'ADF Stat': temp[0], 'ADF p-value': [temp[1]], 'Lag': temp[2],
                                                '1% CV': temp[4]['1%'], '5% CV': temp[4]['5%'], '10% CV': temp[4]['10%']},
                                                index=[i]))
            df_pv = pd.concat(df_pv_list)
        except:
            temp = sm.tsa.stattools.adfuller(df_, maxlag=max_l)
            df_pv = pd.DataFrame({'ADF Stat': temp[0], 'ADF p-value': [temp[1]], 'Lag': temp[2],
                                '1% CV': temp[4]['1%'], '5% CV': temp[4]['5%'], '10% CV': temp[4]['10%']},
                                index=['Series'])

        warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
        print('Reject Null Hypothesis: Unit Root (Non-Stationary) for p-values < threshold (e.g. 0.01 or 0.05)')

        return round(df_pv, 4)
    
    def p_VAR(self, p, constant=True):
        '''
        Return VAR model with p lags (VAR(p) model) using linear algebra: 
        Y = BZ + U
        Therefore:
        B = (Z'Z)^(-1)8(Z'Y) is the matrix with VAR(p) coefficient estimates
        Where if constant=True:
        Y = K x (T+1) matrix
        B = K x (K*P +1) matrix
        Z = (K*P + 1) x (T +1) matrix 
        U = K x (T+1) matrix = model error 
        Parameters
        -----------
        df = dataframe with time series variables
        p = number of optimal logs for implementing VAR
        '''
        # clean data
        y_p = self.df.dropna()
        y_p = np.transpose(np.matrix(y_p))
        y = y_p[:, p:]
        # parameters:
        k_p = y_p.shape[0]
        T_p = y_p.shape[1]
        k = y.shape[0]
        T = y.shape[1]
        # Z matrix = explanatory lagged variables
        Z = list()
        for i in range(1, p + 1):
            for j in range(0, k):
                col = np.array(y_p[j, p - i:T_p - i])
                Z.append(col)
        Z = np.matrix([Z[i][0] for i in range(len(Z))])
        if constant == True:
            Z = np.vstack((np.ones(T), Z))
        # solving for B:
        B = ((Z * Z.T).I * (Z * y.T)).T
        # dataframe:
        if constant == True:
            idx = list(['const'])
        else:
            idx = list()

        cols = self.df.columns
        for i in range(1, p + 1):
            for j in self.df.columns:
                idx.append('L' + str(i) + '.' + j)

        B_df = pd.DataFrame(B.T, columns=cols, index=idx)
        return B_df
    
    def acf_plot(self,ts, lags_=40):
        ts_ = pd.Series(ts)
        list1 = []
        for i in range(min(len(ts), lags_)):
            list1.append(ts_.autocorr(i))
        df = pd.DataFrame(list1, columns=['ts'])
        df['ts'].plot.bar()
        plt.title('Autocorrel Plot')
        plt.axhline(np.std(list1))
        plt.axhline(-np.std(df.dropna()['ts']), c='g')
        plt.axhline(np.std(df.dropna()['ts']), c='g')
        plt.axhline(-2 * np.std(df.dropna()['ts']), c='r')
        plt.axhline(2 * np.std(df.dropna()['ts']), c='r')

    def coint_test_bulk(self, start, end, alpha=0.01, max_d=4, max_lag=None):
        '''
        Returns dataframe with Engle Granger cointegration test results. OLS regression uses constant.
        Two different ADF statistics returned: "ADF stat" and "ADF stat sm" with the latter calculated from the 
        function "adfuller" from Python's library StatsModels to compare results.
        Remember ADF test with Null Ho: Unit Root exists (non-stationary)
        Parameters
        ----------
        df = dataframe with columns as variables to test cointegration
        start = start date. Format 'YYYY-MM_DD'
        end = end date. Format 'YYYY-MM_DD'
        alpha = default 1%.  significance level for ADF test p-values.
        max_d= 4. Maximum number of difference transformations to test for ADF test.
        max_lag = None. Maximum number of lags to be used in ADF test. If None Schwert (1989) optimal max_lag ADF is used.
        '''
        if max_lag == None:
            nobs = int(self.df.shape[0])
            max_lag = int(round((12 * nobs / 100)**(0.25), 0))

        data = self.df.loc[start:end]
        combo = list(itertools.permutations(data.columns, 2))
        dfs = []  # Create an empty list to store the dataframes
        for y, x in combo:
            for d in range(0, max_d + 1):
                if d == 0:
                    y_t = data[y].dropna()
                    x_t = data[x].dropna()
                    x_t = add_constant(x_t)
                else:
                    y_t = data[y].diff(d).dropna()
                    x_t = data[x].diff(d).dropna()
                    x_t = add_constant(x_t)

                ols = OLS(y_t, x_t).fit()
                res = ols.resid
                res_diff = np.diff(res)
                adf_df = pd.DataFrame()
                for l in range(1, max_lag + 1):
                    res_dlags = lagmat(res_diff[:, None], l, trim='both', original='in')
                    n = res_dlags.shape[0]
                    res_dlags[:, 0] = res[-n - 1:-1]
                    dy_t = res_diff[-n:]
                    dx_t = res_dlags
                    ols_adf = OLS(dy_t, dx_t).fit()
                    adf_sm = adfuller(res, maxlag=l, regression='nc', autolag=None)
                    adf_df = adf_df.append({'AIC': ols_adf.aic, 'BIC': ols_adf.bic, 'ADF Lags': l, 'ADF Stat': ols_adf.tvalues[0], 'ADF Stat sm': adf_sm[0],
                                        '1%CV': adf_sm[4]['1%'], '5%CV': adf_sm[4]['5%'], '10%CV': adf_sm[4]['10%'],
                                        'ADF P-Value': adf_sm[1], 'Diff': d, 'Index': l}, ignore_index=True)
                adf_df.set_index('Index', inplace=True)
                adf_df.index.rename(None, inplace=True)
                best_model = adf_df.sort_values('AIC').iloc[0, :]
                temp_df = pd.DataFrame({'y': [y], 'x': [x], 'diff': d, 'ADF Lags': best_model['ADF Lags'],
                                        'ADF stat': round(best_model['ADF Stat'], 1), 'ADF stat sm': round(best_model['ADF Stat sm'], 1),
                                        '1%CV': best_model['1%CV'], '5%CV': best_model['5%CV'], '10%CV': best_model['10%CV'],
                                        'ADF p-value': best_model['ADF P-Value'],
                                        'Cointegrated': best_model['ADF P-Value'] < alpha},
                                    index=[[(y, x) if d == 0 else (y + '_d' + str(d), x + '_d' + str(d))]])

                dfs.append(temp_df)  # Add the temporary dataframe to the list of dataframes

                if best_model['ADF P-Value'] > alpha:
                    continue
                else:
                    break

        df_out = pd.concat(dfs, ignore_index=False)  # Concatenate the dataframes in the list
        return df_out.round(3)
    


class Backtester:
    '''	
    Backtester class for testing cointegrated pairs trading strategy.

    Methods:
    --------
    OU_Process_OLS(y, x, start, end, constant_=True) -> dict:
        Returns a dictionary with Ordinary Least Squares (OLS) regression results for the
        Ornstein-Uhlenbeck (OU) process, and entry/exit points calculated from the AR(1) model.
        
    spread(y, x, start, end, constant_=True) -> DataFrame:
        Returns a DataFrame with the spread calculated from the OLS regression between the input time series y and x.
    '''

    
    def OU_Process_OLS(self, y,x, start, end, constant_=True):
        '''
        Returns OU Process OLS and entry/exit output calculated from AR(1) model using OU process:
        e_t = C + B*e_t_1 + eps_t_tau
        where e_t are the residuals from:
        y_t = constant + beta*x_t + e_t
        
        Params
        ------
        y = Dependent Variable e.g. security price level caused by x
        x = Independent Variable e.g. security price level causing y
        start = start date. Format 'YYYY-MM_DD'
        end = end date. Format 'YYYY-MM_DD' 
        constant = True. Constant present in OU process regression. 
        '''
        constant = True
        y_t = y.loc[start:end].dropna() # 
        x_t = x.loc[start:end].dropna()
        if constant == True:
            x_t = add_constant(x_t) # add intercept = columns of 1s to x_t
        # OLS regression: Static Equilibrium Model
        ols = OLS(y_t, x_t).fit()  # validate result with statsmodels
        res = ols.resid
        # OU SDE Solution Regression: e_t = C + B*et_1 + eps_t_tau
        constant = True
        res_t = res[1:] 
        res_t_1 = res.shift(1).dropna()
        if constant == True:
            x = add_constant(res_t_1) # add intercept = columns of 1s to x_t
        x.rename(columns={0: 'res_t_1'}, inplace=True)
        ols = OLS(res_t, x).fit()
        # Entry/Exit Params:
        mu_e = ols.params[0] / (1-ols.params[1]) # equilibrium level
        tau = 1/(365) # daily data frequency 
        theta = - np.log(ols.params[1])/tau # speed of reversion
        half_l = np.log(2) / theta #  half life
        sigma_OU = np.sqrt( 2* theta * np.var(ols.resid) / (1- np.exp(-2*theta*tau)) ) # diffusion over small time scale (volatility coming from small ups and downs of BM)
        sigma_eq = sigma_OU / np.sqrt(2*theta)# use to determine exit/trading points = mu_e +/- sigma_eq
        # entry/exit points:
        ee_h = mu_e + sigma_eq
        ee_l = mu_e - sigma_eq
        df_out = {'spread':res,'mu_e':[mu_e],'tau':tau,'theta': theta,
                            'sigma_OU':sigma_OU,'sigma_eq':sigma_eq,
                            'ee_high':ee_h,'ee_low':ee_l}
        print('mu_e = ',mu_e)
        print('tau =', tau)
        print('theta = ', theta)
        print('sigma_OU = ', sigma_OU)
        print('sigma_eq = ', sigma_eq)
        print('high/low entry/exit points:', ee_l, ee_h)
        print('#'*50)
        print('############= OU Process Regression ##############')
        df_plot = pd.DataFrame({'res':res,'ee_h':np.repeat(ee_h,len(res)),'ee_l':np.repeat(ee_l,len(res)),
                                'mu_e':np.repeat(mu_e,len(res))}, index=res.index)
        df_plot.plot()
        return df_out
    
    def spread(self, y,x, start, end, constant_=True):
        '''
        Returns spread from regression:
        y = constant + 
        '''
        constant = True
        y_t = y.loc[start:end].dropna() 
        x_t = x.loc[start:end].dropna()
        if constant == True:
            x_t = add_constant(x_t) # add intercept = columns of 1s to x_t
        # OLS regression: Static Equilibrium Model
        ols = OLS(y_t, x_t).fit()  # validate result with statsmodels
        res = ols.resid
        return res
    
 