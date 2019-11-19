import pandas as pd
from pandas_datareader import data
import numpy as np
from datetime import timedelta
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

#File reader function

def file_readr(ticker_list, columns):
    """ Reads in .csv files based on the ticker list provided.
        Returns a dictionary object. """

    my_dict = {}
    for ticker in ticker_list:
        try:
            current_df = pd.read_csv(f'./insiders_{ticker}.csv', header = None, names = columns)
            if current_df.shape[0] > 0:
                my_dict[ticker] = current_df
            else:
                continue
            del current_df
        except:
            continue
    
    return my_dict

#Function for cleaning insider trading data dictionary

def clean_dict(x_dict):
    """ Dummifies Sale and Purchase variables, fixes time/date indexing and more for each item in input dictionary.
        Returns dictionary object. """

    local_dict = {}
    
    for ticker in x_dict.keys():
        df = x_dict[ticker]
        
        if df.shape[0] < 25:
            continue
            
        else:
            
            df.set_index(pd.to_datetime(df.transaction_date), inplace = True)

            df.drop('transaction_date',axis=1, inplace = True)

            transactions_list = [item.split('(')[0] for item in df.transaction_type.tolist()]

            df['sale_num'] = [1 if item == 'Sale' else 0 for item in transactions_list]
            df['buy_num'] = [1 if item == 'Purchase' else 0 for item in transactions_list]

            weekly = df.set_index(df.index-timedelta(days=7)).resample('W-SUN').sum()[['shares_traded','total_price','sale_num','buy_num']]

            weekly.index.name = 'Date'

            local_dict[ticker] = weekly

            print(f'{ticker} cleaned successfully')

    return local_dict
    
#Function to fetch stock data for tickers present in insider trading data dictionary

def get_stocks(x_dict): 
    """ Fetches stock data for each ticker and date range provided in input dictionary.
        Returns a dictionary object. """

    stocks_dict = {}
    
    for ticker in x_dict.keys():
        start_date = str(x_dict[ticker].index[0]).split()[0]
        end_date = str(x_dict[ticker].index[-1]).split()[0]
        
        try:
            stocks_df = data.DataReader(ticker,'yahoo',start_date,end_date)
        except:
            continue
        
        stocks_dict[ticker]=stocks_df
        
    return stocks_dict
        
#Function to clean fetched stock data dictionary

def clean_stocks(x_dict):
    """ Reindexes time/date for each item in stock data dictionary. Calculates standard deviation of weekly return.
        Returns a dictionary object. """

    stocks_dict = {}
    
    for item in x_dict.keys():
        ticker = item
        stocks_df = x_dict[ticker]
        stocks_df['return_std'] = stocks_df['Adj Close'].diff()
        
        stocks_weekly = stocks_df.set_index(stocks_df.index-timedelta(days=7)).resample('W-SUN').std()[['return_std']]
        
        stocks_dict[ticker]=stocks_weekly
        
    return stocks_dict

#Function to merge insider trading data and stocks data dictionaries

def merge_dicts(x,y):
    """ Merges item for item in stock data dictionary and insiders trading data dictionary. Performs some final cleaning including dropping NA values.
        Returns a dictionary object. """

    full_dict = {}
    x_dict = x
    y_dict = y
    
    
    for ticker in x_dict.keys():
        
        if ticker not in y_dict.keys():
            continue
            
        else:
            
            index = [item for item in list(zip(x_dict[ticker].index,y_dict[ticker].index))]
            index_test = all([item[0] == item[1] for item in index])
            
            if index_test:
                index = [item[0] for item in index]
                
                try:
                    
                    left = x_dict[ticker].loc[index]
                    right = y_dict[ticker].loc[index]

                    full_df = pd.merge(left,right, on=left.index)

                    full_df = full_df.set_index('key_0')

                    full_df.index.name = 'Date'

                    full_df.sale_num[full_df.sale_num > 0] = 1

                    full_df.buy_num[full_df.buy_num>0] = 1

                    full_df['risk_dummy'] = np.where(full_df.return_std > 1.5, 1, 0)

                    full_df = full_df.dropna()
                    
                    full_df = full_df.drop(['shares_traded','total_price'], axis = 1)
                    
                    full_dict[ticker] = full_df


                except:
                    print(f'error on ticker:{ticker}')
            
    return full_dict
        
#Function to plot a sample of stocks adjusted close price over time with vertical lines indicating insider trades

def plot_dict(insider_dict,stocks_dict,choose = 10):
    """ Chooses a random sample of stocks. Plots Adjusted Close price for chosen sample over entire date range and places vertical lines indicating insider trading activity. """

    fig = plt.figure(figsize=(20,15))
    
    tickers = np.random.choice(list(insider_dict.keys()),choose, replace=False)
    
    for ticker in tickers:
        
        if ticker not in stocks_dict.keys():
            
            continue
            
        else:
            
            stock_df = stocks_dict[ticker]
            insider_df = insider_dict[ticker]
            
            sns.lineplot(x=stock_df.index,y=stock_df['Adj Close']).set_title('Adjusted Close with Insider Trading Vertical Indicators')
            for x in insider_df.index.tolist():
                plt.axvline(x,linestyle=':',linewidth=0.1,c='g')
                    
#Function to create a dataframe indicating the count of low and high risk observations per ticker plus their totals

def create_class_balance_df(x_dictionary, label_col, columns):
    """ Calculates the number of high risk and low risk occurences as well as totals for each ticker in input dictionary.
        Returns a pandas DataFrame object. *!Check for NaNs!* """

    ticker_dict = {}

    for ticker in x_dictionary.keys():
        key_val_dict = {}
        for key,value in x_dictionary[ticker][label_col].value_counts().items():
            key_val_dict[key] = value
        ticker_dict[ticker] = key_val_dict
        
    
    
    ticker_list = list(ticker_dict.keys())
    zeros_label = x_dictionary[ticker][label_col].unique()[0]
    ones_label = x_dictionary[ticker][label_col].unique()[1]
    zeros_list = [item.get(zeros_label) for item in ticker_dict.values()]
    ones_list = [item.get(ones_label) for item in ticker_dict.values()]
    
    class_balance_df = pd.DataFrame(list(zip(ticker_list,zeros_list,ones_list)), columns = columns)
    class_balance_df.set_index('Ticker')
    class_balance_df['Sample_Size'] = np.where(pd.isna(class_balance_df.Low_Risk + class_balance_df.High_Risk) != True,class_balance_df.Low_Risk + class_balance_df.High_Risk, class_balance_df.Low_Risk) 
    class_balance_df['Low_Risk_Pct'] = class_balance_df.Low_Risk / class_balance_df.Sample_Size * 100
    class_balance_df['High_Risk_Pct'] = class_balance_df.High_Risk / class_balance_df.Sample_Size * 100
    
    return class_balance_df

#Function to fit model to data for each ticker and return as dictionary

def model_fit(x_dict, model_list,**kwargs):
    """ Fits model to each item in input dictionary.
        Returns dictionary object. """
        
    rf_dict = {}
    gbm_dict = {}
    svm_dict = {}
    
    for model in model_list:

        for ticker in x_dict.keys():
            
            df = x_dict[ticker]
                        
            X = np.array(df[['sale_num','buy_num']])
            Y = np.array(df['risk_dummy'])
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = kwargs['test_size'], stratify = Y)
            class_counts = np.unique(y_test,return_counts=True)
            baseline = max(class_counts[1])/sum(class_counts[1])
            sample_weights = compute_sample_weight('balanced',y_train)

            if model == 'rf':
                rf = ensemble.RandomForestClassifier(n_jobs = -1, n_estimators = kwargs['n_estimators'], class_weight  = kwargs['class_weight'])
                rf.fit(x_train, y_train, sample_weights)
                prediction = rf.predict(x_test)
                accuracy = accuracy_score(y_test, prediction)
                roc = roc_auc_score(y_test, prediction)
                rf_dict[ticker] = {'prediction':prediction, 'accuracy':accuracy, 'baseline':baseline, 'auc':roc}

            elif model == 'gbm':
                gbm = ensemble.GradientBoostingClassifier(max_features='auto', n_estimators = kwargs['n_estimators'])
                gbm.fit(x_train, y_train, sample_weights)
                prediction = gbm.predict(x_test)
                accuracy = accuracy_score(y_test, prediction)
                roc = roc_auc_score(y_test, prediction)
                gbm_dict[ticker] = {'prediction':prediction, 'accuracy':accuracy, 'baseline':baseline, 'auc':roc}

            elif model == 'svm':
                svm_fit = svm.SVC(class_weight = kwargs['class_weight'], gamma = 'auto')
                param_grid  = {'C':np.arange(1,10), 'degree':np.arange(1,10), 'kernel':['poly','sigmoid','rbf']}
                svm_fit = GridSearchCV(svm_fit, param_grid, cv = 5, n_jobs = -1, iid = False)
                svm_fit.fit(x_train, y_train)
                prediction = svm_fit.best_estimator_.predict(x_test)
                accuracy = accuracy_score(y_test, prediction)
                roc = roc_auc_score(y_test, prediction)
                svm_dict[ticker] = {'prediction':prediction, 'accuracy':accuracy, 'baseline':baseline, 'auc':roc}

    models_dict = {'rf':rf_dict, 'gbm':gbm_dict, 'svm':svm_dict}

    return models_dict



        