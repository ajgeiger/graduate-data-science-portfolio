import pandas as pd, numpy as np
from src.dao.DAO import DAO
import numpy as np
from  datetime import datetime, timedelta

# pre-process data
class Pipeline(DAO):
    '''
    code inspired by the following kaggle notebooks
    https://www.kaggle.com/kernelgenerator/m5-accuracy-tweedie-is-back
    https://www.kaggle.com/kyakovlev/m5-custom-features
    https://www.kaggle.com/kyakovlev/m5-lags-features
    https://www.kaggle.com/kyakovlev/m5-simple-fe
    '''


    # init object method
    def __init__(self,path,
                 first_day=1000,
                 tr_last = 1913,
                 verbose = True):

        self.path      = path
        self.first_day = first_day
        self.tr_last   = tr_last
        self.verbose   = verbose
        self.startDate = datetime(2016,4, 25) 
        self.max_lags  = 57
        self.start =1 
        self.end = 29
        self.lastKnownDay  = 1913
   
    # build test data
    def buildTestData(self):

        # get feature set
        df = self.getData(is_train = False)
        
        # to date time
        df['date'] = pd.to_datetime(df.date)
        
        # return datetime
        return df

    #build test features
    def buildTestFeatures(self,df,tdelta):

        # calculate date
        day = self.startDate + timedelta(days=tdelta)

        # get slice of dataframe
        tst = df[(df.date >= day - timedelta(days=self.max_lags)) & (df.date <= day)].copy()
        
        # add features
        tst = self.__addFeatures(tst)
        
        # return data
        return tst.loc[tst.date == day].copy(), day

    # build feature set
    def buildFeatureSet(self):

        # get feature set
        dt = self.getData(is_train = True)
        
        # return dataframe
        return self.__addFeatures(dt)
    
    # build classification dataset
    def buildClassificationFeatureSet(self):

        # build feature set
        df = self.buildFeatureSet()

        # add group
        group = ['store_id','id','week','year']

        # get best day were
        df['id'] = df['id'].apply(lambda x: x.split('_')[0])

        # calculate best week as highest sales
        df['best_day'] = (df.groupby(group)['sales'].transform('max') ==df['sales'])*(df['wday'] +1)

        # calculate best week day 
        df['best_day'] = df.groupby(group)['best_day'].transform('max')

        # calculate best week day
        df = df.loc[df.groupby(group)['date'].transform('min') == df['date']].copy()

        # return dataframe
        return df


    # FEATURE METHODS
    #----------------

    # build general features
    def buildGeneralFeatures(self,df):

        # Some items are can be inflation dependent
        df['price_nunique_items'] = df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')

        # and some items are very "stable"
        df['item_nunique_price'] = df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

        # and some items are very "stable"
        df['item_nunique_weekly'] = df.groupby(['store_id','week'])['item_id'].transform('nunique')
    
        icols =  [['item_id']]

        for col in icols:
            print('Encoding', col)
            col_name = '_'+'_'.join(col)+'_'
            df['enc'+col_name+'mean'] = df.groupby(col)['sales'].transform('mean').astype(np.float16)
            df['enc'+col_name+'std'] = df.groupby(col)['sales'].transform('std').astype(np.float16)

    # build roling window features
    def buildRollingWindowFeatures(self,dt):

        # run rolling window features
        lag_cols = [f"lag_{lag}" for lag in self.getLags()]
        for win in self.getWindows():
            for lag,lag_col in zip(self.getLags(), lag_cols):
                dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    # add lag feature
    def buildLagFeature(self,dt):

        lag_cols = [f"lag_{lag}" for lag in self.getLags()]
        for lag, lag_col in zip(self.getLags(), lag_cols):
            dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
    
    # build dataset
    def buildDateFeatures(self,dt):

        # transform data to datetime
        dt["date"] = pd.to_datetime(dt["date"])

        # get date features
        date_features = self.getDateFeatures()
    
        # build date features
        for date_feat_name, date_feat_func in date_features.items():

            # check if date feature name is in column
            if date_feat_name in dt.columns:

                # if so transform to an int 16 to save memory
                dt[date_feat_name] = dt[date_feat_name].astype("int16")
            else:

                # if date not present calcuate date 
                dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")



    # TEST METHODS
    #--------------
     # create test submission
    def createSubmission(self,te):

        te_sub = te.loc[te.date >= self.startDate, ["id", "sales"]].copy()
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
        te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][self.forecastDays()].reset_index()
        te_sub.fillna(0., inplace = True)
        te_sub.sort_values("id", inplace = True)
        te_sub.reset_index(drop=True, inplace = True)

        return te_sub

    # forecast number of days
    def forecastDays(self):

        return [f"F{i}" for i in range(self.start,self.end)]


    # PRIVATE METHODS
    # ----------------

    # add features
    def __addFeatures(self,dt):
        
        # build lag features
        self.buildLagFeature(dt)

        # build rolling window features
        self.buildRollingWindowFeatures(dt)
        
        # get date features
        self.buildDateFeatures(dt)

        # add general features
        self.buildGeneralFeatures(dt)

        return dt



    @staticmethod
    def getWindows():

        return [7, 28]
    
    @staticmethod
    def getLags():

        return [7, 28]

    @staticmethod
    def getDateFeatures():

        return {"wday":    "weekday",
                "week":    "weekofyear",
                "month":   "month",
                "quarter": "quarter",
                "year":    "year",
                "mday":    "day"}