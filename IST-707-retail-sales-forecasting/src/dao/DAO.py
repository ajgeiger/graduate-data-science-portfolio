import pandas as pd, numpy as np

class DAO(object):
    '''
    code inspired by the following kaggle notebooks
    https://www.kaggle.com/kernelgenerator/m5-accuracy-tweedie-is-back
    https://www.kaggle.com/kyakovlev/m5-custom-features
    https://www.kaggle.com/kyakovlev/m5-lags-features
    https://www.kaggle.com/kyakovlev/m5-simple-fe
    '''

    # get dataset
    def getData(self,is_train = True, nrows = None):

        # get categorical columns to setup composite key
        catcols = self.getCatColumns()

        # get price data
        prices = self.getSellPrices()
        
        # get calendar data   
        cal = self.getCalendarData()
        
        # get targets
        dt = self.getTargets(is_train,nrows)
        
        # get value variables
        value_vars = [col for col in dt.columns if col.startswith("d_")]

        dt = pd.melt(dt,
                      id_vars = catcols,
                      value_vars = value_vars,
                      var_name = "d",
                      value_name = "sales")
    
        # merge calendar data
        dt = dt.merge(cal, on= "d", copy = False)

        # merge price data
        dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
        # return data
        return dt


    # get calendar data
    def getCalendarData(self):

        data_types ={"event_name_1": "category", 
                     "event_name_2": "category",
                    "event_type_1": "category", 
                    "event_type_2": "category",
                    "weekday": "category",
                    'wm_yr_wk': 'int16', 
                    "wday": "int16",
                    "month": "int16",
                    "year": "int16",
                    "snap_CA": "float32",
                    'snap_TX': 'float32', 
                    'snap_WI': 'float32'}

        # define file path and file name
        file = self.path+"/calendar.csv"

        # import calendar data
        cal = pd.read_csv(file, dtype = data_types)

        # reduce memory
        cal = self.__reduceCategoryMemory(cal)

        return cal


    # get sell prices
    def getSellPrices(self):

        # prices data types
        data_types = {"store_id": "category",
                      "item_id":   "category",
                      "wm_yr_wk":   "int16",
                      "sell_price": "float32" }

        # define file path and file name
        file = self.path+"/sell_prices.csv"


        # import price data
        prices = pd.read_csv(file, dtype = data_types)

        # transform categorical values to int to save memory
        prices = self.__reduceCategoryMemory(prices)

        # return prices
        return prices
    

    # get model targets
    def getTargets(self,is_train,nrows):

        # get categorical columns
        catcols = self.getCatColumns()
        

        # define file path and file name
        file = self.path+"/sales_train_validation.csv"


        # get number columns
        numcols = self.__getNumCols(is_train)

        
        # define data types
        dtype = {numcol:"float32" for numcol in numcols} 


        # update dtypes
        dtype.update({col: "category" for col in catcols if col != "id"})


        # get target data set
        dt = pd.read_csv(file, nrows = nrows, usecols = catcols + numcols, dtype = dtype)

        # minimize memory
        for col in catcols:
            if col != "id":
                dt[col] = dt[col].cat.codes.astype("int16")
                dt[col] -= dt[col].min()
    
        # if dataset is test add dummy rows
        if not is_train:
            for day in range(self.lastKnownDay+1, self.lastKnownDay+ 28 +1):
                dt[f"d_{day}"] = np.nan

        return dt


    # reduce memory
    def __reduceCategoryMemory(self,df):

        # transform categorical values to int to save memory
        for col, col_dtype in df.dtypes.items():

            try:
                logic = col_dtype == "category"
            
            except Exception as e:

                # if verbose is on print exception
                if self.verbose:
                    print(str(e),col_dtype)
                
                # set default logic to false
                logic = False


            if logic:
                df[col] = df[col].cat.codes.astype("int16")
                df[col] -= df[col].min()

        return df


    # get categorical columns
    def getCatColumns(self):

        return ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']


    # get number columns
    def __getNumCols(self,is_train):

        # calculate start day
        start_day = max(1 if is_train  else self.tr_last - self.max_lags, self.first_day)


        # get number of columns
        numcols = [f"d_{day}" for day in range(start_day,self.tr_last+1)]

        # return forecast columns
        return numcols