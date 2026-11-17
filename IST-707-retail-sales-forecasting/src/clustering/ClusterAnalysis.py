
import pandas as pd
import plotly.express as px


class ClusterAnalysis(object):
    
    # setup analysis object
    def __init__(self):
        
        self.scores = pd.DataFrame()
        
    # add record    
    def addRecord(self,model,data,cluster):
        
        # score model
        score = model.score(data)
        
        # create model info
        model_info = pd.DataFrame({'score':[score],'clusters':[cluster]})
        
        # record scores
        self.scores = pd.concat([model_info,self.scores],sort=False).reset_index(drop=True)
        
    # display results   
    def display(self):
        
        # plot line
        fig = px.line(self.scores, x="clusters", y="score", title='Clustering Experiment Analysis')

        # update figure layout
        fig.update_layout(autosize=False, width=1000, height=400)
        
        # show figure
        fig.show()
        
        

