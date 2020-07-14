import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
import pickle


match_results=pd.read_csv('PLdataset.csv')

PL_matchresults=match_results[['HomeTeam','AwayTeam','HomeTeam_Prob','AwayTeam_Prob','HomeTeam_Goals','AwayTeam_Goals','Result']]

winner=[]
for i in range (len(match_results['HomeTeam'])):
    if PL_matchresults['HomeTeam_Goals'][i] > PL_matchresults['AwayTeam_Goals'][i]:
        winner.append(PL_matchresults['HomeTeam'][i])
    elif PL_matchresults['HomeTeam_Goals'][i] < PL_matchresults ['AwayTeam_Goals'][i]:
        winner.append(PL_matchresults['AwayTeam'][i])
    else:
        winner.append('Draw')
PL_matchresults['Winner'] = winner

PL_matchresults.drop('Result',axis=1,inplace=True)


PremierLeague_Teams=['Arsenal','Bournemouth','Burnley','Brighton',
                    'Cardiff','Chelsea','Crystal Palace','Everton','Fulham',
                    'Huddersfield','Leicester','Liverpool','Man City','Man United',
                    'Newcastle','Southampton','Tottenham','Watford','West Ham',
                    
                    'Wolves','Bradford','Middlesbrough','Charlton','Leeds','Aston Villa','Sunderland','Derby']


PL_matchresults=PL_matchresults[PL_matchresults['HomeTeam'].isin(PremierLeague_Teams) & PL_matchresults['AwayTeam'].isin(PremierLeague_Teams)]

PL_matchresults.reset_index(inplace=True)

PL_matchresults.drop('index',inplace=True,axis=1)

PL_matchresults.drop_duplicates(keep='first',inplace=True)

matchresults_PL=PL_matchresults.drop('Winner',axis=1)

X=matchresults_PL.iloc[: , 0:4]

y=matchresults_PL.iloc[: ,4:]

X=pd.get_dummies(X,columns=['HomeTeam','AwayTeam'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn_regressor=neighbors.KNeighborsRegressor(n_neighbors =50)

knn_regressor.fit(X_train,y_train)
# predictions=knn_regressor.predict(X_test)

# predictions=predictions.round()

# print(predictions[70])


filename='Premier-League-Prediction-model.pkl'
pickle.dump(knn_regressor, open(filename, 'wb'))


