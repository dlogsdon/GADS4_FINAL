# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
from sklearn.metrics import roc_curve, auc, auc_score

# <codecell>

#####Bring in primary 2 databases. 
#PV_Master is the list of customers who have installed solar.
PV_MASTER = pd.read_csv("/Users/davidlogsdon/data_science/Final Project/PV_master.csv")
#ROOFTOP_POTENTIALS is a database of all customers with large roofs who may or may not have installed solar. 
ROOFTOP_POTENTIALS = pd.read_csv("/Users/davidlogsdon/data_science/Final Project/rooftop_for_merge.csv")
#modify account no. column name for later joining.
PV_MASTER.rename(columns={"Account No": "Acct_No"}, inplace=True)
#This new 'Has_PV' column will be used later to identify which customers have already installed solar. 
ROOFTOP_POTENTIALS['Has_PV']=0
ROOFTOP_POTENTIALS.fillna(0,inplace=True)

# <codecell>

#######DATA MUNGING...
##Acount numbers are either 15 or 14 digits in either database, going to 14 digits to standardize for later joining
PV_MASTER['Acct_No'] = PV_MASTER['Acct_No'].map( lambda x : str(x)[0:14])
ROOFTOP_POTENTIALS['Acct_No'] = ROOFTOP_POTENTIALS['Acct_No'].map(lambda x : str(x)[0:14])

##Boro refers to the NYC Borrough (or WS for Westchester)
ROOFTOP_POTENTIALS['Boro_QN']= ROOFTOP_POTENTIALS['Boro'].map(lambda x: 1 if (x=='QN') else 0)
ROOFTOP_POTENTIALS['Boro_BX']= ROOFTOP_POTENTIALS['Boro'].map(lambda x: 1 if (x=='BX') else 0)
ROOFTOP_POTENTIALS['Boro_MN']= ROOFTOP_POTENTIALS['Boro'].map(lambda x: 1 if (x=='MN') else 0)
ROOFTOP_POTENTIALS['Boro_SI']= ROOFTOP_POTENTIALS['Boro'].map(lambda x: 1 if (x=='SI') else 0)
ROOFTOP_POTENTIALS['Boro_WS']= ROOFTOP_POTENTIALS['Boro'].map(lambda x: 1 if (x=='WS') else 0)

##Zonal Data for NYISO electricity market zone id. 
ROOFTOP_POTENTIALS['Zone_H'] = 0
ROOFTOP_POTENTIALS['Zone_H'] = ROOFTOP_POTENTIALS['Zone'].map(lambda x: 1 if (x=='H') else 0)
ROOFTOP_POTENTIALS['Zone_H'] = ROOFTOP_POTENTIALS['Zone_H'].map(lambda x: int(x))
ROOFTOP_POTENTIALS['Zone_I'] = 0
ROOFTOP_POTENTIALS['Zone_I'] = ROOFTOP_POTENTIALS['Zone'].map(lambda x: 1 if (x=='I') else 0)
ROOFTOP_POTENTIALS['Zone_I'] = ROOFTOP_POTENTIALS['Zone_I'].map(lambda x: int(x))
ROOFTOP_POTENTIALS['Zone_J'] = 0
ROOFTOP_POTENTIALS['Zone_J'] = ROOFTOP_POTENTIALS['Zone'].map(lambda x: 1 if (x=='J') else 0)
ROOFTOP_POTENTIALS['Zone_J'] = ROOFTOP_POTENTIALS['Zone_J'].map(lambda x: int(x))

##Some fomatting on the entries to smooth them out for the logit model.
ROOFTOP_POTENTIALS['Max_kW'] = ROOFTOP_POTENTIALS['Max_kW'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['Max_kW'] = ROOFTOP_POTENTIALS['Max_kW'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['Max_kW'] = ROOFTOP_POTENTIALS['Max_kW'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['min load'] = ROOFTOP_POTENTIALS['min load'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['min load'] = ROOFTOP_POTENTIALS['min load'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['min load'] = ROOFTOP_POTENTIALS['min load'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['LF'] = ROOFTOP_POTENTIALS['LF'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['LF'] = ROOFTOP_POTENTIALS['LF'].astype(float)
ROOFTOP_POTENTIALS['roof capacity'] = ROOFTOP_POTENTIALS['roof capacity'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['roof capacity'] = ROOFTOP_POTENTIALS['roof capacity'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['roof capacity'] = ROOFTOP_POTENTIALS['roof capacity'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['Yr_kWh'] = ROOFTOP_POTENTIALS['Yr_kWh'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['Yr_kWh'] = ROOFTOP_POTENTIALS['Yr_kWh'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['Yr_kWh'] = ROOFTOP_POTENTIALS['Yr_kWh'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['BldgArea'] = ROOFTOP_POTENTIALS['BldgArea'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['BldgArea'] = ROOFTOP_POTENTIALS['BldgArea'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['BldgArea'] = ROOFTOP_POTENTIALS['BldgArea'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['BldgFront'] = ROOFTOP_POTENTIALS['BldgFront'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['BldgFront'] = ROOFTOP_POTENTIALS['BldgFront'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['BldgFront'] = ROOFTOP_POTENTIALS['BldgFront'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['BldgDepth'] = ROOFTOP_POTENTIALS['BldgDepth'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['BldgDepth'] = ROOFTOP_POTENTIALS['BldgDepth'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['BldgDepth'] = ROOFTOP_POTENTIALS['BldgDepth'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['AreaPerFloor'] = ROOFTOP_POTENTIALS['AreaPerFloor'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['AreaPerFloor'] = ROOFTOP_POTENTIALS['AreaPerFloor'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['AreaPerFloor'] = ROOFTOP_POTENTIALS['AreaPerFloor'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['kw export'] = ROOFTOP_POTENTIALS['kw export'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['kw export'] = ROOFTOP_POTENTIALS['kw export'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['kw export'] = ROOFTOP_POTENTIALS['kw export'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['FrontTimesDepth'] = ROOFTOP_POTENTIALS['FrontTimesDepth'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['FrontTimesDepth'] = ROOFTOP_POTENTIALS['FrontTimesDepth'].map(lambda x : str(x).replace(',',''))
ROOFTOP_POTENTIALS['FrontTimesDepth'] = ROOFTOP_POTENTIALS['FrontTimesDepth'].map(lambda x : float(x))
ROOFTOP_POTENTIALS['Ntwk'] = ROOFTOP_POTENTIALS['Ntwk'].replace(['#DIV/0!', np.nan], 0)
ROOFTOP_POTENTIALS['Ntwk'] = ROOFTOP_POTENTIALS['Ntwk'].map(lambda x : str(x))


# <codecell>

##Merge the two databases on account number.
#new_data will contain only entries that both have PV and have account numbers appearing in ROOFTOP_POTENTIALS.
new_data = pd.merge(ROOFTOP_POTENTIALS, PV_MASTER, on='Acct_No')

# <codecell>

###Might come back to this someday, need a way to incorporate geography.
#ROOFTOP_POTENTIALS['Ntwk'].value_counts().index[:40]
#dummy_ranks = pd.get_dummies(ROOFTOP_POTENTIALS['Ntwk'], prefix='Network')
#ROOFTOP_POTENTIALS = ROOFTOP_POTENTIALS.join(dummy_ranks)

# <codecell>

###Loop through ROOFTOP_POTENTIALS and set Has_PV = 1 if the Account number appears in the new, merged data frame. 
#note: there's probably a more elegant way to do this. 
n=0
for i in ROOFTOP_POTENTIALS['Acct_No']:
    n+=1
    for j in new_data['Acct_No']:
        if i == j:
            ROOFTOP_POTENTIALS['Has_PV'][n] = 1

# <codecell>

###Create the train and test datasets.
#sorted_roofs puts all the roofs having solar at the front of the database for splitting.
sorted_roofs = ROOFTOP_POTENTIALS.sort('Has_PV', ascending = False)

#full_train gets all 163 roofs that we know have installed PV as well as 163 roofs that have not. 
full_train = sorted_roofs[0:326]
full_data = sorted_roofs[327:]

###For cross validation, divide this full train down to a train and test set. 
#The training set contains 106 of the rooftops have have installed PV (65% of the total) and 106 rooftops that have not. 
train = full_train[0:106]
train = train.append(full_train[164:269])
#The test set contains 56 roofs that have installed PV (35% of the total) and 56 that have not.
test = full_train[107:163]
test = test.append(full_train[269:])

#If we were to use the final model to make predictions we would apply it to full_data 

# <codecell>

ROOFTOP_POTENTIALS

# <codecell>

##First, try the model with ALL possible inputs. Leaving out BLoack and lot and lat/long, network, and M&S plate as they are not binned widely enough. Need to consider geography more next time.
#train_cols = pd.DataFrame(train, columns=['Yr_kWh','Max_kW', 'LF', 'min load','kw export', 'NumBldgs', 'BldgArea', 'NumFloors', 'BldgFront', 'BldgDepth', 'AreaPerFloor', 'roof capacity', 'Zone_H', 'Zone_I', 'Zone_J','Boro_QN', 'Boro_BX', 'Boro_MN', 'Boro_SI', 'Boro_WS', 'Cis_Lat', 'Cis_Long', 'Block','Lot'])
#test_cols = pd.DataFrame(test, columns=['Yr_kWh','Max_kW', 'LF', 'min load','kw export','NumBldgs', 'BldgArea', 'NumFloors', 'BldgFront', 'BldgDepth', 'AreaPerFloor', 'roof capacity', 'Zone_H', 'Zone_I', 'Zone_J','Boro_QN', 'Boro_BX', 'Boro_MN', 'Boro_SI', 'Boro_WS','Cis_lat', 'Cis_Long', 'Block','Lot'])

##Screen it down to only those with p values < .1
train_cols = pd.DataFrame(train, columns=['Yr_kWh', 'LF', 'BldgArea', 'NumFloors', 'BldgDepth'])
test_cols = pd.DataFrame(test, columns=['Yr_kWh','LF', 'BldgArea', 'NumFloors', 'BldgDepth'])


# <codecell>

#clf = LogisticRegression(C=.8).fit(train['Has_PV'],train['LF'])
logit = sm.Logit(train['Has_PV'].astype(int), train_cols)
#logit = sm.Logit(train['Has_PV'].astype(int), Network_array.T)

# <codecell>

result = logit.fit()

# <codecell>

print result.summary()

# <codecell>


predictions = result.predict(np.asmatrix(test_cols.astype(float)))

# <codecell>

predictions

# <codecell>

actuals = np.asmatrix(test['Has_PV'].astype(float))
actuals

# <codecell>

fpr, tpr, thresholds = roc_curve(actuals, predictions)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve, Logit : %f" % roc_auc

# <codecell>

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


