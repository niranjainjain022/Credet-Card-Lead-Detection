#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
# Print versions of libraries
print(f"Numpy version : Numpy {np.__version__}")
print(f"Pandas version : Pandas {pd.__version__}")
print(f"Seaborn version : Seaborn {sns.__version__}")
print(f"SkLearn version : SkLearn {sklearn.__version__}")
# Magic Functions for In-Notebook Display
%matplotlib inline
# Setting seabon style
sns.set(style='darkgrid', palette='colorblind')

#GETTING THE DATA
train = pd.read_csv('train.csv', encoding='latin_1')
# Converting all column names to lower case
train.columns = train.columns.str.lower()
 
test = pd.read_csv('test.csv', encoding='latin_1')
# Converting all column names to lower case
test.columns = test.columns.str.lower()
test.head()
print("")
print("SOME INSIGHTS ON WHAT KIND OF DATA ARE WE WORKING ON")
print("")
print("Information about the training data")
print("")
print(train.info())
print("")
print("Decription of the continuous features of training data")
print("")
print(train.describe())
print("")
print("")
print("")
print("Information about the testing data")
print("")
print(test.info())
print("")
print("Decription of the continuous features of testing data")
print("")
print(test.describe())
print("")
def training(train, test):
    #assign a third category to the missing data
    train['credit_product'].fillna('notassigned', inplace=True)
    test['credit_product'].fillna('notassigned', inplace=True)
    #Assigning range to the continuous data in both training and testing
    age = test['age'].values
    nage = []
    vintage = test['vintage'].values
    nvintage = []
    avg_account_balance = test['avg_account_balance'].values
    navg_account_balance = []
    for i in age:
      i = int(i)
      if i>=23 and i<=30:
        nage.append('agerange1')
      if i>30 and i<=43:
        nage.append('agerange2')
      if i>43 and i<=54:
        nage.append('agerange3')
      if i>54 and i<=85:
        nage.append('agerange4')
      if i>85:
        nage.append('agerange5')
    for j in vintage:
      j = int(j)
      if j>=7 and j<=20:
        nvintage.append('vintageagerange1')
      if j>20 and j<=32:
        nvintage.append('vintageagerange2')
      if j>32 and j<=73:
        nvintage.append('vintageagerange3')
      if j>73 and j<=135:
        nvintage.append('vintageagerange4')
      if j>135:
        nvintage.append('vintageagerange5')
    for k in avg_account_balance:
      k = int(k)
      if k>=20790 and k<=604310:
        v = 'balancerange1'
        navg_account_balance.append(v)
      if k>604310 and k<=894601:
        v = 'balancerange2'
        navg_account_balance.append(v)
      if k>894601 and k<=1366666:
        v = 'balancerange3'
        navg_account_balance.append(v)
      if k>1366666 and k<=10352009:
        v = 'balancerange4'
        navg_account_balance.append(v)
      if k>10352009:
        v = 'balancerange5'
        navg_account_balance.append(v)
    test['age']=nage
    test['vintage']=nvintage
    test['avg_account_balance']=navg_account_balance
    age = train['age'].values
    nage = []
    vintage = train['vintage'].values
    nvintage = []
    avg_account_balance = train['avg_account_balance'].values
    navg_account_balance = []
    for i in age:
      i = int(i)
      if i>=23 and i<=30:
        nage.append('agerange1')
      if i>30 and i<=43:
        nage.append('agerange2')
      if i>43 and i<=54:
        nage.append('agerange3')
      if i>54 and i<=85:
        nage.append('agerange4')
      if i>85:
        nage.append('agerange5')
    for j in vintage:
      j = int(j)
      if j>=7 and j<=20:
        nvintage.append('vintageagerange1')
      if j>20 and j<=32:
        nvintage.append('vintageagerange2')
      if j>32 and j<=73:
        nvintage.append('vintageagerange3')
      if j>73 and j<=135:
        nvintage.append('vintageagerange4')
      if j>135:
        nvintage.append('vintageagerange5')
    for k in avg_account_balance:
      k = int(k)
      if k>=20790 and k<=604310:
        v = 'balancerange1'
        navg_account_balance.append(v)
      if k>604310 and k<=894601:
        v = 'balancerange2'
        navg_account_balance.append(v)
      if k>894601 and k<=1366666:
        v = 'balancerange3'
        navg_account_balance.append(v)
      if k>1366666 and k<=10352009:
        v = 'balancerange4'
        navg_account_balance.append(v)
      if k>10352009:
        v = 'balancerange5'
        navg_account_balance.append(v)
    train['age']=nage
    train['vintage']=nvintage
    train['avg_account_balance']=navg_account_balance
    #Label Encoding our data.
    le = LabelEncoder()
    for x in train:
        if train[x].dtypes=='object':
            train[x] = le.fit_transform(train[x].astype(str))
    #Dropping some columns with low accuracy
    train.drop('id', axis = 1, inplace=True) 
    train.drop('region_code', axis = 1, inplace=True) 
    #Now lets start building our model.
    X = train.iloc[:,:-1] # X value contains all the variables except labels and ID
    y = train.iloc[:,-1] # these are the labels

    # We create the test train split first
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

    #We have now fit and transform the data into a scaler for accurate reading and results.
    mms = MinMaxScaler()
    X_scaled = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(mms.transform(X_test), columns=X_test.columns)

    #Now we carryout oversampling to adjust the class distribution of a data set
    oversample = SMOTE()
    X_balanced, y_balanced = oversample.fit_resample(X_scaled, y_train)
    X_test_balanced, y_test_balanced = oversample.fit_resample(X_test_scaled, y_test)
    '''
    To get the best hyper-parameters for our model we are going to use GridSearchCV. Just uncomment the code below to get the best hyper
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.model_selection import GridSearchCV
    # params = {'max_depth':list(range(0,30)),
    #           'criterion' : ["gini", "entropy"],
    #           'max_features' : ["int","float","None", "auto", "sqrt", "log2"]
    #          }
    # grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params, verbose=1, cv=3)
    # grid_search_cv.fit(X_balanced, y_balanced)
    # model = grid_search_cv.best_estimator_
    # We will now train Decision tree model on the data.
    '''
    print("")
    print("Training Model")
    print("")
    import time #just to check how much time it takes to train
    train_scores = []
    test_scores = []
    tic = time.perf_counter()
    model = DecisionTreeClassifier(criterion='entropy', max_depth=27, max_features='log2')
    model.fit(X_balanced, y_balanced)
    toc = time.perf_counter() #time ends here
    print("it took {tt} seconds".format(tt=tic-toc))
    model.fit(X_balanced, y_balanced)
    toc = time.perf_counter() #time ends here
    print("it took {tt} seconds".format(tt=tic-toc))
    
    #Train Accuracy
    from sklearn.metrics import roc_auc_score
    print("TRAINING ACCURACY: ", roc_auc_score(y_balanced, model.predict_proba(X_balanced)[:, 1]))
    
    #Test Accuracy
    from sklearn.metrics import roc_auc_score
    print("TESTING ACCURACY: ", roc_auc_score(y_test_balanced, model.predict_proba(X_test_balanced)[:, 1]))

    # Save the model as a pickle in a file
    return model

# MAKING A FUNCTION FOR PREDICTION
def prediction(test, model): #enter the file as a data frame'
    print("")
    print("PROCESSING THE DATA TO BE PREDICTED")
    IDlite = np.array(test['id'])
    le = LabelEncoder()
    test.drop('region_code', axis = 1, inplace=True) 
    for x in test:
        if test[x].dtypes=='object':
            test[x] = le.fit_transform(test[x].astype(str))
    test = test.iloc[:,1:] # X value contains all the variables except ID
    mms = MinMaxScaler()
    test = pd.DataFrame(mms.fit_transform(test), columns=test.columns)
    test = np.array(test)
    # Use the loaded model to make predictions
    a = model.predict(test)
    print("Completed.")
    return IDlite, a

#training the model and getting the predictions
model = training(train, test)
ID, Is_Lead = prediction(test, model) #getting ID and Is_Lead

#making a data frame
import pandas as pd
import numpy as np
dataset = pd.DataFrame({'ID': ID, 'Is_Lead': list(Is_Lead)}, columns=['ID', 'Is_Lead'])

#saving the data frame as csv
dataset.to_csv(path_or_buf='Solution.csv', sep=',', index=None)