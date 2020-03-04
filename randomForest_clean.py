import pandas as pd		#處理資料
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier		#隨機森林
from sklearn import preprocessing						#normalize

pd.options.mode.chained_assignment = None		#no pd warning

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Age
title = ["Mr.", "Sir.", "Dr.", "Major.", "Master.", "Ms.", "Miss.", "Mrs.", "Lady."]
title_age = {"Mr.":float(30), "Sir.":float(35), "Dr.":float(35), "Major.":float(50), "Master.":float(5), "Ms.":float(25), "Miss.":float(20), "Mrs.":float(35), "Lady.":float(50)}
cnt = 0
for age in train["Age"] :
	if (math.isnan(age)) :
		for name in title :
			if (train["Name"][cnt].find(name) != -1) :
				train["Age"][cnt] = title_age[name]
	cnt = cnt+1

train.loc[train["Age"] <= 21, "Age"] = 0
train.loc[(train["Age"] > 21) & (train["Age"] <= 30), "Age"] = 1
train.loc[(train["Age"] > 30) & (train["Age"] <= 35), "Age"] = 2
train.loc[train["Age"] > 35 ,"Age"] = 3

cnt = 0
for age in test["Age"] :
	if (math.isnan(age)) :
		for name in title :
			if (test["Name"][cnt].find(name) != -1) :
				test["Age"][cnt] = title_age[name]
	cnt = cnt+1

test.loc[test["Age"] <= 21, "Age"] = 0
test.loc[(test["Age"] > 21) & (test["Age"] <= 30), "Age"] = 1
test.loc[(test["Age"] > 30) & (test["Age"] <= 35), "Age"] = 2
test.loc[test["Age"] > 35, "Age"] = 3

#Sex
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

#Embarked
cnt = 0
for em in train["Embarked"] :
	if (pd.isnull(em)) :
		if (train["Pclass"][cnt] == 1) : 
			train["Embarked"][cnt] = "C"
		else : 
			train["Embarked"][cnt] = "S"
	cnt = cnt+1

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

#Fare
cnt = 0
for f in test["Fare"] :
	if (math.isnan(f)) :
		if (test["Pclass"][cnt] == 1) :
			test["Fare"][cnt] = 60
		elif (test["Pclass"][cnt] == 2) :
			test["Fare"][cnt] = 20
		elif (test["Pclass"][cnt] == 3) :
			test["Fare"][cnt] = 8
	cnt = cnt+1

#Family Size
train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = train["SibSp"] + train["Parch"]

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

RFC = RandomForestClassifier(random_state=3, n_estimators=250,min_samples_split=50,oob_score=True)
RFC.fit(train[predictors], train["Survived"])
print(RFC.oob_score_)

pred = RFC.predict(test[predictors])
submission = pd.DataFrame({
                            "PassengerId": test["PassengerId"],
                            "Survived": pred
                         })
submission.to_csv('submission.csv', index=False)
