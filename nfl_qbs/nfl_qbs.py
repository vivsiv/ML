import os
import csv
import numpy as np
import sklearn as sk
import sklearn.cluster
import sklearn.ensemble
from sklearn.feature_selection import SelectKBest, f_classif
import pandas
import re
import matplotlib.pyplot as plt

def is_hof(name):
	star_search = re.search('\*',name)
	if star_search:
		return 1
	else:
		return 0

inactive_qbs = pandas.read_csv("inactive_qbs.csv")
inactive_qbs["TD/Int"] = inactive_qbs["TD"] / inactive_qbs["Int"]
inactive_qbs["WinPct"] = inactive_qbs["W"] / inactive_qbs["GS"]
inactive_qbs["HOF"] = inactive_qbs["Name"].apply(is_hof)

#Select K Best Factors
# predictors = ["CmpPct", "IntPct", "Y/A", "AY/A", "ANY/A", "TD/Int", "Rate", "Y/G"]
# selector = SelectKBest(f_classif, k=5)
# selector.fit(inactive_qbs[predictors], inactive_qbs["HOF"])
# scores = -np.log10(selector.pvalues_)

# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

predictors = ["Y/A", "AY/A", "ANY/A", "WinPct", "Rate", "Y/G"]
model = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
scores = sk.cross_validation.cross_val_score(model, inactive_qbs[predictors], inactive_qbs["HOF"], cv=5)
print("Average Score over 5 folds is:", scores.mean())
model.fit(inactive_qbs[predictors], inactive_qbs["HOF"])

# preds = model.predict((inactive_qbs[predictors]).head(25))
# print((inactive_qbs["HOF"]).head(25),preds)
#print(inactive_qbs["Y/A"].median())
#print(active_qbs["Y/A"].median())

active_qbs = pandas.read_csv("active_qbs.csv")
active_qbs["TD/Int"] = active_qbs["TD"] / active_qbs["Int"]
active_qbs["WinPct"] = active_qbs["W"] / active_qbs["GS"]
active_qbs["HOF"] = active_qbs["Name"].apply(is_hof)
preds = model.predict(active_qbs[predictors])

result = pandas.DataFrame({
	"Name": active_qbs["Name"],
	"HOF": preds
})

hof = result.loc[result["HOF"] == 1]
print(hof)





