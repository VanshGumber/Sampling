import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(10)

data = pd.read_csv("Creditcard_data.csv")
X = data.drop("Class", axis=1)
y = data["Class"]

sm = SMOTE(random_state=10)
Xb, yb = sm.fit_resample(X, y)

bal = pd.DataFrame(Xb, columns=X.columns)
bal["Class"] = yb

train, test = train_test_split(
    bal, test_size=0.25, stratify=bal["Class"], random_state=10
)

def random(d, f=0.65):
    return d.sample(frac=f, random_state=10)

def strat(d, f=0.65):
    return d.groupby("Class", group_keys=False).apply(
        lambda x: x.sample(frac=f, random_state=10)
    )

def sys(d, step=3):
    d = d.sample(frac=1, random_state=10).reset_index(drop=True)
    return d.iloc[::step]

def cluster(d, k=8, pick=4):
    feat = d.drop("Class", axis=1)
    km = KMeans(n_clusters=k, random_state=10, n_init=10)
    d2 = d.copy()
    d2["cid"] = km.fit_predict(feat)
    sel = np.random.choice(k, pick, replace=False)
    return d2[d2["cid"].isin(sel)].drop("cid", axis=1)

def boot(d, f=0.65):
    n = int(f * len(d))
    return d.sample(n=n, replace=True, random_state=10)

samples = {
    "S1": random(train),
    "S2": strat(train),
    "S3": sys(train),
    "S4": cluster(train),
    "S5": boot(train),
}

models = {
    "M1": LogisticRegression(max_iter=5000, random_state=10),
    "M2": DecisionTreeClassifier(max_depth=7, random_state=10),
    "M3": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=10),
    "M4": GaussianNB(),
    "M5": SVC(random_state=10),
}

res = pd.DataFrame(index=models.keys(), columns=samples.keys())

Xt = test.drop("Class", axis=1)
yt = test["Class"]

for sname, sdata in samples.items():
    Xtr = sdata.drop("Class", axis=1)
    ytr = sdata["Class"]

    for mname, m in models.items():
        model = clone(m)

        if mname in ["M1", "M5"]:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xt_s = sc.transform(Xt)
            model.fit(Xtr_s, ytr)
            pred = model.predict(Xt_s)
        else:
            model.fit(Xtr, ytr)
            pred = model.predict(Xt)

        res.loc[mname, sname] = accuracy_score(yt, pred) * 100

res = res.astype(float)

model_names = {
    "M1": "Logistic Regression",
    "M2": "Decision Tree",
    "M3": "Random Forest",
    "M4": "Naive Bayes",
    "M5": "SVM"
}

sample_names = {
    "S1": "Simple Random",
    "S2": "Stratified",
    "S3": "Systematic",
    "S4": "Cluster",
    "S5": "Bootstrap"
}

plot_df = res.copy()
plot_df.index = [model_names[i] for i in res.index]
plot_df.columns = [sample_names[i] for i in res.columns]

plt.figure(figsize=(12,6))
sns.heatmap(
    plot_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Accuracy (%)'}
)
plt.title("Model Accuracy Across Sampling Techniques")
plt.xlabel("Sampling Technique")
plt.ylabel("Machine Learning Model")
plt.tight_layout()
plt.show()

ax = plot_df.T.plot(kind="bar", figsize=(14,6))
plt.title("Model Accuracy Comparison by Sampling Technique")
plt.xlabel("Sampling Technique")
plt.ylabel("Accuracy (%)")
plt.legend(title="Machine Learning Models", bbox_to_anchor=(1.02,1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
