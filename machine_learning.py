from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import os, pprint, pickle, shutil
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import tree
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

class MachineLearning():
    def __init__(self):
        # if not f'{os.getcwd()}/data/my_progress/census.pkl' and not f'{os.getcwd()}/data/my_progress/credit.pkl':
        # load_csv_files
        base_credit = pd.read_csv(f'{os.getcwd()}/data/my_progress/credit_data.csv')
        base_census = pd.read_csv(f'{os.getcwd()}/data/my_progress/census.csv')

        # treatment_data_credit
        base_credit.loc[base_credit["age"] < 0, "age"] = base_credit["age"][base_credit["age"] > 0].mean()
        base_credit["age"].fillna(base_credit["age"].mean(), inplace=True)
        scaler_credit = StandardScaler()
        X_credit = scaler_credit.fit_transform(base_credit.iloc[:, 1:4].values)
        y_credit = base_credit.iloc[:, 4].values

        # treatment_data_cencus_LabelEncoder
        fit_transform = [1, 3, 5, 6, 7, 8, 9, 13]
        X_census = base_census.iloc[:, 0:14].values
        y_census = base_census.iloc[:, 14].values
        for index in range(14):
            if index in fit_transform:
                X_census[:, index] = LabelEncoder().fit_transform(X_census[:, index])
        # treatment_data_census_OneHotEncoder
        onehotencoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), fit_transform)],
                                                 remainder="passthrough")
        X_census = onehotencoder_census.fit_transform(X_census).toarray()
        # scaling
        X_census = StandardScaler().fit_transform(X_census)

        # training
        self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste = \
            train_test_split(X_credit, y_credit, test_size=0.25, random_state=0)
        self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste = \
            train_test_split(X_census, y_census, test_size=0.15, random_state=0)

        # treatment_data_cencus_LabelEncoder
        base_risco_credito = pd.read_csv(f'{os.getcwd()}/data/my_progress/risco_credito.csv')

        self.X_risco_credito = base_risco_credito.iloc[:, 0:4].values
        self.y_risco_credito = base_risco_credito.iloc[:, 4].values
        for index in range(4):
            self.X_risco_credito[:, index] = LabelEncoder().fit_transform(self.X_risco_credito[:, index])

        with open('credit.pkl', mode="wb") as f:
            pickle.dump([self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste], f)
        shutil.move(f'{os.getcwd()}/credit.pkl', f'{os.getcwd()}/data/my_progress/credit.pkl')

        with open('census.pkl', mode="wb") as f:
            pickle.dump([self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste], f)
        shutil.move(f'{os.getcwd()}//census.pkl', f'{os.getcwd()}/data/my_progress/census.pkl')

        with open('risco_credito.pkl', mode="wb") as f:
            pickle.dump([self.X_risco_credito, self.y_risco_credito], f)
        shutil.move(f'{os.getcwd()}//risco_credito.pkl', f'{os.getcwd()}/data/my_progress/risco_credito.pkl')
        # else:
        #     with open(f'{os.getcwd()}/data/my_progress/credit.pkl', 'rb') as f:
        #         self.X_credit_treinamento, self.y_credit_treinamento, self.X_credit_teste, self.y_credit_teste = pickle.load(f)
        #
        #     with open(f'{os.getcwd()}/data/my_progress/credit.pkl', 'rb') as f:
        #         self.X_census_treinamento, self.y_census_treinamento, self.X_census_teste, self.y_census_teste = pickle.load(f)

    def statistics(self, method, data_base, trainer, x_treinamento, x_teste, y_treinamento, y_teste, previsoes, previsores):
        cm = ConfusionMatrix(trainer)
        cm.fit(x_treinamento, y_treinamento)

        print(f"\n{method}_{data_base}")
        print(f"accuracy_score: {cm.score(x_teste, y_teste)}")
        pprint.pprint(f"confusion_matrix: {confusion_matrix(y_teste, previsoes)}")
        print("classification_report: ")
        print(classification_report(y_teste, previsoes))
        if (method != "naive_bayes" and data_base != "credit_risk") and (method != "random_forest"):
            names_class = list()
            for name_class in trainer.classes_:
                names_class.append(str(name_class))

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
            if method == "decision_tree" and data_base == "census":
                tree.plot_tree(trainer, class_names=names_class, filled=True)
            else:
                tree.plot_tree(trainer, feature_names=previsores, class_names=names_class, filled=True)
            fig.savefig(f'{data_base}_{method}.png')
            shutil.move(f'{os.getcwd()}//{data_base}_{method}.png', f'{os.getcwd()}/data/imagens/{data_base}_{method}.png')

    def naive_bayes(self, method,  data_base, previsores):
        if data_base == "credit_risk":
            naive_risco_credito = GaussianNB()
            naive_risco_credito.fit(self.X_risco_credito, self.y_risco_credito)
            previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
            print(f"\n{method}_{data_base}: {previsao}")
        elif data_base == "credit":
            naive_credit_data = GaussianNB()
            naive_credit_data.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = naive_credit_data.predict(self.X_credit_teste)
            self.statistics(method, data_base, naive_credit_data, self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste, previsoes, previsores)
        elif data_base == "census":
            naive_census_data = GaussianNB()
            naive_census_data.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = naive_census_data.predict(self.X_census_teste)
            self.statistics(method, data_base, naive_census_data, self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste, previsoes, previsores)

    def decision_tree(self, method,  data_base, previsores):
        if data_base == "credit_risk":
            tree_credit_risk = DecisionTreeClassifier(criterion="entropy")
            tree_credit_risk.fit(self.X_risco_credito, self.y_risco_credito)
            previsao = tree_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
            print(f"\n{method}_{data_base}: {previsao}")
        elif data_base == "credit":
            tree_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
            tree_credit.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = tree_credit.predict(self.X_credit_teste)
            self.statistics(method, data_base, tree_credit, self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste, previsoes, previsores)
        elif data_base == "census":
            tree_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
            tree_census.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = tree_census.predict(self.X_census_teste)
            self.statistics(method, data_base, tree_census, self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste, previsoes, previsores)

    def random_forest(self, method,  data_base, previsores):
        if data_base == "credit":
            credit_random_forest = RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)
            credit_random_forest.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = credit_random_forest.predict(self.X_credit_teste)
            self.statistics(method, data_base, credit_random_forest, self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste, previsoes, previsores)
        elif data_base == "census":
            census_random_forest = RandomForestClassifier(n_estimators=40, criterion="entropy", random_state=0)
            census_random_forest.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = census_random_forest.predict(self.X_census_teste)
            self.statistics(method, data_base, census_random_forest, self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste, previsoes, previsores)

    def knn(self, method, data_base):
        print(f"{method} {data_base}")
        if data_base == "credit":
            knn_credit = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
            knn_credit.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = knn_credit.predict(self.X_credit_teste)

            cm = ConfusionMatrix(knn_credit)
            cm.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            print(f"accuracy_score: {cm.score(self.X_credit_teste, self.y_credit_teste)}")
            pprint.pprint(f"confusion_matrix: {confusion_matrix(self.y_credit_teste, previsoes)}")
        elif data_base == "census":
            knn_census = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
            knn_census.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = knn_census.predict(self.X_census_teste)

            cm = ConfusionMatrix(knn_census)
            cm.fit(self.X_census_treinamento, self.y_census_treinamento)
            print(f"accuracy_score: {cm.score(self.X_census_teste, self.y_census_teste)}")
            pprint.pprint(f"confusion_matrix: {confusion_matrix(self.y_census_teste, previsoes)}")



def launcher():
    warehouse = ["credit", "census"]

    machine_learning = MachineLearning()
    for data_base in warehouse:
        machine_learning.knn("knn", data_base)
        # machine_learning.naive_bayes("naive_bayes", data_base, warehouse[data_base])
        # machine_learning.decision_tree("decision_tree", data_base, warehouse[data_base])
        # machine_learning.random_forest("random_forest", data_base, warehouse[data_base])

launcher()