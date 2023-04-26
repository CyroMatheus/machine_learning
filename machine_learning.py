from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
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

class MachineLearning():
    def __init__(self):
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

    def statistics(self, trainer, x_treinamento, x_teste, y_treinamento, y_teste, previsoes):
        cm = ConfusionMatrix(trainer)
        cm.fit(x_treinamento, y_treinamento)

        print(accuracy_score(y_teste, previsoes))
        print(cm.score(x_teste, y_teste))
        print(confusion_matrix(y_teste, previsoes))
        print(classification_report(y_teste, previsoes))

    def naive_bayes(self, data_base):
        if data_base == "risco_credito":
            naive_risco_credito = GaussianNB()
            naive_risco_credito.fit(self.X_risco_credito, self.y_risco_credito)
            previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
            print(previsao)
        elif data_base == "credit":
            naive_credit_data = GaussianNB()
            naive_credit_data.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = naive_credit_data.predict(self.X_credit_teste)
            # self.statistics(naive_credit_data, previsoes, self.X_credit_treinamento, self.y_credit_treinamento, self.X_credit_teste, self.y_credit_teste)
        elif data_base == "census":
            naive_census_data = GaussianNB()
            naive_census_data.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = naive_census_data.predict(self.X_census_teste)
            # self.statistics(naive_census_data, previsoes, self.X_census_treinamento, self.y_census_treinamento, self.X_census_teste, self.y_census_teste)

    def decision_tree(self, data_base):
        previsores = ['income', 'age', 'loan']
        if data_base == "risco_credito":
            previsores = ['história', 'dívida', 'garantias', 'renda']
            tree_credit_risk = DecisionTreeClassifier(criterion="entropy")
            tree_credit_risk.fit(self.X_risco_credito, self.y_risco_credito)
            print(tree_credit_risk.feature_importances_)
            pprint.pprint(tree.plot_tree(tree_credit_risk, feature_names=previsores, class_names=tree_credit_risk.classes_, filled=True))
            figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
            print(self.X_risco_credito, self.y_risco_credito)
            previsoes = tree_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
            print(previsoes)
        elif data_base == "credit":
            tree_credit = DecisionTreeClassifier(criterion="entropy", random_state=0)
            tree_credit.fit(self.X_credit_treinamento, self.y_credit_treinamento)
            previsoes = tree_credit.predict(self.X_credit_teste)

            names_class = list()
            for name_class in tree_credit.classes_:
                names_class.append(str(name_class))

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
            tree.plot_tree(tree_credit, feature_names=previsores, class_names=names_class, filled=True)
            fig.savefig('credit_decision_tree.png')
            shutil.move(f'{os.getcwd()}//credit_decision_tree.png', f'{os.getcwd()}/data/Imagens/credit_decision_tree.png')

            self.statistics(tree_credit, self.X_credit_treinamento, self.X_credit_teste, self.y_credit_treinamento, self.y_credit_teste, previsoes)
        elif data_base == "census":
            tree_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
            tree_census.fit(self.X_census_treinamento, self.y_census_treinamento)
            previsoes = tree_census.predict(self.X_census_teste)

            names_class = list()
            for name_class in tree_census.classes_:
                names_class.append(str(name_class))
            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
            # tree.plot_tree(tree_census, feature_names=previsores, class_names=names_class, filled=True)
            # fig.savefig('census_decision_tree.png')
            # shutil.move(f'{os.getcwd()}//census_decision_tree.png', f'{os.getcwd()}/data/Imagens/census_decision_tree.png')

            self.statistics(tree_census, self.X_census_treinamento, self.X_census_teste, self.y_census_treinamento, self.y_census_teste, previsoes)


def launcher():
    warehouse = ["risco_credito", "credit", "census"]
    machine_learning = MachineLearning()
    for data_base in warehouse:
        if data_base != "risco_credito":# and data_base != "census":
            # print("Naive Bayes")
            # machine_learning.naive_bayes(data_base)
            print("Decision Tree")
            machine_learning.decision_tree(data_base)

launcher()