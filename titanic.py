# librairie pour data analisis
import pandas as pd

# librairie pour visualisation de données
import seaborn as sns
import matplotlib.pyplot as plt

# Algo random forest
from sklearn.ensemble import RandomForestClassifier

# Importation des fichiers
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# traitement des données
for dataset in combine:

    # Répartition en tranche d'age
    dataset.loc[dataset['Age'].isnull(), 'Age'] = 0  #Si null alors egal a 0
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 1 # Si inferieur ou egal à 16 ans alors egal à 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 30), 'Age'] = 2 # Si supérieur à 16 ans et inférieur ou egal a 30 alors egal à 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 60), 'Age'] = 3 # Si supérieur à 30 ans et inférieur ou egal a 60 alors egal à 3
    dataset.loc[dataset['Age'] > 60, 'Age'] = 4 # Si supérieur a 60 alors égal a 4

    # Répartition en tranche de prix de la même maniere que pour l'age
    dataset.loc[dataset['Fare'] < 7, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] >= 7) & (dataset['Fare'] <= 14), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] >= 14) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    # convertir String de Sex en integer
    # Si dataset['Sex'] = "female" alors dataset['Sex'] =1
    # Si dataset['Sex'] = "male" alors dataset['Sex']=0
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Convertir String de Embarked en integer
    dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = 0 #Si null alors egal a 0
    dataset.loc[dataset['Embarked'] == "C", 'Embarked'] = 1 # Si Embarked est égal a "C" alors Embarked egal à 1
    dataset.loc[dataset['Embarked'] == "Q", 'Embarked'] = 2 #...
    dataset.loc[dataset['Embarked'] == "S", 'Embarked'] = 3 #...

    # Creation d'un nouveau jeu de donnée : si la personne et seul ou non
    dataset['seul'] = 0 # Création de la table
    dataset.loc[dataset['SibSp'] + dataset['Parch'] == 0, 'seul'] = 1 # Si la personne n'a ni son époux ni ces enfants à bord du Titanic alors il est bien seul (égale a 1)
    


# Suppretion des données non traité
train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)  # Suppression de 'Ticket', 'Cabin', 'Name' de train ...
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)  # ... et de test

# Affichage des Valeurs
print('-' * 70)
print("Nombre total de passagers : ", train_df["Survived"].count())
print("Nombre de passagers ayant survecu : ", train_df["Survived"].mean() * train_df["Survived"].count())
print("Soit ", 100 * train_df["Survived"].mean().round(2), end=" %\n")
print('-' * 70)
print("Description de la table : ", train_df.describe(), sep='\n')
print('-' * 70)
print("Survivant en fonction des classes : ",
      train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean(), sep="\n")
print('-' * 70)
print("Survivant en fonction de leur genre :",
      train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean(), sep='\n')
print('-' * 70)
print("Survivant en fonction du prix :",
      train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean(), sep='\n')
print('-' * 70)
print("Survivant en fonction de si la personne et seul ou non :",
      train_df[['seul', 'Survived']].groupby(['seul'], as_index=False).mean(), sep='\n')
print('-' * 70)

# Affichage des Graphiques
i = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
i.map(plt.hist, 'Age', alpha=.5, bins=20)
i.add_legend()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Embarked', bins=20)
#plt.show()

# Application du random forest
X_train = train_df.drop("Survived", axis=1)
# La variable X_train va permettre a l'agorithme de s'entrainer.
# Il faut donc attribuer la table train_df et supprimer la collone Survived
Y_train = train_df["Survived"]
# Y_train servira pour comparer les resultat obtenu avec l'agorithme. Il faut donc attribuer la collone Survived
random_forest = RandomForestClassifier(n_estimators=5000, oob_score=True, verbose=True)
random_forest.fit(X_train, Y_train)

print("Score : ", round(random_forest.oob_score_ * 100, 2)) #Affiche le score arrondi a 2 décimales

#print(type(test_df))
#submission = pd.DataFrame(random_forest.predict(test_df), index=range(892, 1310), columns=['Survived'])
#submission.index.name = 'PassengerId'
#submission.to_csv('submission.csv')
