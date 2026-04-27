from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def get_data():
    df = pd.read_csv('dataset.csv')
    df[['Group', 'PersonInGroup']] = df['PassengerId'].str.split('_', expand=True)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df[['CabinDeck', 'CabinNum', 'CabinSize']] = df['Cabin'].str.split('/', expand=True)
    df[['FirstName', 'LastName']] = df['Name'].str.split(' ', n=1, expand=True)
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)
    bills = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df.loc[(df[bills].sum(axis=1)>0) & (df['CryoSleep'].isna()), 'CryoSleep'] = False
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    for col in bills:
        df.loc[(df['CryoSleep'] == True) & (df[col].isna()), col] = 0
    for col in ['Age'] + bills:
        df[col] = df[col].fillna(df[col].median())
    df['TotalSpend'] = df[bills].sum(axis=1)
    df['IsInfant'] = (df['Age'] <= 5).astype(int)
    categories = ['HomePlanet', 'Destination', 'VIP', 'CabinDeck', 'CabinSize']
    for col in categories:
        df[col] = df[col].fillna(df[col].mode()[0])
    bools = ['CryoSleep', 'VIP', 'Transported']
    for col in bools:
        df[col] = df[col].astype(int)
    strs = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSize']
    df = pd.get_dummies(df, columns=strs, drop_first=False)
    df = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'FirstName', 'PersonInGroup', 'Group', 'CabinNum', 'LastName'])
    df = df.astype(float)
    X = df.drop(columns=['Transported']).values
    y = df['Transported'].astype(int).values
    cols = df.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    features, result = [col for col in cols if col != 'Transported'], 'Transported'
    return X_train, X_test, y_train, y_test, features, result
