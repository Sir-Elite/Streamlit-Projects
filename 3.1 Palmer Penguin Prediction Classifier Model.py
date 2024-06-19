import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

penguins = pd.read_csv('penguins_cleaned.csv')

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

# encode: use true or false in separate columns for each category instead of categorical column
for column in encode:
    dummy = pd.get_dummies(df[column], prefix=column)  # encoding function
    df = pd.concat([df, dummy], axis=1)
    print(df)
    del df[column]

target_mapper = {
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
}


def target_encoder(value):
    return target_mapper[value]


df['species'] = df['species'].apply(target_encoder)

x = df.drop(target, axis=1)
y = df[target]

# Build a random forest classifier
classifier = RandomForestClassifier()
classifier.fit(x, y)

# Save the model "pickle.dump(model_name, open('file_name.pkl', 'wb'))"
pickle.dump(classifier, open('penguins_classifier.pkl', 'wb'))
