from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
import csv
import ssl
from sklearn.tree import DecisionTreeClassifier

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_url_path='/static')


# Charger les données à partir d'un fichier CSV distant
data = pd.read_csv('https://raw.githubusercontent.com/MFN-ofc/dc5b_scrapping_robert_rouquet/main/scrapping%20akinatorzer/fichier%20parse2%203.csv', delimiter=';', quoting=csv.QUOTE_NONE)

# Créer un LabelEncoder pour chaque caractéristique (colonne)
le_genres = preprocessing.LabelEncoder()  # LabelEncoder pour les genres
le_decades = preprocessing.LabelEncoder()  # LabelEncoder pour les décennies
le_lengths = preprocessing.LabelEncoder()  # LabelEncoder pour les durées
le_classifications = preprocessing.LabelEncoder()  # LabelEncoder pour les classifications

# Ajuster les LabelEncoders et transformer les données correspondantes

# Pour les genres, nous concaténons les colonnes Genre1, Genre2 et Genre3, supprimons les espaces en début et fin, puis obtenons les valeurs uniques
all_genres = pd.concat([data['Genre1'], data['Genre2'], data['Genre3']]).str.strip().unique()
le_genres = le_genres.fit(all_genres)  # Apprentissage du LabelEncoder pour les genres
data['Genre1'] = le_genres.transform(data['Genre1'].str.strip())  # Transformation des valeurs de la colonne Genre1
data['Genre2'] = le_genres.transform(data['Genre2'].str.strip())  # Transformation des valeurs de la colonne Genre2
data['Genre3'] = le_genres.transform(data['Genre3'].str.strip())  # Transformation des valeurs de la colonne Genre3

# Convertir l'année de sortie en décennie

# Nous définissons les plages de décennies et utilisons la fonction pd.cut pour convertir les années en décennies correspondantes
decades = ['1920-1929', '1930-1939', '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999', '2000-2009', '2010-2019', '2020-2029', '2030-2039']
data['Decade'] = pd.cut(data['Année de sortie'], bins=range(1920, 2040, 10), labels=False, right=False)
data['Decade'] = data['Decade'].map(lambda x: decades[x] if pd.notnull(x) else x)  # Remplacer les valeurs numériques par les décennies correspondantes
le_decades = le_decades.fit(decades)  # Apprentissage du LabelEncoder pour les décennies
data['Decade'] = le_decades.transform(data['Decade'])  # Transformation des valeurs de la colonne Decade en valeurs numériques

# Convertir la durée en catégorie

# Nous définissons les plages de durées et utilisons la fonction pd.cut pour convertir les durées en catégories correspondantes
lengths = ['Moins de 90 minutes', '90-120 minutes', 'Plus de 120 minutes']
data['Length'] = pd.cut(data['Durée (min)'], bins=[0, 90, 120, float('inf')], labels=lengths, right=False)
le_lengths = le_lengths.fit(lengths)  # Apprentissage du LabelEncoder pour les durées
data['Length'] = le_lengths.transform(data['Length'])  # Transformation des valeurs de la colonne Length en valeurs numériques

# Convertir la classification en catégorie

# Nous filtrons les données pour inclure uniquement les classifications spécifiées, puis utilisons le LabelEncoder pour les convertir en valeurs numériques
classifications = data['Classification'].unique().tolist()
data = data[data['Classification'].isin(classifications)]
le_classifications = le_classifications.fit(classifications)  # Apprentissage du LabelEncoder pour les classifications
data['Classification'] = le_classifications.transform(data['Classification'])  # Transformation des valeurs de la colonne Classification en valeurs numériques

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtenir les données soumises par l'utilisateur
        selected_genres = request.form.getlist('genres')
        selected_decade = request.form.get('decade')
        selected_length = request.form.get('length')
        selected_classification = request.form.get('classification')

        # Convertir la décennie sélectionnée en chaîne correspondante
        selected_decade_str = decades[int(selected_decade)]
        selected_decade_num = le_decades.transform([selected_decade_str])[0]

        # Convertir la durée sélectionnée en chaîne correspondante
        selected_length_str = lengths[int(selected_length)]
        selected_length_num = le_lengths.transform([selected_length_str])[0]

        # Convertir la classification sélectionnée en chaîne correspondante
        selected_classification_str = classifications[int(selected_classification)]
        selected_classification_num = le_classifications.transform([selected_classification_str])[0]

        selected_genres = request.form.getlist('genres')

        # Convertir les genres sélectionnés en valeurs numériques
        selected_genres_nums = [le_genres.transform([genre.strip()])[0] for genre in selected_genres]

        # Créer un nouveau DataFrame avec les caractéristiques sélectionnées par l'utilisateur
        selected_data = pd.DataFrame({
            'Genre1': [selected_genres_nums[0]] if len(selected_genres_nums) > 0 else [-1],
            'Genre2': [selected_genres_nums[1]] if len(selected_genres_nums) > 1 else [-1],
            'Genre3': [selected_genres_nums[2]] if len(selected_genres_nums) > 2 else [-1],
            'Decade': [selected_decade_num],
            'Length': [selected_length_num],
            'Classification': [selected_classification_num]
        })

        selected_data['Genre1'] = selected_data['Genre1'].apply(lambda x: x if x != -1 else selected_data['Genre1'].mode().iloc[0])
        selected_data['Genre2'] = selected_data['Genre2'].apply(lambda x: x if x != -1 else selected_data['Genre2'].mode().iloc[0])
        selected_data['Genre3'] = selected_data['Genre3'].apply(lambda x: x if x != -1 else selected_data['Genre3'].mode().iloc[0])

        # Filtrer les données en fonction des préférences spécifiques de l'utilisateur
        filtered_data = data[
            (data['Genre1'].isin(selected_genres)) |
            (data['Genre2'].isin(selected_genres)) |
            (data['Genre3'].isin(selected_genres)) |
            (data['Decade'] == selected_decade_num) |
            (data['Length'] == selected_length_num) |
            (data['Classification'] == selected_classification_num)
        ]

        # Créer un classifieur Random Forest et l'entraîner
        clf = RandomForestClassifier(random_state=0, class_weight='balanced')
        X = filtered_data[['Genre1', 'Genre2', 'Genre3', 'Decade', 'Length', 'Classification']]
        y = filtered_data['Nom']
        X.loc[X['Genre2'] == -1, 'Genre2'] = X.loc[X['Genre2'] == -1, 'Genre1']
        X.loc[X['Genre3'] == -1, 'Genre3'] = X.loc[X['Genre3'] == -1, 'Genre1']
        clf.fit(X, y)

        # Prédire le film
        movie = clf.predict(selected_data)
        probas = clf.predict_proba(selected_data)[0]
        top_indices = probas.argsort()[-6:-1][::-1]
        top_movies = y.iloc[top_indices].tolist()
        top_probabilities = [probas[i] for i in top_indices]
        top_probabilities = [f"{prob*100:.2f}%" for prob in top_probabilities]

        return redirect(url_for('results', movie=movie, top_movies=top_movies, top_probabilities=top_probabilities))

    # Obtention des listes de genres, décennies, durées et classifications pour l'affichage dans le formulaire
    genres = le_genres.classes_.tolist()
    decades_list = ['1920-1929', '1930-1939', '1940-1949', '1950-1959', '1960-1969', '1970-1979', '1980-1989', '1990-1999', '2000-2009', '2010-2019', '2020-2029', '2030-2039']
    lengths_list = ['Moins de 90 minutes', '90-120 minutes', 'Plus de 120 minutes']
    classifications_list = ['Tous publics', '12+', '16+', '18+']

    return render_template('index.html', genres=genres, decades=decades_list, lengths=lengths_list, classifications=classifications_list)

@app.route('/results')
def results():

    movie = request.args.get('movie')
    top_movies = request.args.getlist('top_movies')
    top_probabilities = request.args.getlist('top_probabilities')
    movie = movie.strip("['']")

    clf = RandomForestClassifier(random_state=0,class_weight='balanced')
    X = data[['Genre1', 'Genre2', 'Genre3', 'Decade', 'Length', 'Classification']]
    y = data['Nom']
    X.loc[X['Genre2'] == -1, 'Genre2'] = X.loc[X['Genre2'] == -1, 'Genre1']
    X.loc[X['Genre3'] == -1, 'Genre3'] = X.loc[X['Genre3'] == -1, 'Genre1']
    clf.fit(X, y)

    # Afficher l'importance des fonctionnalités
    feature_importance = clf.feature_importances_
    feature_names = ['Genre1', 'Genre2', 'Genre3', 'Decade', 'Length', 'Classification']
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    return render_template('results.html', movie=movie, top_movies_probs=zip(top_movies, top_probabilities), feature_importance=feature_importance_dict)

if __name__ == '__main__':
    app.run(debug=True)
