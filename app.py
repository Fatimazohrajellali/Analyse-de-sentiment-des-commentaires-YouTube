import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Télécharger les stopwords si ce n'est pas encore fait
#nltk.download('stopwords')

# Charger le modèle de spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Fonction de nettoyage du texte
def clean_text(text):
    """
    Nettoie le texte en supprimant les caractères spéciaux, en le convertissant en minuscules,
    en lemmatisant et en supprimant les stopwords.
    """
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text, re.I | re.A)
    # Conversion en minuscules
    text = text.lower()

    # Tokenisation et lemmatisation avec spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]

    # Suppression des stopwords en français
    stop_words = set(stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]

    # Reconstitution du texte nettoyé
    return ' '.join(tokens)

# Charger le modèle et le vectoriseur
@st.cache_resource  # Cache pour éviter de recharger à chaque interaction
def load_artifacts():
    """
    Charge le modèle Random Forest et le vectoriseur TF-IDF.
    """
    model = joblib.load("C:/Users/INKA/Desktop/ANALYSE DE SENTIMENT SUR YOUTUBE/rf_model.pkl")
    vectorizer = joblib.load("C:/Users/INKA/Desktop/ANALYSE DE SENTIMENT SUR YOUTUBE/tfidf_vectorizer.pkl")
    return model, vectorizer

# Charger les artefacts
model, vectorizer = load_artifacts()

# Mapping manuel des labels
label_mapping = {0: "Neutre", 1: "Négatif", 2: "Positif"}

# Titre de l'application
st.title("Analyse de Sentiment des Commentaires YouTube")

# Zone de texte pour saisir le commentaire
comment = st.text_area("Entrez un commentaire à analyser :", "")

# Bouton pour lancer l'analyse
if st.button("Analyser le sentiment"):
    if comment.strip() == "":
        st.warning("Veuillez entrer un commentaire.")
    else:
        # Nettoyer le commentaire
        cleaned_comment = clean_text(comment)

        # Vectoriser le commentaire
        comment_vectorized = vectorizer.transform([cleaned_comment])

        # Prédiction du sentiment
        prediction = model.predict(comment_vectorized)[0]  # Récupère la valeur numérique prédite
        sentiment = label_mapping[prediction]  # Convertit en label textuel

        # Affichage du résultat
        st.success(f"Le sentiment du commentaire est : **{sentiment}**")

        

        # Visualisation 2 : Mots les plus fréquents dans le commentaire
        st.subheader("Mots les plus fréquents dans le commentaire")
        words = cleaned_comment.split()
        word_counts = Counter(words)
        top_words = word_counts.most_common(10)  # Top 10 des mots les plus fréquents

        # Création d'un DataFrame pour afficher les mots les plus fréquents
        top_words_df = pd.DataFrame(top_words, columns=['Mot', 'Fréquence'])
        
        # Graphique à barres
        st.bar_chart(top_words_df.set_index('Mot'))

        # Affichage des mots les plus fréquents sous forme de tableau
        st.write("Liste des mots les plus fréquents :")
        st.dataframe(top_words_df)

# Visualisation 1 : Camembert (Pie Chart) pour la répartition des sentiments
        st.subheader("Répartition des sentiments dans les données d'entraînement")
        sentiment_distribution = {
            "Positif": 10429,
            "Négatif": 4882,
            "Neutre": 2584
        }
        fig1, ax1 = plt.subplots()
        ax1.pie(
            sentiment_distribution.values(), 
            labels=sentiment_distribution.keys(), 
            autopct='%1.1f%%', 
            colors=['green', 'red', 'blue']
        )
        ax1.axis('equal')  # Assure que le camembert est un cercle
        st.pyplot(fig1)