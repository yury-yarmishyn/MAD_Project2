import statistics
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew as calc_skew
from sklearn.ensemble import VotingClassifier


# Track class
class Track:
    # Track initialization
    def __init__(self, name, artist, genre, subgenre, danceability,
                 energy, key, loudness, mode, speechiness, acousticness,
                 instrumentalness, liveness, valence, tempo, duration):
        self.name = name
        self.artist = artist
        self.genre = genre
        self.subgenre = subgenre
        self.danceability = float(danceability)
        self.energy = float(energy)
        self.key = int(key)
        self.loudness = float(loudness)
        self.mode = int(mode)
        self.speechiness = float(speechiness)
        self.acousticness = float(acousticness)
        self.instrumentalness = float(instrumentalness)
        self.liveness = float(liveness)
        self.valence = float(valence)
        self.tempo = float(tempo)
        self.duration = int(duration)

    # Show data
    def __repr__(self):
        return (f"Track(name={self.name}, artist={self.artist}, genre={self.genre}, subgenre = {self.subgenre}, danceability={self.danceability}, "
                f"energy={self.energy}, key={self.key}, loudness={self.loudness}, mode={self.mode}, speechiness={self.speechiness}, "
                f"acousticness={self.acousticness}, instrumentalness={self.instrumentalness}, liveness={self.liveness}, "
                f"valence={self.valence}, tempo={self.tempo}, duration={self.duration})")

# Get track list
def create_track_list(file_path):
    track_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = []
            variable = ""
            is_quotes = False

            for char in line:
                if char == '"':
                    is_quotes = not is_quotes
                if char == ',' and not is_quotes:
                    data.append(variable)
                    variable = ''
                else:
                    variable += char

            data.append(variable)

            track = Track(
                name=data[1],
                artist=data[2],
                genre=data[9],
                subgenre=data[10],
                danceability=data[11],
                energy=data[12],
                key=data[13],
                loudness=data[14],
                mode=data[15],
                speechiness=data[16],
                acousticness=data[17],
                instrumentalness=data[18],
                liveness=data[19],
                valence=data[20],
                tempo=data[21],
                duration=data[22]
            )
            track_list.append(track)
    return track_list

# Get unique values by parameter
def get_unique_values(tracks, parameter):
    unique_values = set()
    for track in tracks:
        val = getattr(track, parameter, None)
        if val is not None:
            unique_values.add(val)
    return list(unique_values)

# Get analytics
def calculate_statistics(tracks):
    numeric_parameters = ['danceability', 'energy', 'key', 'loudness', 'mode',
                          'speechiness', 'acousticness', 'instrumentalness',
                          'liveness', 'valence', 'tempo', 'duration']
    all_stats = {}

    for parameter in numeric_parameters:
        values = [getattr(track, parameter) for track in tracks if isinstance(getattr(track, parameter), (int, float))]

        if not values:
            continue

        stats = {}
        stats['mean'] = statistics.mean(values)
        stats['median'] = statistics.median(values)
        stats['min'] = min(values)
        stats['max'] = max(values)
        stats['std_dev'] = statistics.stdev(values) if len(values) > 1 else 0.0
        stats['skewness'] = calc_skew(values) if len(values) > 1 else 0.0

        # Visualization - Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'{parameter.capitalize()} Distribution')
        plt.xlabel(parameter.capitalize())
        plt.ylabel('Frequency')
        plt.grid(True)

        # Annotating statistical values
        plt.text(0.05, 0.95, f"Mean: {stats['mean']:.2f}\nMedian: {stats['median']:.2f}\nMin: {stats['min']:.2f}\nMax: {stats['max']:.2f}\nStd Dev: {stats['std_dev']:.2f}\nSkewness: {stats['skewness']:.2f}",
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

        plt.show()

        all_stats[parameter] = stats

    return all_stats

# Convert the list of tracks to a DataFrame
def tracks_to_dataframe(tracks):
    data = []
    for track in tracks:
        data.append({
            'name': track.name,
            'artist': track.artist,
            'genre': track.genre,
            'subgenre': track.subgenre,
            'danceability': track.danceability,
            'energy': track.energy,
            'key': track.key,
            'loudness': track.loudness,
            'mode': track.mode,
            'speechiness': track.speechiness,
            'acousticness': track.acousticness,
            'instrumentalness': track.instrumentalness,
            'liveness': track.liveness,
            'valence': track.valence,
            'tempo': track.tempo,
            'duration': track.duration
        })
    return pd.DataFrame(data)

# Train and evaluate the RandomForest model
def train_model_rf(X_train, y_train):
    model_path = "random_forest_model.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
    return pipeline

# Train and evaluate the SVM model
def train_model_svm(X_train, y_train):
    model_path = "svm_model.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(SVC(kernel='linear', random_state=42)))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
    return pipeline

# Train and evaluate the KNN model
def train_model_knn(X_train, y_train):
    model_path = "knn_model.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(KNeighborsClassifier()))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
    return pipeline


# Train and evaluate the hybrid model
def train_model_hybrid(X_train, y_train):
    model_path = "hybrid_model.pkl"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        rf_classifier = RandomForestClassifier(random_state=42)
        svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
        knn_classifier = KNeighborsClassifier()

        hybrid_classifier = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('svm', svm_classifier),
                ('knn', knn_classifier)
            ],
            voting='soft'
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(hybrid_classifier))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
    return pipeline

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy_genre = accuracy_score(y_test['genre'], y_pred[:, 0])
    accuracy_subgenre = accuracy_score(y_test['subgenre'], y_pred[:, 1])
    report_genre = classification_report(y_test['genre'], y_pred[:, 0])
    report_subgenre = classification_report(y_test['subgenre'], y_pred[:, 1])
    return accuracy_genre, report_genre, accuracy_subgenre, report_subgenre

# Predict the genre and subgenre for a new track
def predict_genre_and_subgenre(model, track):
    track_df = pd.DataFrame([track])
    predicted = model.predict(track_df)
    return predicted[0][0], predicted[0][1]

# Create DataFrame from track list
track_list = create_track_list('Tracks.txt')
df = tracks_to_dataframe(track_list)

# # Get genres and subgenres
# print(get_unique_values(track_list, 'genre'))
# print(get_unique_values(track_list, 'subgenre'))

# # Get statistics
# # stats = calculate_statistics(track_list)

# Split data into features (X) and target variables (y_genre, y_subgenre)
X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']]
y = df[['genre', 'subgenre']]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
model_rf = train_model_rf(X_train, y_train)
model_svm = train_model_svm(X_train, y_train)
model_knn = train_model_knn(X_train, y_train)
model_hybrid = train_model_hybrid(X_train, y_train)

# Evaluate the RandomForest model
accuracy_genre_rf, report_genre_rf, accuracy_subgenre_rf, report_subgenre_rf = evaluate_model(model_rf, X_test, y_test)
print("RandomForest Genre Accuracy:", accuracy_genre_rf)
print("RandomForest Genre Classification Report:\n", report_genre_rf)
print("RandomForest Subgenre Accuracy:", accuracy_subgenre_rf)
print("RandomForest Subgenre Classification Report:\n", report_subgenre_rf)

# Evaluate the SVM model
accuracy_genre_svm, report_genre_svm, accuracy_subgenre_svm, report_subgenre_svm = evaluate_model(model_svm, X_test, y_test)
print("SVM Genre Accuracy:", accuracy_genre_svm)
print("SVM Genre Classification Report:\n", report_genre_svm)
print("SVM Subgenre Accuracy:", accuracy_subgenre_svm)
print("SVM Subgenre Classification Report:\n", report_subgenre_svm)

# Evaluate the KNN model
accuracy_genre_knn, report_genre_knn, accuracy_subgenre_knn, report_subgenre_knn = evaluate_model(model_knn, X_test, y_test)
print("KNN Genre Accuracy:", accuracy_genre_knn)
print("KNN Genre Classification Report:\n", report_genre_knn)
print("KNN Subgenre Accuracy:", accuracy_subgenre_knn)
print("KNN Subgenre Classification Report:\n", report_subgenre_knn)

# Evaluate the hybrid model
accuracy_genre_hybrid, report_genre_hybrid, accuracy_subgenre_hybrid, report_subgenre_hybrid = evaluate_model(model_hybrid, X_test, y_test)
print("Hybrid Genre Accuracy:", accuracy_genre_hybrid)
print("Hybrid Genre Classification Report:\n", report_genre_hybrid)
print("Hybrid Subgenre Accuracy:", accuracy_subgenre_hybrid)
print("Hybrid Subgenre Classification Report:\n", report_subgenre_hybrid)

random_tracks = df.sample(n=10, random_state=42)

# Predict genre and subgenre for 10 random tracks
for index, track in random_tracks.iterrows():
    new_track = track.to_dict()
    track_features = {k: new_track[k] for k in X.columns}
    track_name = new_track['name']
    track_artist = new_track['artist']

    # Predict with RandomForest model
    predicted_genre_rf, predicted_subgenre_rf = predict_genre_and_subgenre(model_rf, track_features)
    print(f"RandomForest Predicted Genre for '{track_name}' by {track_artist}: {predicted_genre_rf}")
    print(f"RandomForest Predicted Subgenre for '{track_name}' by {track_artist}: {predicted_subgenre_rf}")

    # Predict with SVM model
    predicted_genre_svm, predicted_subgenre_svm = predict_genre_and_subgenre(model_svm, track_features)
    print(f"SVM Predicted Genre for '{track_name}' by {track_artist}: {predicted_genre_svm}")
    print(f"SVM Predicted Subgenre for '{track_name}' by {track_artist}: {predicted_subgenre_svm}")

    # Predict with KNN model
    predicted_genre_knn, predicted_subgenre_knn = predict_genre_and_subgenre(model_knn, track_features)
    print(f"KNN Predicted Genre for '{track_name}' by {track_artist}: {predicted_genre_knn}")
    print(f"KNN Predicted Subgenre for '{track_name}' by {track_artist}: {predicted_subgenre_knn}")

    # Predict with Hybrid model
    predicted_genre_hybrid, predicted_subgenre_hybrid = predict_genre_and_subgenre(model_hybrid, track_features)
    print(f"Hybrid Predicted Genre for '{track_name}' by {track_artist}: {predicted_genre_hybrid}")
    print(f"Hybrid Predicted Subgenre for '{track_name}' by {track_artist}: {predicted_subgenre_hybrid}")
    print("\n")

# Example of predicting the genre and subgenre for a new track

# Symptom of the Universe - Black Sabbath
new_track = {
    'danceability': 0.29,
    'energy': 0.75,
    'key': 6,
    'loudness': -9.09,
    'mode': 0,
    'speechiness': 0.05,
    'acousticness': 0.03,
    'instrumentalness': 0.46,
    'liveness': 0.12,
    'valence': 0.42,
    'tempo': 174.0,
    'duration': 389000
}

# Predict with RandomForest model
predicted_genre_rf, predicted_subgenre_rf = predict_genre_and_subgenre(model_rf, new_track)
print("RandomForest Predicted Genre:", predicted_genre_rf)
print("RandomForest Predicted Subgenre:", predicted_subgenre_rf)

# Predict with SVM model
predicted_genre_svm, predicted_subgenre_svm = predict_genre_and_subgenre(model_svm, new_track)
print("SVM Predicted Genre:", predicted_genre_svm)
print("SVM Predicted Subgenre:", predicted_subgenre_svm)

# Predict with KNN model
predicted_genre_knn, predicted_subgenre_knn = predict_genre_and_subgenre(model_knn, new_track)
print("KNN Predicted Genre:", predicted_genre_knn)
print("KNN Predicted Subgenre:", predicted_subgenre_knn)

# Predict the genre and subgenre for a new track using the hybrid model
predicted_genre_hybrid, predicted_subgenre_hybrid = predict_genre_and_subgenre(model_hybrid, new_track)
print("Hybrid Predicted Genre:", predicted_genre_hybrid)
print("Hybrid Predicted Subgenre:", predicted_subgenre_hybrid)