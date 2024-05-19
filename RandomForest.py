import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from main import create_track_list


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

# Create DataFrame from track list
track_list = create_track_list('Tracks.txt')
df = tracks_to_dataframe(track_list)

# Split data into features (X) and target variables (y_genre, y_subgenre)
X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']]
y = df[['genre', 'subgenre']]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multi-output model
def train_model(X_train, y_train):
    # Create a pipeline with data normalization and a multi-output classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    # Train the model
    pipeline.fit(X_train, y_train)
    return pipeline

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Predict genres and subgenres on the test set
    y_pred = model.predict(X_test)
    # Calculate accuracy for each output
    accuracy_genre = accuracy_score(y_test['genre'], y_pred[:, 0])
    accuracy_subgenre = accuracy_score(y_test['subgenre'], y_pred[:, 1])
    # Generate classification reports for each output
    report_genre = classification_report(y_test['genre'], y_pred[:, 0])
    report_subgenre = classification_report(y_test['subgenre'], y_pred[:, 1])
    return accuracy_genre, report_genre, accuracy_subgenre, report_subgenre

# Evaluate the model and print the results
accuracy_genre, report_genre, accuracy_subgenre, report_subgenre = evaluate_model(model, X_test, y_test)
print("Genre Accuracy:", accuracy_genre)
print("Genre Classification Report:\n", report_genre)
print("Subgenre Accuracy:", accuracy_subgenre)
print("Subgenre Classification Report:\n", report_subgenre)

# Predict the genre and subgenre for a new track
def predict_genre_and_subgenre(model, track):
    # Convert the track dictionary to a DataFrame
    track_df = pd.DataFrame([track])
    # Predict the genre and subgenre
    predicted = model.predict(track_df)
    return predicted[0][0], predicted[0][1]

# Example of predicting the genre and subgenre for a new track
new_track = {
    'danceability': 0.74,
    'energy': 0.54,
    'key': 1,
    'loudness': -10.16,
    'mode': 1,
    'speechiness': 0.26,
    'acousticness': 0.22,
    'instrumentalness': 0.0,
    'liveness': 0.1,
    'valence': 0.43,
    'tempo': 171.0,
    'duration': 187000
}
predicted_genre, predicted_subgenre = predict_genre_and_subgenre(model, new_track)
print("Predicted Genre:", predicted_genre)
print("Predicted Subgenre:", predicted_subgenre)