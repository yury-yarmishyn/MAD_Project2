import statistics
from scipy.stats import skew as calc_skew

# Track class
class Track:
    # Track initialization
    def __init__(self, name, artist, genre, subgenre, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration):
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

# Get parameter analytics
def calculate_statistics(tracks, parameter):
    values = [getattr(track, parameter) for track in tracks if isinstance(getattr(track, parameter), (int, float))]

    if not values:
        return None

    stats = {}
    stats['mean'] = statistics.mean(values)
    stats['median'] = statistics.median(values)
    stats['min'] = min(values)
    stats['max'] = max(values)
    stats['std_dev'] = statistics.stdev(values) if len(values) > 1 else 0.0
    stats['skewness'] = calc_skew(values) if len(values) > 1 else 0.0

    return stats

# Get data
track_list = create_track_list('Tracks.txt')

# for track in track_list[:10]:
#     print(track)

parameter_stats = calculate_statistics(track_list, 'key')
print(parameter_stats)