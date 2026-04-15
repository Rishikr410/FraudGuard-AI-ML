from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

def run_model(df):

    # Preprocessing
    df = df.fillna(df.mean(numeric_only=True))
    df = df.select_dtypes(include=['float64', 'int64'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Fraud detection
    iso = IsolationForest(contamination=0.02, random_state=42)
    df['Fraud'] = iso.fit_predict(scaled_data)
    df['Fraud'] = df['Fraud'].map({1: 0, -1: 1})

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    return df, scaled_data