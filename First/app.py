from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from model import run_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)

    df, scaled_data = run_model(df)

    # HEATMAP
    plt.figure()
    sns.heatmap(df.corr(), annot=False)
    plt.title("Heatmap")
    plt.savefig("static/heatmap.png")
    plt.close()

    # FRAUD GRAPH
    plt.figure()
    df['Fraud'].value_counts().plot(kind='bar')
    plt.title("Fraud vs Normal")
    plt.savefig("static/fraud.png")
    plt.close()

    # PCA GRAPH
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['Cluster'])
    plt.title("Clustering (PCA)")
    plt.savefig("static/pca.png")
    plt.close()

    # CLUSTER GRAPH
    plt.figure()
    df['Cluster'].value_counts().plot(kind='bar')
    plt.title("Cluster Distribution")
    plt.savefig("static/cluster.png")
    plt.close()

    # FRAUD IN CLUSTER
    plt.figure()
    sns.countplot(x='Cluster', hue='Fraud', data=df)
    plt.title("Fraud in Clusters")
    plt.savefig("static/fraud_cluster.png")
    plt.close()

    fraud_count = df['Fraud'].sum()
    result = "⚠️ Fraud Detected" if fraud_count > 0 else "✅ No Fraud"

    return render_template("index.html",
                           tables=[df.head().to_html(classes='data')],
                           heatmap="static/heatmap.png",
                           fraud="static/fraud.png",
                           pca="static/pca.png",
                           cluster="static/cluster.png",
                           fraud_cluster="static/fraud_cluster.png",
                           result=result)


if __name__ == '__main__':
    app.run(debug=True)