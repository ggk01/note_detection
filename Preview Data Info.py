import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("aligned_features_always.csv")

# ---- Feature Distributions ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['MIDI_Note'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of MIDI Notes')

sns.histplot(df['Pitch'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Pitch')

sns.histplot(df['Power'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Power')

sns.histplot(df['Spectral_Centroid'], bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Spectral Centroid')

plt.tight_layout()
plt.show()

# ---- Class Distribution of MIDI Notes ----
midi_counts = df['MIDI_Note'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
sns.barplot(x=midi_counts.index, y=midi_counts.values, color="blue")
plt.xlabel("MIDI Note")
plt.ylabel("Count")
plt.title("Class Distribution of MIDI Notes")
plt.xticks(rotation=90)
plt.show()

# ---- PCA Visualization ----
features = ["Pitch", "Power", "Spectral_Centroid"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["MIDI_Note"], palette="coolwarm", alpha=0.6, edgecolor=None, legend=False)
plt.title("PCA Visualization of MIDI Note Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
