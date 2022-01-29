import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("")
df["abc"].hist()

z=np.array(list(df['total_origin_price']))
z = z.reshape(-1,1)

km = KMeans()
km.fit(z)
km.cluster_centers_