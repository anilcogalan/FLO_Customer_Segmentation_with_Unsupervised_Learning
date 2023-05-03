###############################################################
# Customer Segmentation with Unsupervised Learning
###############################################################

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("flo_data_20k.csv")

df.head()
df.info()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]') # en son kaç gün önce alışveriş yaptı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

# SKEWNESS

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    plt.show(block=True)
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))



cols_to_plot = ['order_num_total_ever_online',
                'order_num_total_ever_offline',
                'customer_value_total_ever_offline',
                'customer_value_total_ever_online',
                'recency',
                'tenure']

for i in cols_to_plot:
    check_skew(model_df, i)

# NOTE: If there is skewness, logarithm transformation should be done.

# Log transformation to ensure normal distribution

def apply_log1p_transform(df, cols):
    """Apply np.log1p transformation to specified columns in a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to transform.
        cols (list of str): A list of column names to transform.

    Returns:
        pandas.DataFrame: The transformed dataframe.

    Example:
        cols_to_transform = ['order_num_total_ever_online', 'order_num_total_ever_offline',
                             'customer_value_total_ever_offline', 'customer_value_total_ever_online',
                             'recency', 'tenure']
        transformed_df = apply_log1p_transform(model_df, cols_to_transform)

    """
    transformed_df = df.copy()
    transformed_df[cols] = np.log1p(df[cols])
    return transformed_df

cols_to_transform = cols_to_plot.copy()

transformed_df = apply_log1p_transform(model_df, cols_to_transform)

# Scaling

sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

# K-MEANS

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)

# Model & Customer_Segments

k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_
segments

final_df = df[["master_id",
               "order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]

final_df["segment"] = segments
final_df.head()


# Analyze each segment statistically

for i in cols_to_transform:
    final_df.groupby("segment").agg({i:["mean","min","max"]})


# The optimum number of clusters

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)

# Model & Customer_Segments

hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()

# Analyze each segment statistically
final_df.groupby("segment").agg({i:["mean","min","max"]})

final_df["segment"].value_counts()

# OUTPUT:
# 0    8637
# 3    4618
# 1    3844
# 2    2076
# 4     770
# Name: segment, dtype: int64