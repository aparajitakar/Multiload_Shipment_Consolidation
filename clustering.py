import numpy as np
import pandas as pd
# from ComputeDistance import ComputeDistance
import hdbscan
pd.options.mode.chained_assignment = None  # default='warn'

def clustering(df):
    # only clustering orders instead of items
    temp = df.drop_duplicates(subset=['ORDER'], keep='first')

    # split zones
    us_ind = temp[temp['LINE SHIP POSTAL CODE'].str.isdigit() == True].index
    ca_idx = temp[temp['LINE SHIP POSTAL CODE'].str.isdigit() == False].index

    us = temp.loc[us_ind]
    ca = temp.loc[ca_idx]

    # hdbscan
    # initialize parameters for clustering
    # define eps
    # miles = 100  # calculates 100 miles radius threshold
    miles = input('Enter the radius of delivery zones in miles (press enter to process with default value 100): ')
    if miles == '':
        miles = 100
    else:
        miles = int(miles)
    while not (0 < miles < 3000):
        print('Please enter a valid positive number')
        miles = input('Enter the radius of delivery zones in miles (press enter to process with default value 100): ')
        if miles == '':
            miles = 100
        else:
            miles = int(miles)
    kms = miles * 1.60934
    eps = kms / 6371

    # define min_cluster_size
    min_orders = input('Enter the minimum orders in delivery zones (press enter to process with default value 3): ')
    if min_orders == '':
        min_orders = 3
    else:
        min_orders = int(min_orders)
    while not (min_orders >= 3):
        print('Please enter a integer value greater than or equal to 3')
        min_orders = input('Enter the minimum orders of delivery zones (press enter to process with default value 3): ')
        if min_orders == '':
            min_orders = 3
        else:
            min_orders = int(min_orders)

    # for United States
    X_us = us.dropna()
    X1 = us[["Latitude", "Longitude"]].to_numpy()

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_orders, metric='haversine', cluster_selection_epsilon=eps,
                          cluster_selection_method='leaf')
    hdb.fit(np.radians(X1))

    labels = hdb.labels_
    hdb_clusters = len(set(labels))
    clusters_hdb = pd.Series([X1[labels == n] for n in range(hdb_clusters)])
    print('Number of clusters in US: {}'.format(hdb_clusters))

    X_us['cluster_label'] = hdb.fit_predict(np.radians(X1))
    X_us['cluster_label'].replace({-1: np.nan}, inplace=True)

    # For Canada
    X_ca = ca.dropna()
    X2 = X_ca[["Latitude", "Longitude"]].to_numpy()

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_orders, metric='haversine', cluster_selection_epsilon=eps,
                          cluster_selection_method='eom')
    hdb.fit(np.radians(X2))

    labels = hdb.labels_
    hdb_clusters = len(set(labels))
    clusters_hdb = pd.Series([X2[labels == n] for n in range(hdb_clusters)])
    print('Number of clusters in Canada: {}'.format(hdb_clusters))

    X_ca['cluster_label'] = hdb.fit_predict(np.radians(X2))
    X_ca['cluster_label'].replace({-1: np.nan}, inplace=True)

    # concat us clusters and ca clusters
    n_clu_us = X_us['cluster_label'].nunique()
    # n_clu_ca = X_ca['cluster_label'].nunique()
    X_ca.loc[:,'cluster_label'] = X_ca['cluster_label'] + n_clu_us
    temp = pd.concat([X_us, X_ca])
    temp['cluster_label'] = temp['cluster_label'].fillna('LTL')

    # merge temp and df to recover items
    df = df.merge(temp[['ORDER', 'cluster_label']], on='ORDER', how='left')
    # print(df)

    df.to_csv('output/pd_clustered.csv', index=False, header=True)

    X_ca.to_csv("output/ca.csv", index=False, header=True)
    X_us.to_csv("output/us.csv", index=False, header=True)

    return df
