import pandas as pd
from sklearn.linear_model import ElasticNet
pd.options.mode.chained_assignment = None  # default='warn'

def regression():
    # import historical shipment dataset
    # Daikin can update the lasso model on a new historical shipment dataset
    shipment = pd.read_excel('input/shipment.xlsx', header=0)

    # drop unwanted columns
    shipment = shipment[['Shipment ID', 'Mode', 'Loaded Distance', 'Total Gross Weight', 'Total Gross Volume', 'Total Actual Cost']]
    # drop unwanted rows
    # remove rows whose Mode is AIR or PARCEL
    shipment = shipment.drop(shipment[(shipment['Mode'] == 'AIR') | (shipment['Mode'] == 'PARCEL')].index)

    # rename columns
    shipment = shipment.rename(columns={'Mode': 'MODE',
                                        'Loaded Distance': 'DISTANCE',
                                        'Total Gross Weight': 'WEIGHT',
                                        'Total Gross Volume': 'VOLUME'})

    # re-construct categorical data (0: MODE == 'LTL'; 1: MODE =='TL')
    shipment['MODE'] = shipment['MODE'].astype('category').cat.codes
    shipment['MODE'] = shipment['MODE'].astype('category')

    shipment = shipment[shipment['Total Actual Cost'] >= 0]
    outlier = shipment['Total Actual Cost'] > 10000
    shipment = shipment.drop(shipment[outlier].index)

    # initialize model parameter
    Elastic_reg = ElasticNet(alpha=1, l1_ratio=0.5)

    # fit regression model
    X = shipment.drop(columns=['Shipment ID', 'Total Actual Cost'])
    y = shipment[['Total Actual Cost']]
    Elastic_reg.fit(X, y)

    # extract coefficients and intercept
    # coef_mode, coef_distance, coef_weight, coef_volume
    coef_m, coef_d, coef_w, coef_v = Elastic_reg.coef_.transpose()
    intercept = Elastic_reg.intercept_[0]

    return intercept, coef_m, coef_d, coef_w, coef_v