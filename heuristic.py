# import numpy as np
import pandas as pd
from regression import regression
pd.options.mode.chained_assignment = None  # default='warn'

def heuristic(orders):

    # parameter definition
    # truckload volume capacity (in cubic inch)
    TV = 6028992*0.8
    intercept, coef_m, coef_d, coef_w, coef_v = regression()

    # calculate cost of 2-stop TL
    def costTL_2(order1_id, order2_id):
        order1 = orders[orders['ORDER'] == order1_id]
        order2 = orders[orders['ORDER'] == order2_id]

        d = max(order1.DISTANCE.values[0], order2.DISTANCE.values[0])
        w = order1.WEIGHT.values[0] + order2.WEIGHT.values[0]
        v = order1.VOLUME.values[0] + order2.VOLUME.values[0]

        cost_pred = coef_m * 1 + coef_d * d + coef_w * w + coef_v * v + intercept + 150
        return cost_pred

    # calculate cost of 3-stop TL
    def costTL_3(order1_id, order2_id, order3_id):
        order1 = orders[orders['ORDER'] == order1_id]
        order2 = orders[orders['ORDER'] == order2_id]
        order3 = orders[orders['ORDER'] == order3_id]

        d = max(order1.DISTANCE.values[0], order2.DISTANCE.values[0], order3.DISTANCE.values[0])
        w = order1.WEIGHT.values[0] + order2.WEIGHT.values[0] + order3.WEIGHT.values[0]
        v = order1.VOLUME.values[0] + order2.VOLUME.values[0] + order3.VOLUME.values[0]

        cost_pred = coef_m * 1 + coef_d * d + coef_w * w + coef_v * v + intercept + 300
        return cost_pred

    # calculate LTL cost
    def costLTL(order_id):
        order = orders[orders['ORDER'] == order_id]
        d = order.DISTANCE.values[0]
        w = order.WEIGHT.values[0]
        v = order.VOLUME.values[0]
        cost_pred = coef_d * d + coef_w * w + coef_v * v + intercept
        return cost_pred

    # initial a recommendation list for this cluster
    output = pd.DataFrame(columns=['ZONE', 'ORDER 1', 'ORDER 2', 'ORDER 3', 'LTL COST', 'TL COST'])

    # heuristic loop
    for c in orders['cluster_label'].unique():
        if c != 'LTL':
            clu = orders[orders['cluster_label'] == c]
            for i in clu['ORDER'].tolist():
                i_idx = orders[orders['ORDER'] == i].index
                v_i = orders.iloc[i_idx]['VOLUME'].values[0]

                for j in clu['ORDER'].tolist():
                    j_idx = orders[orders['ORDER'] == j].index
                    v_j = orders.iloc[j_idx]['VOLUME'].values[0]
                    if i_idx < j_idx:
                        if (v_i + v_j) <= TV:
                            ltl = costLTL(i) + costLTL(j)
                            tl = costTL_2(i, j)
                            output = output.append({'ZONE': c,
                                                    'ORDER 1': i,
                                                    'ORDER 2': j,
                                                    'LTL COST': ltl,
                                                    'TL COST': tl},
                                                   ignore_index=True)

                    for k in clu['ORDER'].tolist():
                        k_idx = orders[orders['ORDER'] == k].index
                        v_k = orders.iloc[k_idx]['VOLUME'].values[0]
                        if i_idx < j_idx & j_idx < k_idx:
                            if (v_i + v_j + v_k) <= TV:
                                ltl = costLTL(i) + costLTL(j) + costLTL(k)
                                tl = costTL_3(i, j, k)
                                output = output.append({'ZONE': c,
                                                        'ORDER 1': i,
                                                        'ORDER 2': j,
                                                        'ORDER 3': k,
                                                        'LTL COST': ltl,
                                                        'TL COST': tl},
                                                       ignore_index=True)

    return output