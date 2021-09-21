import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.linalg as ln
from ot_simple_connector.connector import Connector

def mySlice(df, d, metric):
    dd = datetime(d.year, d.month, d.day)
    temp = df[df['Description'] == metric].loc[:, ['_time', 'tt', 'value']]
    temp = temp[temp['_time'] > dd.timestamp()]
    dd = d + timedelta(days=1)
    dd = datetime(dd.year, dd.month, dd.day)
    temp = temp[temp['_time'] < dd.timestamp()]
    return temp

def ReceiveDataFromEva(query, fileName):
    KNS_cache_filename = 'F:\\Work iSG\\DataPPD\\' + fileName + '.csv'
    KNS_df = None
    try:
        KNS_df = pd.read_csv(KNS_cache_filename)
    except:
        pass

    if KNS_df is None:
        conn = Connector(host="192.168.4.65", port='80', user="admin", password="12345678")
        job = conn.jobs.create(query_text=query, cache_ttl=60, tws=0, twf=0)
        print(job.status)
        res = job.dataset.load()
        KNS_df = pd.DataFrame(res)
        KNS_df.to_csv(KNS_cache_filename)

    return KNS_df


def NOM(df, KNS_no, pump_no):
    if KNS_no == 1:
        serial_to_no = {'1': 50, '2': 260, '3': 421, '4': 806}
    elif KNS_no == 2:
        serial_to_no = {'1': 293, '2': 310, '3': 309, '4': 470}
    else:
        print('There is no such KNS in data!')
        return None
    PQ_nom = df[df['serial_number'] == serial_to_no[pump_no]].loc[:, ['model', 'P', 'Q', 'eff', 'power']]
    PQ_nom = PQ_nom.drop_duplicates()
    PQ_nom = PQ_nom.dropna()

    return PQ_nom


def NRH(df, pump_no):
    pump = 'НА-' + pump_no + ' '

    P_out = df[df['Description'] == pump + 'Давление на выкиде'].loc[:, ['T', 'value']]
    P_out.index = P_out['T']
    P_out.drop(['T'], axis=1, inplace=True)
    P_in = df[df['Description'] == pump + 'Давление на приеме'].loc[:, ['T', 'value']]
    P_in.index = P_in['T']
    P_in.drop(['T'], axis=1, inplace=True)
    P = P_out - P_in
    P.reset_index(inplace=True)
    P.rename(columns={'value': 'P'}, inplace=True)
    Q = df[df['Description'] == pump + 'Объём мгн.'].loc[:, ['T', 'value']]
    Q.rename(columns={'value': 'Q'}, inplace=True)
    PQ = pd.merge(Q, P, on='T')
    PQ = PQ[PQ['Q'] > 0.0]
    PQ.sort_values(by='T', inplace=True)

    return PQ

def myPlot(df1, df2, KNS_no, pump_no, mode=1):
    PQ = NRH(df1, pump_no)
    PQ_nom = NOM(df2, KNS_no, pump_no)
    fPQ = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['P']), kind='linear')
    feff = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['eff']), kind='linear')
    feff_calc = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['P'] * PQ_nom['Q'] / PQ_nom['power']), kind='linear')

    x = [list(PQ['Q']), list(PQ_nom['Q']),
         list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10)),
         list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10))]
    y = [list(PQ['P']), list(PQ_nom['eff']), list(PQ_nom['P'])]

    min_x = min(PQ['Q'].min(), PQ_nom['Q'].min())
    min_y = min(PQ['P'].min(), PQ_nom['P'].min())
    max_x = max(PQ['Q'].max(), PQ_nom['Q'].max())
    max_y = max(PQ['P'].max(), PQ_nom['P'].max())

    if mode == 1:
        fig, ax = plt.subplots()
        plt.axis([min_x - 10, max_x + 10, min_y - 10, max_y + 10])
        ax.plot(x[0], y[0], 'r*', x[2], y[2], 'o', x[3], fPQ(x[3]), 'g--')
        plt.legend(['НРХ', 'НРХ_ном', 'НРХ_ном'], loc='best')
        plt.xlabel('Q')
        plt.ylabel('P')
        plt.grid(True)

    if mode == 2:
        fig, ax = plt.subplots()
        plt.axis([PQ_nom['Q'].min() - 10, PQ_nom['Q'].max() + 10, 0, 100])
        plt.plot(x[1], y[1], 'o', x[4], feff(x[4]), 'g--', x[4], feff_calc(x[4]), 'r-')
        plt.legend(['КПД', 'КПД', 'КПД_калк'], loc='best')
        plt.xlabel('Q')
        plt.ylabel('eff')
        plt.grid(True)

    return fig


def SqReg(x, y):
    # x and y should be "np.array"
    A = [(x**4).sum(), (x**3).sum(), (x**2).sum(),
         (x**3).sum(), (x**2).sum(), x.sum(),
         (x**2).sum(), x.sum(), x.size]
    A = np.array(A).reshape(3, 3)

    b = [(y * x**2).sum(), (y * x).sum(), y.sum()]
    b = np.array(b).reshape(3, 1)

    # return ln.inv(A.T.dot(A)).dot(A.T).dot(b)
    return ln.inv(A).dot(b)