import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import interpolate
from ot_simple_connector.connector import Connector
from tools import ReceiveDataFromEva, NRH, NOM, mySlice

"""
# Сводная информация по КНС
"""

flag = 0

firstOp = pd.Series(['Данные по насосам на КНС', 'Закачка на конец суток'])
option = st.sidebar.selectbox(
    'Отображаемые данные',
     firstOp)

if option == firstOp.iloc[0]:
    opt1 = pd.Series([1,2])
    option1 = st.sidebar.selectbox(
    'Выбор КНС',
     opt1)

    opt2 = pd.Series([1,2,3,4])
    option2 = st.sidebar.selectbox(
        'Выбор насоса',
        opt2)

    opt3 = pd.Series([1,2])
    option3 = st.sidebar.selectbox(
        'НРХ или КПД',
        opt3)

    if st.sidebar.button('Отобразить'):
        query1 = '| readFile format=parquet path=FS/PPD/kns' + str(option1) + '_long '
        query1 += '| fields _time, Description, value, EU'
        dfKNS = ReceiveDataFromEva(query1, 'ppdKNS'+str(option1))

        query2 = '| readFile format=parquet path=FS/PPD/pump_curves_water '
        query2 += '| eval head_p=head_m/10 ' + '| eval P_nomp=P_nom/10 '
        query2 += '| fields _time, model, serial_number, P_nomp, Q_nom, head_p,\
                    flow_m3_per_hour, efficiency_percent, power_kW'
        dfNom = ReceiveDataFromEva(query2, 'ppdNOM')

        dfKNS['T'] = dfKNS['_time'].map(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        dfNom['T'] = dfNom['_time'].map(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        dfNom.rename(columns={'P_nomp': 'P_nom', 'head_p': 'P', 'flow_m3_per_hour': 'Q', 'efficiency_percent': 'eff',
                              'power_kW': 'power'}, inplace=True)

        KNS_no = option1
        pump_no = str(option2)
        mode = option3

        PQ = NRH(dfKNS, pump_no)
        PQ_nom = NOM(dfNom, KNS_no, pump_no)
        PQ.sort_values(by='Q', inplace=True)
        PQ_nom.sort_values(by='Q', inplace=True)
        fPQ = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['P']), kind='linear')
        feff = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['eff']), kind='linear')
        feff_calc = interpolate.interp1d(list(PQ_nom['Q']), list(2.7 * PQ_nom['P'] * PQ_nom['Q'] / PQ_nom['power']),
                                         kind='linear')

        x = [list(PQ['Q']), list(PQ_nom['Q']),
             list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10)),
             list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10))]
        y = [list(PQ['P']), list(PQ_nom['eff']), list(PQ_nom['P'])]

        min_x = min(PQ['Q'].min(), PQ_nom['Q'].min())
        min_y = min(PQ['P'].min(), PQ_nom['P'].min())
        max_x = max(PQ['Q'].max(), PQ_nom['Q'].max())
        max_y = max(PQ['P'].max(), PQ_nom['P'].max())

        if mode == 1:
            model = PQ_nom['model'].unique()
            m = ' || '
            for i in range(model.size):
                m += model[i] + ' || '
            st.write('Модель насоса:' + m)
            fig, ax = plt.subplots()
            plt.axis([min_x - 10, max_x + 10, min_y - 10, max_y + 10])
            ax.plot(x[0], y[0], 'r*', x[1], y[2], 'o')
            plt.legend(['НРХ_КНС', 'НРХ_ном'], loc='best')
            plt.xlabel('Q')
            plt.ylabel('P')
            plt.grid(True)

        if mode == 2:
            model = PQ_nom['model'].unique()
            m = ' || '
            for i in range(model.size):
                m += model[i] + ' || '
            st.write('Модель насоса:' + m)
            fig, ax = plt.subplots()
            plt.axis([PQ_nom['Q'].min() - 10, PQ_nom['Q'].max() + 10, 0, 100])
            ax.plot(x[1], y[1], 'o', x[1], feff_calc(x[1]), 'r-')
            plt.legend(['КПД', 'КПД_калк'], loc='best')
            plt.xlabel('Q')
            plt.ylabel('eff')
            plt.grid(True)

        st.pyplot(fig)
    else:
        st.write('Данные не загружены')
elif option == firstOp.iloc[1]:
    option1 = st.sidebar.selectbox(
        'Выбор КНС',
        (1, 2))
    d1 = st.date_input(
        "Выбор даты",
        date(2021, 8, 30))
    if st.sidebar.button('Отобразить'):
        query1 = '| readFile format=parquet path=FS/PPD/kns' + str(option1) + '_long '
        query1 += '| fields _time, Description, value, EU'
        dfKNS = ReceiveDataFromEva(query1, 'ppdKNS'+str(option1))
        dfKNS['tt'] = dfKNS['_time'].map(lambda x: datetime.utcfromtimestamp(x))

        pumpQ = []
        for i in np.arange(1, 5, 1):
            temp = mySlice(dfKNS, d1, 'НА-'+str(i)+' Объём мгн.')
            temp.sort_values(by='tt', inplace=True)
            sum = 0
            for j in range(temp.shape[0]-1):
                h = temp.iloc[j,:]['tt']-temp.iloc[j+1,:]['tt']
                sum += (temp.iloc[j,:]['value']+temp.iloc[j+1,:]['value'])/2 * np.abs(h.total_seconds())/60/60
            pumpQ += [sum]
        pumpQ = pd.Series(pumpQ)
        st.write('Закачка за день на каждом насосе')
        st.write(pd.DataFrame(pumpQ))

        temp3 = mySlice(dfKNS, d1, 'КНС Ожидаемый объём закачки на тек.сутки').loc[:, ['value']].mean()
        st.metric(label='КНС Ожидаемый объём закачки на тек.сутки', value=float(temp3))

        temp1 = float(pd.Series(pumpQ).sum())
        st.metric(label='Суммарная вычисленная закачка', value=temp1, delta=float(temp1 - temp3))

        temp2 = mySlice(dfKNS, d1, 'КНС Расчётный объём закачки на тек.сутки').sort_values(by='tt', ascending=False).iloc[0, :]['value']
        st.metric(label='КНС Расчётный объём закачки на тек.сутки', value=float(temp2), delta=float(temp2-temp3))

        for i in range(1,5):
            temp = mySlice(dfKNS, d1, 'НА-' + str(i) + ' Состояние насоса')
            temp.sort_values(by='tt', inplace=True)
            chart_data = temp.loc[:, ['tt', 'value']]
            chart_data.index = chart_data['tt']
            chart_data.drop(['tt'], axis=1, inplace=True)
            st.write('Насос '+str(i))
            st.line_chart(chart_data)
    else:
        st.write('Данные не загружены')