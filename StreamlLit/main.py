import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import interpolate
from ot_simple_connector.connector import Connector
from tools import ReceiveDataFromEva, NRH, NOM, mySlice, SqReg

"""
# Сводная информация по КНС
"""

flag = 0

firstOp = pd.Series(['Данные по насосам на КНС', 'Закачка на конец суток', 'Отклонения', 'Проверка',
                     'Насосы'])
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

        PQ = NRH(dfKNS, pump_no)
        PQ_nom = NOM(dfNom, KNS_no, pump_no)
        # PQ.sort_values(by='Q', inplace=True)
        PQ_nom.sort_values(by='Q', inplace=True)
        fPQ = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['P']), kind='linear')
        feff = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['eff']), kind='linear')
        feff_calc = interpolate.interp1d(list(PQ_nom['Q']), list(2.7 * PQ_nom['P'] * PQ_nom['Q'] / PQ_nom['power']),
                                         kind='linear')

        x = [list(PQ['Q']), list(PQ_nom['Q']),
             list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10)),
             list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10))]
        y = [list(PQ['P']), list(PQ_nom['eff']), list(PQ_nom['P']), list(PQ_nom['power'])]

        min_x = min(PQ['Q'].min(), PQ_nom['Q'].min())
        min_y = min(PQ['P'].min(), PQ_nom['P'].min())
        max_x = max(PQ['Q'].max(), PQ_nom['Q'].max())
        max_y = max(PQ['P'].max(), PQ_nom['P'].max())

        model = list(PQ_nom['model'].unique())
        # m = ' || '
        # for i in range(len(model)):
            # m += model[i] + ' || '
        # st.write('Модель насоса:' + m)

        # option3 = st.selectbox(
            # 'Выбор модели насоса',
            # model)

        for option3 in model:
            st.write(option3)
            fig = plt.figure(figsize=(12, 8))
            host = fig.add_subplot(111)

            par0 = host.twinx()
            par1 = host.twinx()
            par2 = host.twinx()

            host.set_xlim(min_x - 10, max_x + 10)
            host.set_ylim(min_y - 10, max_y + 10)
            par1.set_ylim(0, 100)
            par2.set_ylim(PQ_nom[PQ_nom['model']==option3].loc[:,['power']].min()[0],
                      PQ_nom[PQ_nom['model']==option3].loc[:,['power']].max()[0])

            host.set_xlabel("Q")
            host.set_ylabel("PRESSURE")
            par1.set_ylabel("EFFICIENCY")
            par2.set_ylabel("POWER")

            # cm = plt.get_cmap('summer')
            # host.set_color_cycle([cm(1. * i / (len(x[0]) - 1)) for i in range(len(x[0]) - 1)])
            R = np.arange(0, 1, 1/len(x[0]))
            for i in range(len(x[0])):
                color = (R[i], 0, 0)
                host.plot(x[0][i], y[0][i], '.', color=color, markersize=8, alpha=1)
            # host.scatter(x[0], y[0], c=x[0], marker='o', cmap='plasma', label='P')
            p1, = host.plot(list(PQ_nom[PQ_nom['model']==option3].loc[:,['Q']]['Q']),
                        list(PQ_nom[PQ_nom['model']==option3].loc[:,['P']]['P']),marker='o',color='b',label="P_test")
            p2, = par1.plot(list(PQ_nom[PQ_nom['model']==option3].loc[:,['Q']]['Q']),
                        list(PQ_nom[PQ_nom['model']==option3].loc[:,['eff']]['eff']),marker='o',color='g',label="Eff")
            p3, = par2.plot(list(PQ_nom[PQ_nom['model']==option3].loc[:,['Q']]['Q']),
                        list(PQ_nom[PQ_nom['model']==option3].loc[:,['power']]['power']),marker='o',color='y',label="Power")

            lns = [p1, p2, p3]
            host.legend(handles=lns, loc='best')

            par2.spines['right'].set_position(('outward', 60))
            par0.yaxis.set_ticks([])

            host.yaxis.label.set_color(p1.get_color())
            par1.yaxis.label.set_color(p2.get_color())
            par2.yaxis.label.set_color(p3.get_color())
            host.grid(True)

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
elif option == firstOp.iloc[2]:
    option1 = st.sidebar.selectbox(
        'Выбор КНС',
        [1, 2])
    option2 = st.sidebar.selectbox(
        'Выбор насоса',
        [1, 2, 3, 4])
    d1 = st.date_input(
        "Начальная дата",
        date(2020, 8, 7))
    d2 = st.date_input(
        "Конечная дата",
        date(2021, 9, 17))
    cin = st.text_input('Размер окна (пока минимум 10)', '30')
    window = int(cin)
    if st.sidebar.button('Отобразить'):
        query1 = '| readFile format=parquet path=FS/PPD/kns' + str(option1) + '_pump_pressures_long'
        query1 += '| fields _time, Description, value'
        dfKNS = ReceiveDataFromEva(query1, 'ppdKNS' + str(option1) + '_long')
        dfKNS['T'] = dfKNS['_time'].map(lambda x: datetime.utcfromtimestamp(x))
        st.write('Имеются данные с ' + dfKNS['T'].min().strftime('%Y-%m-%d') + \
                 ' по ' + dfKNS['T'].max().strftime('%Y-%m-%d'))
        PQ = NRH(dfKNS, str(option2))
        PQ.dropna(inplace=True)
        PQ = PQ[PQ['T'] >= datetime(d1.year, d1.month, d1.day)]
        PQ = PQ[PQ['T'] <= datetime(d2.year, d2.month, d2.day)]
        N = PQ['P'].count()
        st.write('Количество точек за выбранный период: ' + \
                 str(N))
        # window = 20
        position = 0
        ind = []
        a = []
        b = []
        c = []
        z = []
        while position < N:
            if (position + window <= N-1):
                data = PQ.iloc[position:position+window, :]
                position += 1
                x = np.array(data['Q'])
                y = np.array(data['P'])
                ind.append(data['T'].iloc[0])
                res = SqReg(x, y)
                a.append(float(res[0]))
                b.append(float(res[1]))
                c.append(float(res[2]))
                z.append([float(res[0]), float(res[1]), float(res[2])])
            else:
                if PQ['P'].iloc[position:].count() >= 10:
                    data = PQ.iloc[position:, :]
                    position += 1
                    x = np.array(data['Q'])
                    y = np.array(data['P'])
                    ind.append(data['T'].iloc[0])
                    res = SqReg(x, y)
                    a.append(float(res[0]))
                    b.append(float(res[1]))
                    c.append(float(res[2]))
                    z.append([float(res[0]), float(res[1]), float(res[2])])
                else:
                    break
        a = pd.Series(a, index=ind)
        b = pd.Series(b, index=ind)
        c = pd.Series(c, index=ind)
        z = pd.DataFrame(z, columns=['a', 'b', 'c'], index=ind)
        st.write('Коэффициент A')
        st.line_chart(a)
        st.write('Коэффициент B')
        st.line_chart(b)
        st.write('Коэффициент C')
        st.line_chart(c)

        fig = plt.figure(figsize=(12, 8))
        host = fig.add_subplot(111)
        host.set_xlabel("Q")
        host.set_ylabel("PRESSURE")
        # host.set_xlim(PQ['Q'].min() - 10, PQ['Q'].max() + 10)
        # host.set_ylim(PQ['P'].max() - 10, PQ['P'].min() + 10)
        p0, = host.plot(list(PQ['Q']), list(PQ['P']), 'ro', label="P")
        host.legend(handles=[p0], loc='best')
        host.grid(True)
        st.pyplot(fig)
    else:
        st.write('Данные не загружены')
elif option == firstOp.iloc[3]:
    option1 = st.sidebar.selectbox(
        'Выбор КНС',
        [1, 2])
    option2 = st.sidebar.selectbox(
        'Выбор насоса',
        [1, 2, 3, 4])
    d1 = st.date_input(
        "Начальная дата",
        date(2020, 8, 7))
    d2 = st.date_input(
        "Конечная дата",
        date(2021, 9, 17))
    cin = st.text_input('Размер окна', '30')
    window = int(cin)
    if st.sidebar.button('Отобразить'):
        query1 = '| readFile format=parquet path=FS/PPD/kns' + str(option1) + '_pump_pressures_long'
        query1 += '| fields _time, Description, value'
        dfKNS = ReceiveDataFromEva(query1, 'ppdKNS' + str(option1) + '_long')
        dfKNS['T'] = dfKNS['_time'].map(lambda x: datetime.utcfromtimestamp(x))
        st.write('Имеются данные с ' + dfKNS['T'].min().strftime('%Y-%m-%d') + \
                 ' по ' + dfKNS['T'].max().strftime('%Y-%m-%d'))
        PQ = NRH(dfKNS, str(option2))
        PQ.dropna(inplace=True)

        PQ = PQ[PQ['T'] >= datetime(d1.year, d1.month, d1.day)]
        PQ = PQ[PQ['T'] <= datetime(d2.year, d2.month, d2.day)]
        N = PQ['P'].count()
        st.write('Количество точек за выбранный период: ' + \
                 str(N))
        position = round(np.random.rand()*(N-window-1))
        data = PQ.iloc[position:position + window, :]
        x = np.array(data['Q'])
        y = np.array(data['P'])
        res = SqReg(x, y)
        st.write('A = ' + str(res[0]) + '; B = ' + str(res[1]) + '; C = ' + str(res[2]))

        f = lambda t: res[0]*t**2 + res[1]*t + res[2]
        fig = plt.figure(figsize=(12, 8))
        host = fig.add_subplot(111)
        host.set_xlabel("Q")
        host.set_ylabel("PRESSURE")
        host.plot(x, y, 'ro', np.sort(x), f(np.sort(x)), 'b--')
        host.grid(True)
        st.pyplot(fig)
    else:
        st.write('Данные не загружены')
elif option == firstOp.iloc[4]:
    option1 = st.sidebar.selectbox(
        'Выбор КНС',
        [1, 2])
    st.write('Комментарий: чем чернее, тем старее и чем краснее, тем новее.')
    if st.sidebar.button('Отобразить'):
        query1 = '| readFile format=parquet path=FS/PPD/kns' + str(option1) + '_long '
        query1 += '| fields _time, Description, value, EU'
        dfKNS = ReceiveDataFromEva(query1, 'ppdKNS' + str(option1))

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
        nomer = ['1', '2', '3', '4']

        for pump_no in nomer:
            PQ = NRH(dfKNS, pump_no)
            PQ_nom = NOM(dfNom, KNS_no, pump_no)
            PQ_nom.sort_values(by='Q', inplace=True)
            fPQ = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['P']), kind='linear')
            feff = interpolate.interp1d(list(PQ_nom['Q']), list(PQ_nom['eff']), kind='linear')
            feff_calc = interpolate.interp1d(list(PQ_nom['Q']), list(2.7 * PQ_nom['P'] * PQ_nom['Q'] / PQ_nom['power']),
                                         kind='linear')

            x = [list(PQ['Q']), list(PQ_nom['Q']),
                list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10)),
                list(np.arange(PQ_nom['Q'].min(), PQ_nom['Q'].max(), (PQ_nom['Q'].max() - PQ_nom['Q'].min()) / 10))]
            y = [list(PQ['P']), list(PQ_nom['eff']), list(PQ_nom['P']), list(PQ_nom['power'])]

            min_x = min(PQ['Q'].min(), PQ_nom['Q'].min())
            min_y = min(PQ['P'].min(), PQ_nom['P'].min())
            max_x = max(PQ['Q'].max(), PQ_nom['Q'].max())
            max_y = max(PQ['P'].max(), PQ_nom['P'].max())

            model = list(PQ_nom['model'].unique())

            option3 = model[0]
            st.write('Насос ' + pump_no + ', модель ' + option3)
            fig = plt.figure(figsize=(12, 8))
            host = fig.add_subplot(111)

            par0 = host.twinx()
            par1 = host.twinx()
            par2 = host.twinx()

            host.set_xlim(min_x - 10, max_x + 10)
            host.set_ylim(min_y - 10, max_y + 10)
            par1.set_ylim(0, 100)
            par2.set_ylim(PQ_nom[PQ_nom['model'] == option3].loc[:, ['power']].min()[0],
                          PQ_nom[PQ_nom['model'] == option3].loc[:, ['power']].max()[0])
            host.set_xlabel("Q")
            host.set_ylabel("PRESSURE")
            par1.set_ylabel("EFFICIENCY")
            par2.set_ylabel("POWER")

            R = np.arange(0, 1, 1 / len(x[0]))
            for i in range(len(x[0])):
                color = (R[i], 0, 0)
                host.plot(x[0][i], y[0][i], '.', color=color, markersize=8, alpha=1)
            p1, = host.plot(list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['Q']]['Q']),
                            list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['P']]['P']), marker='o', color='b',
                            label="P_test")
            p2, = par1.plot(list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['Q']]['Q']),
                            list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['eff']]['eff']), marker='o', color='g',
                            label="Eff")
            p3, = par2.plot(list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['Q']]['Q']),
                            list(PQ_nom[PQ_nom['model'] == option3].loc[:, ['power']]['power']), marker='o', color='y',
                            label="Power")
            lns = [p1, p2, p3]
            host.legend(handles=lns, loc='best')
            par2.spines['right'].set_position(('outward', 60))
            par0.yaxis.set_ticks([])
            host.yaxis.label.set_color(p1.get_color())
            par1.yaxis.label.set_color(p2.get_color())
            par2.yaxis.label.set_color(p3.get_color())
            host.grid(True)
            st.pyplot(fig)
    else:
        st.write('Данные не загружены')