import streamlit as st
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import interpolate

"""
# My first app
Here's our first attempt at using data to create a table:
"""

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

if st.sidebar.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option

x = np.linspace(0, 4, 12)
y = np.cos(x**2/3+4)
f2 = interpolate.interp1d(x, y, kind = 'cubic')
xnew = np.linspace(0, 4,30)
fig, ax = plt.subplots()
ax.plot(x, y, 'ro', xnew, f2(xnew), 'g--')
plt.xlabel('x')
plt.ylabel('y')
st.pyplot(fig)

if st.sidebar.button('Say hello'):
   # st.write('Why hello there')
   chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

   chart_data
else:
   st.write('Goodbye')

number = st.number_input('Insert a number')
st.write('The current number is ', number)

d = st.date_input(
    "When's your birthday",
    datetime.date(2019, 7, 6))
st.write('Your birthday is:', d)

left_column, right_column = st.columns(2)
pressed = left_column.button('Press me?')
if pressed:
  right_column.write("Woohoo!")