# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Page title
st.title('NFL Stats Explorer')

st.markdown("""
This app performs webscrapping of NFL player stats data from [pro-football-reference.com](https://www.pro-football-reference.com).
""")

# Sidebar year input
st.sidebar.header('User Input Features')
selectedYear = st.sidebar.selectbox('Year', list(reversed(range(1990, 2021))))

# Webscrapping
@st.cache
def loadData(year):
    url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
    html = pd.read_html(url, header=1)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers
    raw = raw.fillna(0)
    playerStats = raw.drop(['Rk'], axis=1)
    return playerStats
playerStats = loadData(selectedYear)

# Sidebar team selection
teams = sorted(playerStats.Tm.unique())
selectedTeam = st.sidebar.multiselect('Team', teams, teams)

# Sidebar position selection
positions = ['RB','QB','WR','FB','TE']
selectedPosition = st.sidebar.multiselect('Position', positions, positions)

# Filtering data
df_selectedTeam = playerStats[(playerStats.Tm.isin(selectedTeam)) & (playerStats.Pos.isin(selectedPosition))]

st.header('Display Player Stats of the selection')
st.write('Data Dimension: ' + str(df_selectedTeam.shape[0]) + ' rows and ' + str(df_selectedTeam.shape[1]) + ' columns')
st.dataframe(df_selectedTeam)

# Download selected data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def fileDownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings to bytes conversion
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(fileDownload(df_selectedTeam), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selectedTeam.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)
