import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# Insert icon of web app
icon = Image.open("icon.png")
#  Page Layout
st.set_page_config(page_title="Water Properties Analysi", page_icon=icon)
# Insert image
logo = Image.open("background.png")
st.image(logo, width=100, use_column_width=True)

img = Image.open("icon.png")
st.sidebar.image(img)

st.title("WATER ANALYSIS & SCALE INTENDENCY PREDICTION BY MACHINE LEARNING")

tabs = ["Plot data", "SI Calculation", "Machine Learning", "About"]
st.sidebar.subheader("App Navigation")
page = st.sidebar.radio("Select your page", tabs)
upload_file = st.sidebar.file_uploader(label="Please upload your CSV or Excel file!", type=['csv', 'xlsx'])

def get_classifier(clf_name):
    if clf_name == "KNN":
        clf = KNeighborsRegressor()
    elif clf_name == "SVM":
        clf = SVR(kernel='rbf')
    elif clf_name == "DecisionTree":
        clf = DecisionTreeRegressor()
    elif clf_name == "Linear":
        clf = LinearRegression()
    else:
        clf = RandomForestRegressor()
    return clf


def genericml(reg):
    pipemodel = Pipeline([
        #   ('scl', StandardScaler()),
        ('reg', reg)
    ])
    pipemodel.fit(X_train, y_train)
    return pipemodel


def plot_si(name):
    fig5 = plt.figure(figsize=(15, 5))
    plt.scatter(df['Sampledate'], df[name], c='Red', alpha=0.4,
                cmap="viridis")
    plt.xlabel("Date", fontsize=30)
    plt.ylabel(si_name, fontsize=30)
    st.pyplot(fig5)


def plotter(model, modelname):
    fig4, ax = plt.subplots(7, 2, figsize=(15, 30))
    row = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    col = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    date_form = DateFormatter("%Y/%m")
    df['modelname'] = pipe.predict(df[predictors])
    for i, well in enumerate(well_names):
        dfpred = df[df['Well'] == well]
        ax[row[i], col[i]].scatter(dfpred['Sampledate'], dfpred['LSI'], label="Data")
        ax[row[i], col[i]].plot(dfpred['Sampledate'], dfpred['modelname'], color='green', label="Training")
        ax[row[i], col[i]].set_title(well)
        ax[row[i], col[i]].xaxis.set_major_formatter(date_form)
        ax[row[i], col[i]].set_ylabel('LSI')
        ax[row[i], col[i]].legend()
    st.pyplot(fig4)


def load_data():
    global df
    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file).fillna(0)
        except Exception as e:
            print(e)
            df = pd.read_excel(upload_file).fillna(0)
            df['Sampledate'] = pd.to_datetime(df['Sampledate'])
    df.dropna(inplace=True)
    df.drop_duplicates(subset='Sampledate', keep='last')
    df.sort_values('Sampledate', inplace=True)
    df['A'] = (np.log10(df['Tds']) - 1) / 10
    df['B'] = -13.12 * np.log10((df['WBT'] + 273)) + 34.55
    df['C'] = np.log10(df['Ca2+']) - 0.4
    df['D'] = np.log10(0.82 * df['HCO3-'] + 1.667 * df['CO32-'])
    df['pHs'] = (9.3 + df['A'] + df['B']) - (df['C'] + df['D'])
    df['LSI'] = df['pH'] - df['pHs']
    df['RSI'] = 2 * df['pHs'] - df['pH']
    df['PhEQ'] = 1.465 * np.log10(0.82 * df['HCO3-'] + 1.667 * df['CO32-']) + 4.54
    df['PSI'] = 2 * df['pH']-df['pHs']
    return df

def try_read_df(f):
    try:
        return pd.read_csv(f)
    except:
        return pd.read_excel(f)
  
if page == "Plot data":
    with st.sidebar:
        st.write("# Plot water properties")
    try:
        df = load_data()
        st.write(df)
        # Plot water properties
        st.write("# Plot water properties by well")
        well_names = df['Well'].unique().tolist()
        well_selected = st.selectbox("Select well:", well_names)
        data_w = (df[df['Well'] == well_selected]).reset_index()
        all_symbols = ("pH", "Na+", "K+", "Ca2+", "Mg2+", "Fe", "Cl-", "HCO3-", "SO42-", "CO32-")
        symbols = st.multiselect("Choose stocks to visualize", all_symbols)
        st.header("You selected: {}".format(", ".join(symbols)))
        dt = data_w[symbols]
        lent = len(symbols)
        fig1 = plt.figure(figsize=(15, 5))
        for i in range(0, lent):
            plt.scatter(data_w['Sampledate'], dt.iloc[:, i], label=symbols[i])
            plt.legend()
            plt.xlabel('Year', fontsize=30)
            plt.ylabel('Concentration (mg/l)', fontsize=30)
        st.pyplot(fig1)
        # for i in range(0, lent):
        #     fig1 = alt.Chart(data_w).mark_line().encode(
        #         x=data_w['Sampledate'], y=dt.iloc[:, i], label=symbols[i])
        # st.altair_chart(fig1, use_container_width=True)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application")

if page == "SI Calculation":
    with st.sidebar:
        st.write("# SI Calculation")
    try:
        df = load_data()
        st.write(df)
        si_name = st.sidebar.selectbox("Select Scale Index:",
                                       ("LSI", "RSI", "PSI"))
        if st.sidebar.button('Calculation'):

            df['A'] = (np.log10(df['Tds']) - 1) / 10
            df['B'] = -13.12 * np.log10((df['WBT'] + 273)) + 34.55
            df['C'] = np.log10(df['Ca2+']) - 0.4
            df['D'] = np.log10(0.82 * df['HCO3-'] + 1.667 * df['CO32-'])
            df['pHs'] = (9.3 + df['A'] + df['B']) - (df['C'] + df['D'])
            df['LSI'] = df['pH'] - df['pHs']

            df['RSI'] = 2 * df['pHs'] - df['pH']

            df['PhEQ'] = 1.465 * np.log10(0.82 * df['HCO3-'] + 1.667 * df['CO32-']) + 4.54
            df['PSI'] = 2 * df['pH']-df['pHs']
            st.write('# Plot Scaling Index vs Time')
            plot_si(si_name)
            st.write('# Data table updated')
            st.write(df)
        else:
            st.write('# Please press "Calculation" button')
    except Exception as e:
        print(e)
        st.write('# Please choice SI Methods')

if page == "Machine Learning":
    with st.sidebar:
        st.write("# Regression Methods")
    try:
        df = load_data()
        st.write(df)
        classifier_name = st.sidebar.selectbox("Select Regression Method:",
                                               ("KNN", "SVM", "Random forest", "DecisionTree", "Linear"))
        if st.sidebar.button('Regression'):
            st.write("# Plot PCA")
            well_names = df['Well'].unique()
            df_pca = df.iloc[:, 10:23]
            df_pca = df_pca.dropna()
            scaler = StandardScaler()
            scaler.fit(df_pca)
            w_data = scaler.transform(df_pca)
            my_pca = PCA(n_components=2)
            my_pca.fit(w_data)
            w_df = my_pca.transform(w_data)
            # Thành phần comp số 1
            pca_1 = w_df[:, 0]
            # Thành phần comp số 2
            pca_2 = w_df[:, 1]
            # Vẽ đồ thị
            fig2 = plt.figure(figsize=(15, 5))
            plt.scatter(pca_1, pca_2, c='Red', alpha=0.4,
                        cmap="viridis")
            plt.xlabel('Pc1', fontsize=30)
            plt.ylabel('Pc2', fontsize=30)
            st.pyplot(fig2)
            st.write("# Plot Regression Train")
            predictors = ['pH', 'Na+', 'K+', 'Ca2+', 'Mg2+', 'Fe', 'Cl-', 'HCO3-', 'SO42-', 'CO32-']
            X = df[predictors]
            y = df['LSI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            reg = get_classifier(classifier_name)
            pipe = genericml(reg)

            scale_test = pipe.predict(X_test)
            scale_train = pipe.predict(X_train)

            fig3 = plt.figure(figsize=(15, 5))
            plt.scatter(scale_train, y_train)
            plt.xlabel('LSI_train')
            plt.ylabel('LSI_data')
            st.pyplot(fig3)
            st.write("# Plot Regression Test")
            plotter(reg, "LSI_2")

            core_train = pipe.score(X_train, y_train)
            core_test = pipe.score(X_test, y_test)
            st.write("Training accuracy: ")
            st.write(core_train)
            st.write("Test accuracy: ")
            st.write(core_test)
        else:
            st.write('# Please press "Regression" button')
    except Exception as e:
        print(e)
        st.write('# Please press "Regression" button')

if page == "About":
    st.write("This app is built by VuLe. Please contact Vu via email: lmvu103@gmail.com")
