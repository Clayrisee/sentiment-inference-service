from src.dynamo_db import UncertaintyDynamoDB
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def filter_data(df, start_date: str, end_date: str, prediction: str) -> pd.DataFrame:
    prediction = prediction.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp']) # mengubah format timestamp yang tadinya string menjadi datetime objek
    df['confidence'] = pd.to_numeric(df['confidence']) # merubah format confidence yang tadinya string menjadi numerical
    df['uncertainty_score'] = pd.to_numeric(df['uncertainty_score'])
    filtered = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))] #melakukan filter datetime
    if prediction != 'all': #jika jenis prediction tidak all maka akan melakukan filter tambahan dia masuk prediksi yang mana
        filtered = filtered[filtered['prediction'] == prediction]
    return filtered

def plot_score(df: pd.DataFrame,
               time_frame: str,
               score_column: str = 'confidence',
               plot_title: str = 'Mean Confidence Score',
               threshold=0.6
            ):
    proc_df = df.copy()
    #jika data kosong maka akan ada tulisan no data to plot
    if proc_df.empty:
        st.write("No data to plot.")
        return #supaya function berhenti disitu atau melakukan return None
    
    proc_df.set_index('timestamp', inplace=True)
    if time_frame == 'Hourly':
        proc_df_resample = proc_df[score_column].resample('H').mean()
    elif time_frame == 'Daily':
        proc_df_resample = proc_df[score_column].resample('D').mean()
    elif time_frame == 'Monthly':
        proc_df_resample = proc_df[score_column].resample('M').mean()
    elif time_frame == 'Yearly':
        proc_df_resample = proc_df[score_column].resample('A').mean()

    plt.figure(figsize=(10, 4)) #menyesuaikan shape dari plot figure
    plt.plot(proc_df_resample.index, proc_df_resample, marker='o', linestyle='-') #melakukan plot pada data dari hasil resample
    plt.title(f'{plot_title}: {time_frame}') # memberikan title
    plt.ylabel('Mean Confidence') # set y label
    plt.xlabel('Time') # set x label
    plt.grid(True) #menambahkan grid supaya mudah dilihat
    plt.xticks(rotation=45) #melakukan rotasi 45 derajat
    plt.ylim(0, 1)  # Set the limits of the y-axis to be between 0 and 1 
    plt.axhline(y=threshold, color='r', linestyle='--')  # Add a red horizontal line at the threshold
    st.pyplot(plt)

if __name__ == "__main__":
    default_start_date = datetime.date.today() - datetime.timedelta(days=366)
    uncertainty_db = UncertaintyDynamoDB(table_name='uncertainty')
    # Streamlit UI Components
    st.title("DynamoDB Data Viewer for Uncertainty Table")
    data = uncertainty_db.fetch_data()

    with st.sidebar:
        # set threshold menggunakan number_input
        threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
        # set start date dengan kompnen date_input
        start_date = st.date_input("Start Date", value=default_start_date)
        # set end date dengan komponen date_input
        end_date = st.date_input("End Date")
        # set filter prediction type dengan selectbox
        prediction = st.selectbox("Prediction Type", ["All", "Positive", "Negative"]).lower()
        time_frame = st.selectbox("Aggregate Time Frame", ["Hourly", "Daily", "Monthly", "Yearly"])
    
    if not data.empty:
        data_filtered = filter_data(data, start_date, end_date, prediction)
        plot_score(data_filtered, time_frame, threshold=threshold)
        plot_score(data_filtered, time_frame, score_column='uncertainty_score', threshold=0.6, plot_title="Mean Uncertainty Score")
        st.write(data_filtered)
    else:
        st.write("No data available.")
