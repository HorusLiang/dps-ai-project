# Import Libraries
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, request, jsonify
app = Flask(__name__)


def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to convert German feature names to English and filter data before 2021
def preprocess_data(df,latest_month=202101):
    column_name_mapping = {
        "MONATSZAHL": "Category",
        "AUSPRAEGUNG": "Accident-type",
        "JAHR": "Year",
        "MONAT": "Month",
        "WERT": "Value",
        "VORJAHRESWERT": "Previous_Year_Value",
        "VERAEND_VORMONAT_PROZENT": "Change_From_Previous_Month_Percentage",
        "VERAEND_VORJAHRESMONAT_PROZENT": "Change_From_Previous_Year_Month_Percentage",
        "ZWOELF_MONATE_MITTELWERT": "Twelve_Month_Average"
    }
    
    df = df.rename(columns=column_name_mapping)
    df = df[df['Month'] <= latest_month]
    
    columns_to_exclude = ['Previous_Year_Value', 'Change_From_Previous_Month_Percentage', 
                          'Change_From_Previous_Year_Month_Percentage', 'Twelve_Month_Average']
    
    df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])
    
    return df

# Function to extract time-related features
def extract_time_related_features(df, month_column='Month'):
    df['Quarter'] = pd.to_datetime(df[month_column], format='%Y%m').dt.quarter
    df['Month_Start_Weekday'] = pd.to_datetime(df[month_column], format='%Y%m').dt.dayofweek + 1
    df['Month_End_Weekday'] = pd.to_datetime(df[month_column], format='%Y%m').apply(
        lambda x: pd.Timestamp(x.year, x.month, x.daysinmonth).dayofweek + 1
    )
    
    return df

def count_groups(df):
    unique_combinations = df.groupby(['Category', 'Accident-type']).size().reset_index().rename(columns={0: 'Count'})
    filters_list = unique_combinations.apply(lambda row: {'Category': row['Category'], 'Accident-type': row['Accident-type']}, axis=1)
    return filters_list



def apply_time_window_features(df, filters):
    def extract_time_window_features(df, month_column, value_column, filters, rolling_functions, rolling_windows, year_windows):
        if filters:
            for key, value in filters.items():
                assert key in df.columns, f"Filter key {key} not in DataFrame."
                df = df[df[key] == value]

        df = df.sort_values(by=month_column).reset_index(drop=True)

        for func_name in rolling_functions:
            for window in rolling_windows:
                col_name = f'{func_name}_{window}m'
                df[col_name] = df[value_column].shift(1).rolling(window=window, min_periods=window).agg(func_name)

        for year_window in year_windows:
            shift_periods = year_window * 12
            lag_col_name = f'value_{year_window}_years_ago'
            df[lag_col_name] = df[value_column].shift(shift_periods)
        return df
    # Predefined rolling functions, windows, and year windows
    rolling_functions = ['mean']
    rolling_windows = [2, 3, 6, 9]  # note if window size is 1, then std is not meaningful
    year_windows = [1]  # note if window size is 1, then std is not meaningful

    # Extracting time window features
    return extract_time_window_features(df, 
                                        'Month', 
                                        'Value', 
                                        filters, 
                                        rolling_functions, 
                                        rolling_windows, 
                                        year_windows)
def predict_traffic_accidents(input_data):
    model_path = "./models/model_{'Category': 'Alkoholunfälle', 'Accident-type': 'insgesamt'}.joblib"
    year = input_data["year"]
    month = input_data["month"]
    formatted_month = f"{year}{str(month).zfill(2)}"  

    file_path = 'monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv'
    columns_to_remove = ['Category', 'Accident-type', 'Month']
    df = read_data(file_path)
    df = preprocess_data(df, latest_month=202101)
    df = extract_time_related_features(df)

    filtered_data = None
    for gp in count_groups(df):
        df_group = apply_time_window_features(df, {'Category': gp['Category'], 'Accident-type': gp['Accident-type']})
        filtered_data = df_group[(df_group['Category'] == 'Alkoholunfälle') & 
                                 (df_group['Accident-type'] == 'insgesamt') & 
                                 (df_group['Month'] == 202101)]
        if filtered_data.shape!= (0, 13):
            break
    
    if filtered_data is not None and not filtered_data.empty:
        filtered_data.drop(columns_to_remove, axis=1, inplace=True)
        filtered_data = filtered_data.drop('Value', axis=1)
        model = load(model_path)
        X = filtered_data.to_numpy()
        X = np.array([X]) if X.ndim == 1 else X

        predictions = model.predict(X)
        return predictions[0]
    else:
        return None


@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    predictions = predict_traffic_accidents(data)
    return jsonify(predictions)

@app.route('/', methods=['GET'])
def test():
    return "Hello World!"


if __name__ == '__main__':
    app.run(debug=False)