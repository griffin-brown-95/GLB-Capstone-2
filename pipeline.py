import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df

# Convert columns to datetime and add START_YEAR_WEEK
def preprocess_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['START_YEAR_WEEK'] = df['EXECUTION_START_DATE'].dt.to_period('W').dt.to_timestamp()
    return df

# Aggregate the data
def aggregate_data(df):
    agg_df = df.groupby(['PRODUCTION_LOCATION', 'MAINTENANCE_ACTIVITY_TYPE', 
                         'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'EQUIPMENT_ID']).agg(
        average_minutes=('ACTUAL_WORK_IN_MINUTES', 'mean'),
        count=('ACTUAL_WORK_IN_MINUTES', 'count')
    ).reset_index()
    
    pivoted_df = agg_df.pivot_table(
        index=['PRODUCTION_LOCATION', 'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'EQUIPMENT_ID'],
        columns='MAINTENANCE_ACTIVITY_TYPE',
        values=['average_minutes', 'count']
    ).reset_index()
    
    pivoted_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pivoted_df.columns]
    pivoted_df['time_saved'] = (
        (pivoted_df['average_minutes_Unplanned'] - pivoted_df['average_minutes_Planned']) / pivoted_df['average_minutes_Unplanned'] * 100
    )
    return pivoted_df

# Prepare data for Prophet model
def prepare_prophet_data(df, plant, func_area, equipment_id):
    df_top_union = df[
        (df['PRODUCTION_LOCATION'] == plant) &
        (df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == func_area) &
        (df['EQUIPMENT_ID'] == equipment_id)
    ]
    
    selection_aggs = df_top_union.groupby('START_YEAR_WEEK').agg(
        y=('EQUIPMENT_ID', 'count')
    ).reset_index().rename(columns={'START_YEAR_WEEK': 'ds'})
    
    return selection_aggs

# Fit Prophet model and forecast
def forecast_prophet(data, periods=48):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast

# Merge real and predicted data
def merge_forecasts(df, forecast, plant, func_area, equipment_id, avg_mins_planned, avg_mins_unplanned):
    df_bottom_union = forecast[['ds', 'yhat']]
    df_bottom_union['PRODUCTION_LOCATION'] = plant
    df_bottom_union['FUNCTIONAL_AREA_NODE_2_MODIFIED'] = func_area
    df_bottom_union['MAINTENANCE_ACTIVITY_TYPE'] = 'Planned'
    df_bottom_union['EQUIPMENT_ID'] = equipment_id
    df_bottom_union['ACTUAL_WORK_IN_MINUTES'] = df_bottom_union['yhat'] * avg_mins_planned
    df_bottom_union['ACTUAL_Unplanned'] = df_bottom_union['yhat'] * avg_mins_unplanned
    df_bottom_union['source'] = 'predicted'
    
    max_date = df['START_YEAR_WEEK'].max()
    df_bottom_union = df_bottom_union[df_bottom_union['ds'] > max_date].drop('yhat', axis=1)

    df_top_union_final = df[
        (df['PRODUCTION_LOCATION'] == plant) &
        (df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == func_area) &
        (df['EQUIPMENT_ID'] == equipment_id)
    ].copy()
    df_top_union_final['source'] = 'real'
    df_top_union_final = df_top_union_final.rename(columns={'START_YEAR_WEEK': 'ds'})

    df_union = pd.concat([df_bottom_union, df_top_union_final], ignore_index=True).sort_values(by='ds')
    return df_union

# Plot the results
def plot_results(df_union):
    plt.figure(figsize=(12, 6))
    plt.plot(df_union[df_union['source'] == 'real']['ds'], df_union[df_union['source'] == 'real']['ACTUAL_WORK_IN_MINUTES'], label='Actual (Real)', color='black')
    plt.plot(df_union[df_union['source'] == 'predicted']['ds'], df_union[df_union['source'] == 'predicted']['ACTUAL_WORK_IN_MINUTES'], label='Predicted Planned Maintenance', color='blue')
    plt.plot(df_union[df_union['source'] == 'predicted']['ds'], df_union[df_union['source'] == 'predicted']['ACTUAL_Unplanned'], label='Predicted Unplanned Maintenance', color='red')
    plt.xlabel('Date')
    plt.ylabel('Actual Work in Minutes')
    plt.title('Real vs Predicted Planned and Unplanned Maintenance')
    plt.legend(title="Maintenance Type")
    plt.grid(True)
    plt.show()

# Main function to run the entire process
def main():
    file_path = 'IWC_Work_Orders_Extract.csv'
    date_columns = ['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO']
    selected_plant = 'COTA'
    selected_func_area = 'CANLINE'
    selected_equipment = 300025792.0

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_dates(df, date_columns)

    # Aggregate data
    pivoted_df = aggregate_data(df)

    # Calculate average minutes for planned and unplanned maintenance
    avg_mins_planned = pivoted_df[
        (pivoted_df['PRODUCTION_LOCATION'] == selected_plant) &
        (pivoted_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (pivoted_df['EQUIPMENT_ID'] == selected_equipment)
    ]['average_minutes_Planned'].iloc[0]

    avg_mins_unplanned = pivoted_df[
        (pivoted_df['PRODUCTION_LOCATION'] == selected_plant) &
        (pivoted_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (pivoted_df['EQUIPMENT_ID'] == selected_equipment)
    ]['average_minutes_Unplanned'].iloc[0]

    # Prepare data for Prophet model and forecast
    selection_aggs = prepare_prophet_data(df, selected_plant, selected_func_area, selected_equipment)
    forecast = forecast_prophet(selection_aggs)

    # Merge real and predicted data
    df_union = merge_forecasts(df, forecast, selected_plant, selected_func_area, selected_equipment, avg_mins_planned, avg_mins_unplanned)

    # Plot the results
    plot_results(df_union)

# Run the main function
if __name__ == "__main__":
    main()
