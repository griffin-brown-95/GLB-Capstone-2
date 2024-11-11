import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st

def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    # Convert key columns to string to avoid type issues
    df['PRODUCTION_LOCATION'] = df['PRODUCTION_LOCATION'].astype(str)
    df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] = df['FUNCTIONAL_AREA_NODE_2_MODIFIED'].astype(str)
    df['EQUIPMENT_ID'] = df['EQUIPMENT_ID'].astype(str)
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
    
    planned_vs_unplanned_key = agg_df.pivot_table(
        index=['PRODUCTION_LOCATION', 'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'EQUIPMENT_ID'],
        columns='MAINTENANCE_ACTIVITY_TYPE',
        values=['average_minutes', 'count']
    ).reset_index()
    
    planned_vs_unplanned_key.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in planned_vs_unplanned_key.columns]
    planned_vs_unplanned_key['time_saved'] = (
        (planned_vs_unplanned_key['average_minutes_Unplanned'] - planned_vs_unplanned_key['average_minutes_Planned']) / planned_vs_unplanned_key['average_minutes_Unplanned'] * 100
    )

    planned_vs_unplanned_key = planned_vs_unplanned_key[(planned_vs_unplanned_key['count_Planned'] > 50) & (planned_vs_unplanned_key['count_Unplanned'] > 50) & (planned_vs_unplanned_key['time_saved'] > 0)]

    return planned_vs_unplanned_key

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

# Define a function to reset filters
def reset_filters():
    st.session_state['plant'] = "None"
    st.session_state['func_area'] = "None"
    st.session_state['equipment_id'] = "None"

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

# Main Streamlit app
def main():
    st.title("Maintenance Forecast Dashboard")
    st.write("Visualize actual and forecasted maintenance times from a local CSV file.")

    # Specify your local file path here
    file_path = 'IWC_Work_Orders_Extract.csv'

    # Load and preprocess data
    df = load_data(file_path)
    date_columns = ['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO']
    df = preprocess_dates(df, date_columns)
    
    # Aggregate data for planned and unplanned maintenance times
    planned_vs_unplanned_df = aggregate_data(df)

    # Initialize session state for each filter if not already set
    if 'plant' not in st.session_state:
        st.session_state['plant'] = "None"
    if 'func_area' not in st.session_state:
        st.session_state['func_area'] = "None"
    if 'equipment_id' not in st.session_state:
        st.session_state['equipment_id'] = "None"
    if 'run_clicked' not in st.session_state:
        st.session_state['run_clicked'] = False  # Track whether "Run" button was clicked

    # Sidebar selection with reset button
    st.sidebar.header("Filter Selection")

    # Reset button
    if st.sidebar.button("Reset Filters"):
        reset_filters()

    # Plant selection
    plant_options = ["None"] + sorted(df['PRODUCTION_LOCATION'].unique())
    plant = st.sidebar.selectbox("Select Plant", plant_options, key='plant')

    # Conditional display for func_area
    if plant != "None":
        func_area_options = ["None"] + sorted(planned_vs_unplanned_df[planned_vs_unplanned_df['PRODUCTION_LOCATION_'] == plant]['FUNCTIONAL_AREA_NODE_2_MODIFIED_'].unique())
        func_area = st.sidebar.selectbox("Select Functional Area", func_area_options, key='func_area')
    else:
        st.session_state['func_area'] = "None"

    # Conditional display for equipment_id
    if plant != "None" and func_area != "None":
        equipment_options = ["None"] + sorted(planned_vs_unplanned_df[(planned_vs_unplanned_df['PRODUCTION_LOCATION_'] == plant) & 
                                                (planned_vs_unplanned_df['FUNCTIONAL_AREA_NODE_2_MODIFIED_'] == func_area)]['EQUIPMENT_ID_'].unique())
        equipment_id = st.sidebar.selectbox("Select Equipment ID", equipment_options, key='equipment_id')
    else:
        st.session_state['equipment_id'] = "None"
    
    # Validate combination
    valid_combination = (
        plant != "None" and 
        func_area != "None" and 
        equipment_id != "None" and 
        not df[(df['PRODUCTION_LOCATION'] == plant) & 
            (df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == func_area) & 
            (df['EQUIPMENT_ID'] == equipment_id)].empty
    )

    # Display "Run" button only if a valid combination is selected
    if valid_combination:
        if st.button("Run"):
            st.session_state['run_clicked'] = True  # Set flag to indicate the "Run" button was clicked

    # Main processing and visualization logic runs only if "Run" button is clicked
    if st.session_state['run_clicked'] and valid_combination:
        # Filter the selected row for maintenance times
        selected_row = planned_vs_unplanned_df[
            (planned_vs_unplanned_df['PRODUCTION_LOCATION_'] == plant) &
            (planned_vs_unplanned_df['FUNCTIONAL_AREA_NODE_2_MODIFIED_'] == func_area) &
            (planned_vs_unplanned_df['EQUIPMENT_ID_'] == equipment_id)
        ]

        if not selected_row.empty:
            avg_mins_planned = selected_row['average_minutes_Planned'].iloc[0]
            avg_mins_unplanned = selected_row['average_minutes_Unplanned'].iloc[0]

            # Prepare data for Prophet
            selection_aggs = prepare_prophet_data(df, plant, func_area, equipment_id)
            forecast = forecast_prophet(selection_aggs)

            # Merge real and predicted data
            df_bottom_union = forecast[['ds', 'yhat']].copy()
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

            # Combine real and predicted data
            df_union = pd.concat([df_bottom_union, df_top_union_final], ignore_index=True).sort_values(by='ds')

            # Plot results
            st.write("### Maintenance Time: Real vs Forecasted")
            plot_results(df_union)
            st.pyplot(plt)

            # Plot time savings
            st.write("### Time Savings (Unplanned vs. Planned Maintenance)")
            total_unplanned_time_next_year = df_union[df_union['source'] == 'predicted']['ACTUAL_Unplanned'].sum()
            total_planned_time_next_year = df_union[df_union['source'] == 'predicted']['ACTUAL_WORK_IN_MINUTES'].sum()

            minutes_saved = total_unplanned_time_next_year - total_planned_time_next_year
            percent_saved = (minutes_saved / total_unplanned_time_next_year) * 100
            
            # Display time savings as "cards"
            st.write("### Time Savings Summary")
            col1, col2 = st.columns(2)
            col1.metric("Minutes Saved", f"{minutes_saved:,.0f} minutes")
            col2.metric("Percentage Saved", f"{percent_saved:.2f}%")
        else:
            st.warning("No matching records found for the selected options.")

if __name__ == "__main__":
    main()
