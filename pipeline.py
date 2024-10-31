from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd

def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def clean_dates(df):
    date_columns = ['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def aggregate_base_df(df):
    agg_df = df.groupby(['PRODUCTION_LOCATION', 'MAINTENANCE_ACTIVITY_TYPE', 'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'EQUIPMENT_ID']).agg(
        average_minutes=('ACTUAL_WORK_IN_MINUTES', 'mean'),
        count=('ACTUAL_WORK_IN_MINUTES', 'count')
    ).reset_index()

    pivoted_df = agg_df.pivot_table(
        index=['PRODUCTION_LOCATION', 'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'EQUIPMENT_ID'], 
        columns='MAINTENANCE_ACTIVITY_TYPE', 
        values=['average_minutes', 'count']
    ).reset_index()

    pivoted_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pivoted_df.columns]
    pivoted_df['time_saved'] = (pivoted_df['average_minutes_Unplanned'] - pivoted_df['average_minutes_Planned']) / pivoted_df['average_minutes_Unplanned'] * 100

    pivoted_df = pivoted_df.rename(columns=lambda x: x.rstrip('_'))

    final_agg_df = pivoted_df[
        (pivoted_df['average_minutes_Planned'].notna()) &
        (pivoted_df['average_minutes_Unplanned'].notna()) &
        ((pivoted_df['count_Planned'] > 100.0) | (pivoted_df['count_Unplanned'] > 100.0)) &
        (pivoted_df['time_saved'] > 0)
    ]

    return final_agg_df.sort_values(by='count_Unplanned', ascending=False)

def get_selections():
    selected_plant = 'COTA'
    selected_func_area = 'CAN LINE'
    selected_equipment = 300025792.0
    return selected_plant, selected_func_area, selected_equipment

def base_union(df):
    selected_plant, selected_func_area, selected_equipment = get_selections()
    selected_columns = ['PRODUCTION_LOCATION', 'FUNCTIONAL_AREA_NODE_2_MODIFIED', 'MAINTENANCE_ACTIVITY_TYPE', 'EQUIPMENT_ID', 'EXECUTION_START_DATE', 'ACTUAL_WORK_IN_MINUTES']
    df_top_union = df[selected_columns]
    df_top_union_final = df_top_union[
        (df_top_union['PRODUCTION_LOCATION'] == selected_plant) &
        (df_top_union['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (df_top_union['EQUIPMENT_ID'] == selected_equipment)
    ]
    df_top_union_final['source'] = 'real'
    df_top_union_final = df_top_union_final.rename(columns={'EXECUTION_START_DATE': 'ds'})
    return df_top_union_final

def make_initial_prophet_dataframe(df):
    df_top_union = base_union(df)
    selection_aggs = df_top_union.groupby('ds').agg(y=('ACTUAL_WORK_IN_MINUTES', 'sum')).reset_index()
    return selection_aggs

def get_min_max_dates(df):
    selection_aggs = make_initial_prophet_dataframe(df)
    min_date = selection_aggs['ds'].min()
    max_date = selection_aggs['ds'].max()
    return min_date, max_date

def create_calendar(df):
    min_date, max_date = get_min_max_dates(df)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    full_cal = pd.DataFrame(date_range, columns=['ds'])
    return full_cal

def join_calendar_prophet(df):
    full_cal = create_calendar(df)
    selection_aggs = make_initial_prophet_dataframe(df)
    full_date_aggs = pd.merge(full_cal, selection_aggs, on='ds', how='left')
    full_date_aggs['ds'] = pd.to_datetime(full_date_aggs['ds'])
    full_date_aggs = full_date_aggs.fillna(0)
    return full_date_aggs

def train_model(df):
    calendar_prophet = join_calendar_prophet(df)
    model = Prophet()
    model.fit(calendar_prophet)
    return model

def predict_future(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def get_average_minutes(df):
    final_agg_df = aggregate_base_df(df)
    selected_plant, selected_func_area, selected_equipment = get_selections()

    avg_mins_planned = final_agg_df[
        (final_agg_df['PRODUCTION_LOCATION'] == selected_plant) & 
        (final_agg_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (final_agg_df['EQUIPMENT_ID'] == selected_equipment)
    ]['average_minutes_Planned'].iloc[0]

    avg_mins_unplanned = final_agg_df[
        (final_agg_df['PRODUCTION_LOCATION'] == selected_plant) & 
        (final_agg_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (final_agg_df['EQUIPMENT_ID'] == selected_equipment)
    ]['average_minutes_Unplanned'].iloc[0]

    return avg_mins_planned, avg_mins_unplanned

def create_top_union(df):
    model = train_model(df)
    forecast = predict_future(model)
    selected_plant, selected_func_area, selected_equipment = get_selections()
    avg_mins_planned, avg_mins_unplanned = get_average_minutes(df)
    _, max_date = get_min_max_dates(df)

    df_bottom_union = forecast[['ds', 'yhat']]
    df_bottom_union['PRODUCTION_LOCATION'] = selected_plant
    df_bottom_union['FUNCTIONAL_AREA_NODE_2_MODIFIED'] = selected_func_area
    df_bottom_union['MAINTENANCE_ACTIVITY_TYPE'] = 'Planned'
    df_bottom_union['EQUIPMENT_ID'] = selected_equipment
    df_bottom_union['ACTUAL_WORK_IN_MINUTES'] = df_bottom_union['yhat'] * avg_mins_planned
    df_bottom_union['ACTUAL_Unplanned'] = df_bottom_union['yhat'] * avg_mins_unplanned
    df_bottom_union['source'] = 'predicted'

    df_bottom_union = df_bottom_union[df_bottom_union['ds'] > max_date].drop('yhat', axis=1)
    return df_bottom_union

def union(df):
    df_top_union_final = base_union(df)
    df_bottom_union = create_top_union(df)
    df_union = pd.concat([df_top_union_final, df_bottom_union], ignore_index=True).sort_values(by='ds')
    return df_union

def make_plot(df):
    df_union = union(df)

    plt.figure(figsize=(12, 6))

    plt.plot(df_union[df_union['source'] == 'real']['ds'], df_union[df_union['source'] == 'real']['ACTUAL_WORK_IN_MINUTES'], 
             label='Actual (Real)', color='black', linestyle='--')

    plt.plot(df_union[df_union['source'] == 'predicted']['ds'], df_union[df_union['source'] == 'predicted']['ACTUAL_WORK_IN_MINUTES'], 
             label='Predicted Planned Maintenance', color='blue')

    plt.plot(df_union[df_union['source'] == 'predicted']['ds'], df_union[df_union['source'] == 'predicted']['ACTUAL_Unplanned'], 
             label='Predicted Unplanned Maintenance', color='red')

    plt.xlabel('Date')
    plt.ylabel('Actual Work in Minutes')
    plt.title('Real vs Predicted Planned and Unplanned Maintenance')
    plt.legend(title="Maintenance Type")
    plt.grid(True)
    plt.show()
