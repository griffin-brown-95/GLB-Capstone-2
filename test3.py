from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from prophet import Prophet
import pandas as pd

# Data loader and date transformer (pipeline)
class DataFrameLoader(BaseEstimator, TransformerMixin):
    def __init__(self, file_path):
        self.file_path = file_path

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return pd.read_csv(self.file_path, low_memory=False)

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
        X['START_YEAR_WEEK'] = X['EXECUTION_START_DATE'].dt.to_period('W').dt.to_timestamp()
        return X

# Aggregator function (outside pipeline)
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
    return pivoted_df

# Prophet forecasting class (for use after aggregation)
class ProphetForecast(BaseEstimator, TransformerMixin):
    def __init__(self, selected_plant, selected_func_area, selected_equipment, avg_mins_planned, avg_mins_unplanned, periods=48):
        self.selected_plant = selected_plant
        self.selected_func_area = selected_func_area
        self.selected_equipment = selected_equipment
        self.avg_mins_planned = avg_mins_planned
        self.avg_mins_unplanned = avg_mins_unplanned
        self.periods = periods
        self.model = Prophet()

    def fit(self, X, y=None):
        # Filter data for selected equipment
        df_top_union = X[
            (X['PRODUCTION_LOCATION'] == self.selected_plant) &
            (X['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == self.selected_func_area) &
            (X['EQUIPMENT_ID'] == self.selected_equipment)
        ]
        
        selection_aggs = df_top_union.groupby(df_top_union['START_YEAR_WEEK']).agg(
            y=('EQUIPMENT_ID', 'count')
        ).reset_index().rename(columns={'START_YEAR_WEEK': 'ds'})
        
        self.model.fit(selection_aggs)
        return self

    def transform(self, X):
        future = self.model.make_future_dataframe(periods=self.periods, freq='W')
        forecast = self.model.predict(future)
        forecast['ACTUAL_WORK_IN_MINUTES'] = forecast['yhat'] * self.avg_mins_planned
        forecast['ACTUAL_Unplanned'] = forecast['yhat'] * self.avg_mins_unplanned
        forecast['source'] = 'predicted'
        return forecast[['ds', 'ACTUAL_WORK_IN_MINUTES', 'ACTUAL_Unplanned', 'source']]

# Define file path and columns
file_path = 'IWC_Work_Orders_Extract.csv'
date_columns = ['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO']

# Pipeline for data loading and date transformation
pipeline = Pipeline([
    ('data_loader', DataFrameLoader(file_path=file_path)),
    ('date_transform', DateTransformer(date_columns=date_columns))
])

# Apply pipeline to load and preprocess data
df = pipeline.fit_transform(None)

# Use the aggregator outside of the pipeline
aggregated_df = aggregate_data(df)

# Select average planned/unplanned maintenance times for the specific equipment
selected_plant = 'COTA'
selected_func_area = 'CANLINE'
selected_equipment = 300025792.0

avg_mins_planned = aggregated_df[
    (aggregated_df['PRODUCTION_LOCATION'] == selected_plant) &
    (aggregated_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
    (aggregated_df['EQUIPMENT_ID'] == selected_equipment)
]['average_minutes_Planned'].iloc[0]

avg_mins_unplanned = aggregated_df[
    (aggregated_df['PRODUCTION_LOCATION'] == selected_plant) &
    (aggregated_df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
    (aggregated_df['EQUIPMENT_ID'] == selected_equipment)
]['average_minutes_Unplanned'].iloc[0]

# Initialize Prophet forecast with pre-calculated averages
prophet_forecaster = ProphetForecast(
    selected_plant=selected_plant,
    selected_func_area=selected_func_area,
    selected_equipment=selected_equipment,
    avg_mins_planned=avg_mins_planned,
    avg_mins_unplanned=avg_mins_unplanned
)

# Fit and transform Prophet forecast
prophet_forecaster.fit(df)
forecast_df = prophet_forecaster.transform(df)

# Display the top of the forecasted DataFrame
print(forecast_df.head())
