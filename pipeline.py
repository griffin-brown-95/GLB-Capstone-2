from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from prophet import Prophet
import pandas as pd


df = pd.read_csv('IWC_Work_Orders_Extract.csv')

# Ensure date columns are in datetime format
date_columns = ['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Step 1: Define Custom Transformers for Filtering and Aggregation
class FilterData(BaseEstimator, TransformerMixin):
    def __init__(self, selected_plant, selected_func_area, selected_equipment):
        self.selected_plant = selected_plant
        self.selected_func_area = selected_func_area
        self.selected_equipment = selected_equipment
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[
            (X['PRODUCTION_LOCATION'] == self.selected_plant) &
            (X['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == self.selected_func_area) &
            (X['EQUIPMENT_ID'] == self.selected_equipment)
        ]

class AggregateData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Calculate average planned/unplanned times
        agg_df = X.groupby(['MAINTENANCE_ACTIVITY_TYPE']).agg(
            avg_minutes=('ACTUAL_WORK_IN_MINUTES', 'mean')
        ).reset_index()
        
        # Extract planned and unplanned averages
        avg_planned = agg_df.loc[agg_df['MAINTENANCE_ACTIVITY_TYPE'] == 'Planned', 'avg_minutes'].values[0]
        avg_unplanned = agg_df.loc[agg_df['MAINTENANCE_ACTIVITY_TYPE'] == 'Unplanned', 'avg_minutes'].values[0]
        
        # Prepare the result DataFrame
        return pd.DataFrame({
            'avg_planned': [avg_planned],
            'avg_unplanned': [avg_unplanned]
        })

class ProphetForecast(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Fit Prophet model on the count of past occurrences
        self.model = Prophet()
        self.model.fit(X[['ds', 'y']])  # Expecting pre-processed X with date ('ds') and count ('y') columns
        return self
    
    def transform(self, X):
        # Create future DataFrame and predict
        future = self.model.make_future_dataframe(periods=365)
        forecast = self.model.predict(future)
        
        # Return forecasted counts for next year
        return forecast[['ds', 'yhat']]

# Step 2: Assemble the Pipeline
def run_pipeline(selected_plant, selected_func_area, selected_equipment, df):
    pipeline = Pipeline([
        ('filter_data', FilterData(selected_plant, selected_func_area, selected_equipment)),
        ('aggregate_data', AggregateData()),
        ('prophet_forecast', ProphetForecast())
    ])

    # Prepare the data with counts for model training
    filtered_data = df[
        (df['PRODUCTION_LOCATION'] == selected_plant) &
        (df['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == selected_func_area) &
        (df['EQUIPMENT_ID'] == selected_equipment)
    ]
    selection_aggs = filtered_data.groupby(filtered_data['EXECUTION_START_DATE']).agg(
        y=('EQUIPMENT_ID', 'count')
    ).reset_index().rename(columns={'EXECUTION_START_DATE': 'ds'})

    # Run the pipeline
    forecast_results = pipeline.fit_transform(selection_aggs)
    
    # Calculate final planned vs unplanned times based on forecast
    avg_mins = pipeline.named_steps['aggregate_data'].transform(filtered_data)
    forecast_results['Planned_Maintenance'] = forecast_results['yhat'] * avg_mins['avg_planned'].values[0]
    forecast_results['Unplanned_Maintenance'] = forecast_results['yhat'] * avg_mins['avg_unplanned'].values[0]

    return forecast_results

# Example usage
selected_plant = 'COTA'
selected_func_area = 'CAN LINE'
selected_equipment = 300025792.0
forecast_df = run_pipeline(selected_plant, selected_func_area, selected_equipment, df)
print(forecast_df.head())
