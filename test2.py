from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from prophet import Prophet

# Custom data loader for the pipeline
class DataFrameLoader(BaseEstimator, TransformerMixin):
    def __init__(self, file_path):
        self.file_path = file_path

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        # Load the data from CSV and return the DataFrame
        df = pd.read_csv(self.file_path, low_memory=False)
        return df

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
    
class Aggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(['PRODUCTION_LOCATION', 'MAINTENANCE_ACTIVITY_TYPE', 
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
        return planned_vs_unplanned_key
    
class ProphetForecast(BaseEstimator, TransformerMixin):
    def __init__(self, selected_plant, selected_func_area, selected_equipment, periods=48):
        self.selected_plant = selected_plant
        self.selected_func_area = selected_func_area
        self.selected_equipment = selected_equipment
        self.periods = periods
        self.model = Prophet()

    def fit(self, X, y=None):
        df_top_union = X[
            (X['PRODUCTION_LOCATION'] == self.selected_plant) &
            (X['FUNCTIONAL_AREA_NODE_2_MODIFIED'] == self.selected_func_area) &
            (X['EQUIPMENT_ID'] == self.selected_equipment)
        ]
        
        selection_aggs = df_top_union.groupby(df_top_union['START_YEAR_WEEK']).agg(
            y=('EQUIPMENT_ID', 'count')
        ).reset_index().rename(columns={'START_YEAR_WEEK': 'ds'})
        
        self.model.fit(selection_aggs)
        self.avg_mins_planned = selection_aggs['y'].mean()  # Placeholder, replace as needed
        self.avg_mins_unplanned = selection_aggs['y'].mean()  # Placeholder, replace as needed
        return self

# Define the file path for data loading
file_path = 'IWC_Work_Orders_Extract.csv'

# Build the pipeline with the data loader and date transformer
pipeline = Pipeline([
    ('data_loader', DataFrameLoader(file_path=file_path)),
    ('date_transform', DateTransformer(date_columns=['EXECUTION_START_DATE', 'EXECUTION_FINISH_DATE', 'EQUIP_START_UP_DATE', 'EQUIP_VALID_FROM', 'EQUIP_VALID_TO'])),
    ('planned_vs_unplanned_key', Aggregator()),
    ('forecast', ProphetForecast(selected_plant='COTA', selected_func_area='CAN_LINE', selected_equipment=300025792.0, periods=48))
])

# Apply the pipeline and print the top 5 rows
transformed_df = pipeline.fit_transform(None)
print(transformed_df.head())