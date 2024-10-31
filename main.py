import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from pipeline import read_data, clean_dates, aggregate_base_df, train_model, predict_future, union, make_plot

class MaintenancePipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.model = None
        self.forecast = None

    def run(self):
        self.df = read_data(self.file_path)                 # Step 1: Read data
        self.df = clean_dates(self.df)                      # Step 2: Clean dates
        self.df = aggregate_base_df(self.df)                # Step 3: Aggregate base dataframe
        self.model = train_model(self.df)                   # Step 4: Train Prophet model
        self.forecast = predict_future(self.model)          # Step 5: Predict future
        self.plot_results()                                 # Step 6: Plot results

    def plot_results(self):
        df_union = union(self.df)                           # Step 7: Union real and predicted data
        make_plot(df_union)                                 # Plot the result

# Initialize the pipeline with the file path
pipeline = MaintenancePipeline('IWC_Work_Orders_Extract.csv')

# Run the pipeline
pipeline.run()
