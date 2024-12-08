# Swire Coca-Cola Work Order Maintenance
## Prophet Model, Streamlit App
---
[
](https://www.google.com/url?sa=i&url=https%3A%2F%2Fcommons.wikimedia.org%2Fwiki%2FFile%3ASwire_Logo.svg&psig=AOvVaw1od-u858hO7TtHqHobpg9B&ust=1733783960652000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCODL_NuemYoDFQAAAAAdAAAAABAJ)![image](https://github.com/user-attachments/assets/f62a24c8-73e2-4991-bafc-74c042833e3a)

Author: Griffin Brown

Date: Fall 2024

## Problem Statement
----
As the primary bottler of Coca-Cola products in the Western United States, Swire Coca-Cola's operational efficiency affects the entire distribution network. Despite achieving 94% mechanical efficiency, unforeseen machine breakdowns and inadequate predictive maintenance lead to significant downtime. This shortfall in meeting demand results in an annual loss of approximately $60 million in uncaptured revenue. The current process—issuing work orders, waiting for parts, and making repairs—often lacks the foresight to prevent these downtimes, leading to extended production stoppages and impacting overall performance.

## Personal Goal
---
As this was a group project, my personal goals and work are reflected in this project. I wanted to be able to forecast jobs for the future and optimize time, since time spent maintaining resulted in expenses. This was realized by forecasting jobs for the future, and if we know that they are in the future, the job could be considered "planned." Planned maintenance by part was consistenly lower than unplanned, so applying a "planned" schedule resulted in overall less minutes.

# Files
## eda.ipynb
---
This file covers some basic EDA, but overall my thought process could have been better. I was a bit biased towards trying to optimize the scheduling of tasks, which biased my efforts into creating things in the "Breaking Down By Plant" section. I think I could have kept this document a little more high level compared to trying to fit EDA directly to what I thought the solution was.

The ultimate main takeaway was planned vs unplanned maintenance was much lower in terms of time, which was a huge clue of how to optimize time in maintenance.
```
MAINTENANCE_ACTIVITY_TYPE
Planned      48.034311
Unplanned    93.410893
Name: ACTUAL_WORK_IN_MINUTES, dtype: float64
```

## prophet_example.ipynb
---
This file covers playing with the prophet model in order to forecast number of jobs in the future. Eventually, this would turn into forecasting time in minutes for our group project because time was ultimately our target variable.

![image](https://github.com/user-attachments/assets/a6c3c3e4-9134-4e1f-a9a6-b19216b1b069)

As you can see, when predicting jobs, the graphs look a little weird, especially when job numbers are low. We found a stronger confidence in predicting time as a group, which was much better suited for time series modeling.

## unplanned_planned_maintenance.ipynb
---
This file takes the prophet a touch further by aggregating average planned or unplanned maintenance times by part, and amortizing that amount of minutes for yhat. Essentially this is saying, when planning in the future, you can expect this many minutes. When you're not planning, or just continuing to do what you do now, you will spend x amount of minutes more.

The one problem with this, and what we adjusted for in our final group project, was that the forecasted minutes include both planned and unplanned jobs. So say, if a part had 80% planned maintence, and 20% unplanned, we can only apply 20% of those jobs to be planned in the future. If it were vice versa, we could make a bigger impact because there are more jobs to be planned.

![image](https://github.com/user-attachments/assets/999a79b1-79be-4f5b-a50b-688d9fcd6a4a)

## streamlit_app.py
---
This app shows what this might look like for stakeholder use. Eventually, this was turned into a PowerBI app, which was built off of models running upstream on pieces of equipment in order to forecast time. The streamlit app generically forecasted out 100 days, so the metrics aren't really digestible. In PowerBI, we operated on a month to month basis.

![Screenshot 2024-12-08 at 3 56 13 PM](https://github.com/user-attachments/assets/c467f7ed-7ca2-47e7-9adb-c61f014ca503)

Final Dashboard (example three equipments):

![Screenshot 2024-12-08 at 3 59 24 PM](https://github.com/user-attachments/assets/451ef0c0-be01-401b-93d2-7cd904883cad)

## lifetimes.ipynb
---
This was just an effort to see how I could predict next maintenance actiivty date by equipment, I didn't get to flush out this approach too much. What I liked about the Prophet Modeling is that it was dynamic in a sense that it would change autmatically due to trends, rather than just simply an average of time between jobs.
