from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d

def estimate_time_at_probability(time_points, probabilities, target_probability):
    """
    Estimates the time at which the survival probability reaches a certain level.
 
    Parameters:
    time_points (array-like): Array of time points.
    probabilities (array-like): Array of survival probabilities corresponding to the time points.
    target_probability (float): The target survival probability.
 
    Returns:
    float: Estimated time at which the survival probability reaches the target.
    """
    interpolator = interp1d(probabilities, time_points, bounds_error=False, fill_value="extrapolate")
    return interpolator(target_probability)

def new_metric_interp(df_test, non_encoded_df, df_result_test):
    DELAY = 0
    ERROR = 15
    within = 0
    mae_above = 0
    above_count = 0
    mae_below = 0
    below_count = 0
    inf_count = 0
    kmf_dict = {}

    for name, grouped_df in non_encoded_df.groupby(['S_Hour_arrival', 'Link']):
        kmf = KaplanMeierFitter()
        kmf.fit(grouped_df['Time'], event_observed=grouped_df['Event'])
        kmf_dict[name] = kmf
            
    for obs in range(len(df_test)): # Until length of test_df ( starts with obs=0) 
        idx = df_test.index[obs] #Get the index for obs
 
        # Current obs info
        query_time = non_encoded_df.loc[idx]['CRSElapsedTime'] + DELAY# Scheduled Arrival Time from all data (using index)
 
        # 1. Get the p from the fitted kmf 
        kmf_idx = kmf_dict[(non_encoded_df.loc[idx, 'S_Hour_arrival'], non_encoded_df.loc[idx, 'Link'])] # Get the fitted object
        p_kmf = kmf_idx.survival_function_at_times(query_time).values[0] # Find the probability at query time
 
        # 2. Find the corresponding time for the p in the predicted estimate_time_at_probability(p)
        predicted_times = df_result_test.columns
        predicted_p = df_result_test.iloc[obs]  # 'obs' is the specific column (in predicted surv df)
        #print(predicted_times, predicted_p)
        pred_time = estimate_time_at_probability(predicted_times, predicted_p, target_probability=p_kmf) # Gets time for p_kmf
 
        # 3. Calculate and update metrics for new estimated time
 
        # Calculate metrics
        if pred_time == float('-inf'):
            pred_time = 0
        if pred_time == float('inf'):
            inf_count += 1
        else:
            if query_time-ERROR <= pred_time <= query_time+ERROR:
                within += 1
            elif pred_time > query_time+ERROR:
                mae_above += pred_time-query_time
                above_count += 1
            else:
                mae_below += query_time-pred_time
                below_count += 1
 
 
    # Compile the results
    results = {'within': within, 
            'within_pct': within/(len(df_test)-inf_count),
            'mae': (mae_above+mae_below)/(above_count + below_count),
            'mae_above': mae_above/above_count, 
            'mae_below': mae_below/below_count, 
            'above_count': above_count, 
            'below_count': below_count,
            'inf_count': inf_count}
    return results 