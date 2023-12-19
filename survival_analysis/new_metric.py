from lifelines import KaplanMeierFitter

def new_metric(test_df, non_encoded_df, cph, error=15):
    """
    Calculates the percentage of predictions that lands within an error interval. For each
    flight observation it uses the event time with the KM curve to obtain a probability that is 
    then is matched with the corresponding time from the curve obtained by the Cox regression.
 
    Parameters:
    test_df (pd.DataFrame): The test data
    non_encoded_df (pd.DataFrame): The complete data set preprocessing
    cph (CoxPHFitter): The cph fitter object from lifelines
    error (int): The error margin
 
    Returns:
    tuple: a dictionary with metrics and list 
        Dictionary: Calculated within percent, MAE values, infinity points
        List[int]: Indices where the metric returns infinity

    """
    DELAY = 0
    within = 0
    mae_above = 0
    above_count = 0
    mae_below = 0
    below_count = 0
    inf_count = 0
    df_no_dummy = non_encoded_df.loc[test_df.index]
    kmf_dict = {}
    bad = []
    
    #Preprocessing for optimization
    for name, grouped_df in non_encoded_df.groupby(['S_Hour_arrival', 'Link']):
        kmf = KaplanMeierFitter()
        kmf.fit(grouped_df['Time'], event_observed=grouped_df['Event'])
        kmf_dict[name] = kmf

    for index,row in df_no_dummy.iterrows():

        # Get the p from the fitted kmf 
        actual_time = row['CRSElapsedTime'] + DELAY
        p = kmf_dict[(row['S_Hour_arrival'], row['Link'])].survival_function_at_times(actual_time)

        # Find the corresponding time for the p in the predicted estimate_time_at_probability(p)
        pred_time = cph.predict_percentile(test_df.loc[[index]], p)

        # Handles when pred_time returns infinity. Occurs when p=0, in these cases the SAT is overestimated
        # since all historic flights arrived earlier.
        if pred_time == float('inf'):
            inf_count += 1
            bad.append(row.name)
        else:
            # Calculate metrics
            if actual_time-error <= pred_time <= actual_time+error:
                within += 1  
            elif pred_time > actual_time+error:
                mae_above += pred_time-actual_time
                above_count += 1
            else:
                mae_below += actual_time-pred_time
                below_count += 1
        
    return ({'within': within, 
            'within_pct': within/(len(test_df)-inf_count),
            'mae': (mae_above+mae_below)/within,
            'mae_above': mae_above/above_count, 
            'mae_below': mae_below/below_count, 
            'above_count': above_count, 
            'below_count': below_count,
            'inf_count': inf_count}, bad)