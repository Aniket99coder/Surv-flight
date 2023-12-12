from lifelines import KaplanMeierFitter

def new_metric(test_df, non_encoded_df, cph, error=15):
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
        actual_time = row['CRSElapsedTime'] + DELAY
        p = kmf_dict[(row['S_Hour_arrival'], row['Link'])].survival_function_at_times(actual_time)
        pred_time = cph.predict_percentile(test_df.loc[[index]], p)
        if pred_time == float('inf'):
            inf_count += 1
            bad.append(row.name)
        else:
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