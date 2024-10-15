import pandas as pd

def get_mean_performance(results,n_splits):
    mean_performance = {}
    for model in results[0].keys():
        mean_performance[model] = {'Accuracy': 0, 'Macro-Precision': 0, 'Macro-Recall': 0, 'Macro-F-Score': 0}
    for result in results:
        for model in mean_performance.keys():
            for metric in mean_performance[model].keys():
                value = result[model][metric]
                if value is not None:
                    mean_performance[model][metric] += value

    for model in mean_performance.keys():
        for metric in mean_performance[model].keys():
            mean_performance[model][metric] /= n_splits
        
    mean_performance = pd.DataFrame.from_dict(mean_performance, orient='index')
    mean_performance.rename(columns = {
        'Accuracy':'Mean Fold Accuracy',
        'Macro-Precision':'Mean Fold Macro-Precision',
        'Macro-Recall':'Mean Fold Macro-Recall',
        'Macro-F-Score':'Mean Fold Macro-F-Score'
    }, inplace=True)
    return(mean_performance)
