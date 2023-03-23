import numpy as np
import os
import pandas as pd
import sys

def get_metrics_names(metrics):
    """Gets the names of the metrics.

    Args:
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys."""
    if len(metrics) == 0:
        return []
    metrics_dict = next(iter(metrics.values()))
    return list(metrics_dict.keys())

def print_metrics(metrics, weights):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = get_metrics_names(metrics)

    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 90th percentile %g' \
              % (metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))
        
    micros = [metrics[c]['microf1'] for c in sorted(metrics)]
    final_micro = np.average(micros, weights=ordered_weights)
    loss = [metrics[c]['loss'] for c in sorted(metrics)]
    final_loss = np.average(loss, weights=ordered_weights)
    macro = [metrics[c]['macrof1'] for c in sorted(metrics)]
    final_macro = np.average(macro, weights=ordered_weights)
    return final_loss, final_micro, final_macro