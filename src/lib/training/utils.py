"""Training-specific utility functions."""
import logging
import json
import numpy as np
from os import path

log = logging.getLogger(__name__)

def get_time_bundle(times):
    """Given a list of times, returns a tuple containing the
    total time, standard deviation, mean time, and a numpy array of the times.
    """
    times = np.array(times)
    std_time, mean_time = np.std(times), np.mean(times)
    total_time = np.sum(times)
    return (total_time, std_time, mean_time, times)


def write_results(model_name, output_dir, results):
    """Write the given model results to results.json in the provided
    output directory

    Args:
        model_name (str): Model name
        output_dir (str): Output directory
        results (dict): Dictionary of model results
    """
    results_path = path.join(output_dir, 'results.json')

    if path.exists(results_path):
        log.info('Existing file found, appending results')
        with open(results_path, 'rb') as f:
            contents = json.load(f)
        log.debug(f'Existing contents: {contents}')

        contents['results'].extend(results)

        mn = model_name
        if contents['model_name'] != mn:
            log.warn(f'[WARNING]: Model names do not match - {contents["model_name"]} vs {mn}')

        with open(results_path, 'w') as f:
            json.dump({
                'model_name': mn,
                'results': contents['results'],
            }, f, indent=4)
        log.info(f'Appended results to {results_path}')
    else:
        log.info('No results file found, writing to new one')
        with open(results_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'results': results,
            }, f, indent=4)
        log.info(f'Wrote results to file at {results_path}')