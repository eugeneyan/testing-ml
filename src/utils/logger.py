"""
Logger utility
"""
import datetime
import logging
import os


def get_file_handler(job: str, log_path: str, log_variables=None):
    """
    Get a file handler to write logs to

    Args:
        job: Name of job
        log_path: Dir path to write log to
        log_variables: Variables to be in the log name

    Returns:
        File handler

    Examples:
        fh = get_file_handler('ar-poc', '../')
        logger.addHandler(fh)
    """
    if log_variables is None:
        log_variables = []
    log_path = os.path.join(os.getcwd(), log_path, job)
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    log_name = log_path + '_'.join([''] + log_variables) + '_' + current_datetime + '.log'
    fh = logging.FileHandler(os.path.join(os.getcwd(), log_path, log_name), encoding='utf8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    fh.setLevel(logging.DEBUG)
    return fh


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

# add ch to logger
logger.addHandler(ch)
