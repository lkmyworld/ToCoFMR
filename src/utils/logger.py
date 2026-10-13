import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    logfilename = '{}-{}-{}.log'.format(config['model'].lower(), config['dataset'], get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fformatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='[%Y/%m/%d %H:%M:%S]')

    sformatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='[%Y/%m/%d %H:%M:%S]')

    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers = [fh, sh]
    )
    return logfilepath


def format_metrics(result, config):
    parts = []
    for metric in config['metrics']:
        values = ', '.join(f"{result[f'{metric.lower()}@{k}']:.4f}" for k in config['topk'])
        parts.append(f"{metric}: [{values}]")
    return ' | '.join(parts)
