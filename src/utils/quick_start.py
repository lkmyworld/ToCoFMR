from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger, format_metrics
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer
import platform
import os


def quick_start(model, dataset, config_dict=None):
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    init_seed(2026)

    logger.info('=' * 60)
    logger.info('Model: \t\t' + model)
    logger.info('Dataset: \t' + dataset)
    logger.info('Server: \t' + platform.node())
    logger.info('Dir: \t\t' + os.getcwd())
    logger.info('=' * 60)
    parameters = 'Hyper-parameters: \n'

    for idx, (key, value) in enumerate(config.final_config_dict.items()):
        if idx == len(config.final_config_dict) - 1:
            parameters += '\t\t{}: {}'.format(key, str(value))
        else:
            parameters += '\t\t{}: {}\n'.format(key, str(value))
    logger.info(parameters)
    logger.info('=' * 60)
    logger.info('dataset loaded, run model...')

    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0
    hyper_ls = []

    other_hyper_parameters = [
        name
        for name in config['hyper_parameters']
        if name not in ("missing_ratio", "seed")
    ]
    config['hyper_parameters'] = [
        "missing_ratio",
        *other_hyper_parameters,
        "seed",
    ]
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        train_data.pretrain_setup()

        model = get_model(config['model'])(config, train_data).to(config['device'])

        logger.info('========={}/{}: Parameters: {} = {}========='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))
        trainer = get_trainer()(config, model)

        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data)
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('Valid {}'.format(format_metrics(best_valid_result, config)))

        logger.info('Test  {}'.format(format_metrics(best_test_upon_valid, config)))

        logger.info("-")
        logger.info('████Current BEST████\nParameters: {} = {}\nValid {}\nTest  {}\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            format_metrics(hyper_ret[best_test_idx][1], config),
            format_metrics(hyper_ret[best_test_idx][2], config),
        ))

    logger.info('=====================All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {} = {}\nValid {}\nTest  {}\n'.format(
            config['hyper_parameters'],
            p,
            format_metrics(k, config),
            format_metrics(v, config),
        ))

    logger.info('████████████████ BEST RESULT ████████████████\nParameters: {} = {}\nBest valid {}\nBest test  {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        format_metrics(hyper_ret[best_test_idx][1], config),
        format_metrics(hyper_ret[best_test_idx][2], config),
    ))
