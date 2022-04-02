import click
import numpy as np
from config_pipeline import config
from utils.logger import Logger
from utils.models.lgbm import LightGBM


def train():

    model = LightGBM(
        experiment_hparams=config['output_artefacts'],
        optmizer_hparams=config['optmizer_hparams'],
        model_hparams=None,
    )
    model.hyperopt()
    _, fold_scores_val, val_thresholds, fold_scores_test, test_thresholds = model.run_cv(save_model=True)

    logger = Logger()

    # save logs
    logger.kpi_logger.info('=============================================')
    logger.kpi_logger.info(f'Model metric, val = {fold_scores_val}')
    logger.kpi_logger.info(f'Model metric, val_mean = {np.mean(fold_scores_val)}')
    logger.kpi_logger.info(f'Model metric, val_std = {np.std(fold_scores_val)}')

    logger.kpi_logger.info(f'Thresholds, val = {val_thresholds}')
    logger.kpi_logger.info(f'Thresholds, val mean = {np.mean(val_thresholds)}')
    logger.kpi_logger.info(f'Thresholds, val std = {np.std(val_thresholds)}')

    logger.kpi_logger.info(f'Model metric, test = {fold_scores_test}')
    logger.kpi_logger.info(f'Model metric, test_mean = {np.mean(fold_scores_test)}')
    logger.kpi_logger.info(f'Model metric, test_std = {np.std(fold_scores_test)}')

    logger.kpi_logger.info(f'Thresholds, test = {test_thresholds}')
    logger.kpi_logger.info(f'Thresholds, test mean = {np.mean(test_thresholds)}')
    logger.kpi_logger.info(f'Thresholds, test std = {np.std(test_thresholds)}')

    logger.kpi_logger.info('=============================================')

    return True


if __name__ == '__main__':
    train()
