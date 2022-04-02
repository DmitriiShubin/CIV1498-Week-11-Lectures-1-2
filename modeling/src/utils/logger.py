import logging


class Logger:
    def __init__(self):

        # logger to save KPI (metrics, config, etc.)
        self.kpi_logger = logging.getLogger('Training pipeline')
        self.kpi_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('training.log')
        fh.setLevel(logging.DEBUG)
        self.kpi_logger.addHandler(fh)
