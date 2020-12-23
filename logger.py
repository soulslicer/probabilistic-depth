import logging
import logging.config
import logging.handlers
from path import Path
import multiprocessing_logging
multiprocessing_logging.install_mp_handler()

import sys

class ColorPrint:

    @staticmethod
    def print_fail(message, end = '\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_pass(message, end = '\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end = '\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_info(message, end = '\n'):
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end = '\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)

class Logger():
    def __init__(self, level, log_dir, log_name, filename):
        self.logger = logging.getLogger(log_name)

        # logging to file
        # fh = logging.handlers.RotatingFileHandler(
        #     Path(log_dir) / 'main_logger_{:%H%M%S}.log'.format(datetime.now()),
        #     'w', 20 * 1024 * 1024, 5)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / filename, 'w', 20 * 1024 * 1024, 5)
        formatter = logging.Formatter('%(asctime)s %(levelname)5s - %(name)s '
                                      '[%(filename)s line %(lineno)d] - %(message)s',
                                      datefmt='%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # logging to screen
        fh = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s', )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.setLevel(level)
        self.logger.info("Start training")

    def info(self, id, input):
        id_str = " [" + str(id) + "] "
        self.logger.info(id_str + input)

    def warning(self, id, input):
        id_str = " [" + str(id) + "] "
        self.logger.warning(id_str + input)

    def error(self, id, input):
        id_str = " [" + str(id) + "] "
        self.logger.error(id_str + input)

def init_logger(level='INFO', log_dir='./', log_name='main_logger', filename='main.log'):
    return Logger(level, log_dir, log_name, filename)