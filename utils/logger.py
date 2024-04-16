
import logging
import coloredlogs

from pathlib import Path
from rich.logging import RichHandler


class Logger:
    def __init__(self, logger_name='my_logger', level='DEBUG', verbose=False, filepath=None):
        # Create a logger object.
        self.logger = logging.getLogger(logger_name)

        # Set log level.
        self.logger.setLevel(level)

        # Remove all handlers associated with the logger object.
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Set log level.
        if verbose:
            coloredlogs.install(level=level, logger=self.logger)
            rich_handler = RichHandler()
            self.logger.addHandler(rich_handler)

        # Setup logging to file if a filepath is provided.
        if filepath is not None:
            file_handler = logging.FileHandler(
                Path(filepath) / Path(logger_name + '.log'))
            self.logger.addHandler(file_handler)

        # Setup logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# Usage
if __name__ == "__main__":
    logger = Logger('custom_logger', 'INFO', verbose=False, filepath='log.txt')
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
