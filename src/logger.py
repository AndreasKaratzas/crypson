
import os
import time
import hashlib


class Logger:
    def __init__(self, tb_logger, wandb_logger, run_name, project_dir):
        self.tb_logger = tb_logger
        self.wandb_logger = wandb_logger
        self.run_name = run_name
        self.project_dir = project_dir
        self.str_log = ('PARTIAL COPY OF TEXT LOG TO TENSORBOARD TEXT \n'
                        'class Log_and_print() by Arian Prabowo \n'
                        'RUN NAME: ' + run_name + ' \n \n')

        # Generate an 8-digit hash based on the timestamp and run name
        timestamp = time.strftime('%Y%m%d%H%M%S')
        hash_input = f"{timestamp}_{run_name}"
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()[:8]

        # Create the logs directory if it doesn't exist
        logs_dir = os.path.join(project_dir, 'log')
        os.makedirs(logs_dir, exist_ok=True)

        # Create the log file path
        self.log_file = os.path.join(logs_dir, f'experiment-{hash_hex}.log')

    def lnp(self, tag):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'{timestamp} - {tag}\n'
        self.str_log += log_message

        # Write the log message to the file
        with open(self.log_file, 'a') as file:
            file.write(log_message)

    def dump_to_tensorboard(self):
        self.tb_logger.experiment.add_text('log', self.str_log)

    def dump_to_wandb(self):
        self.wandb_logger.experiment.summary['log'] = self.str_log
