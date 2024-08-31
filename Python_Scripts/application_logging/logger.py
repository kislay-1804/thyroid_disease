import os
from datetime import datetime

class App_Logger:
    # Method to log messages with timestamp
    def log(self, log_path, log_message):
        try:
            # Get the current date and time
            now = datetime.now()
            date = now.date()
            current_time = now.strftime("%H:%M:%S")

            # Open the log file and write the log message
            with open(log_path, 'a+') as log_file:
                log_file.write(f"{date}/{current_time}\t\t{log_message}\n")
        except Exception as e:
            raise e