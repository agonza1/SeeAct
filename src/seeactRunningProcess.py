import subprocess
import os
import fcntl
import time
import threading

class SeeactRunningProcess:
    def __init__(self, command):
        self.command = command
        self.process = None
        self.running = False
        self.callback = None
        self.error_callback = None
        self.process_thread = None

    def start_process(self):
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        fcntl.fcntl(self.process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(self.process.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        self.running = True

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.running = False

    def set_callback(self, callback):
        self.callback = callback

    def set_error_callback(self, error_callback):
        self.error_callback = error_callback

    def read_output(self):
        while self.running:
            try:
                line = self.process.stdout.readline().decode().strip()
                stderr_line = self.process.stderr.readline().decode().strip()
                
                if line:
                    print(line)  # Print the line
                    if self.callback:
                        self.callback(line)  # Call the callback function

                if stderr_line:
                    if self.error_callback:
                        self.error_callback(stderr_line)

                time.sleep(0.3)  # Wait for 300ms

            except Exception as e:
                print(f"Error reading output: {e}")
                pass  # Exit loop if error

    def run(self):
        self.start_process()
        self.process_thread = threading.Thread(target=self.read_output)
        self.process_thread.start()

    def join(self):
        if self.process_thread:
            self.process_thread.join()
