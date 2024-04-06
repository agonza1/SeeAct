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
        self.process_thread = None

    def start_process(self):
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        fcntl.fcntl(self.process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
        self.running = True

    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.running = False

    def set_callback(self, callback):
        self.callback = callback

    def read_output(self):
        while self.running:
            try:
                line = self.process.stdout.readline().decode().strip()
                if line:
                    print(line)  # Print the line
                    if self.callback:
                        self.callback(line)  # Call the callback function
                time.sleep(0.5)  # Wait for 500ms
            except:
                pass  # Ignore non-blocking read errors

    def run(self):
        self.start_process()
        self.process_thread = threading.Thread(target=self.read_output)
        self.process_thread.start()

    def join(self):
        if self.process_thread:
            self.process_thread.join()
