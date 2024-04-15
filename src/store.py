import json
import random
import string

class Store:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_task(self, confirmed_task, website):
        # Generate a random task ID
        task_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) + "-SearchTest"

        # Create a dictionary with the task details
        task_data = {
            "confirmed_task": confirmed_task,
            "website": website,
            "task_id": task_id
        }

        # Write the task data to the JSON file, overriding any existing data
        with open(self.file_path, "w") as file:
            json.dump([task_data], file, indent=2)