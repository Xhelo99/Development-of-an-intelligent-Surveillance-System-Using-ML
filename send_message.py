import sqlite3
import uuid

class Database:
    def __init__(self):
        self.connection = sqlite3.connect("database/data.db")
        self.cursor = self.connection.cursor()


    def add_message(self, detected_time, message):
        self.messageId = str(uuid.uuid4())
        self.message = f"Person {message}"

        self.cursor.execute("INSERT INTO messages (messageId, personDetectedTime, messageDiscription) VALUES (?, ?, ?)",
                       (self.messageId, detected_time, self.message))

        self.connection.commit()

