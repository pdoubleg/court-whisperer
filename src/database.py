import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Connect to the database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE dockets (
            id INTEGER PRIMARY KEY,
            absolute_url VARCHAR(255),
            case_name VARCHAR(255),
            date_argued DATE,
            date_reargued DATE
        )
        """,
        """
        CREATE TABLE transcriptions (
            id INTEGER PRIMARY KEY,
            docket_id INTEGER,
            transcription TEXT,
            FOREIGN KEY (docket_id)
            REFERENCES dockets (id)
            ON UPDATE CASCADE ON DELETE CASCADE
        )
        """)
    cur = conn.cursor()
    # create table one by one
    for command in commands:
        cur.execute(command)
    # close communication with the PostgreSQL database server
    cur.close()
    # commit the changes
    conn.commit()

if __name__ == '__main__':
    create_tables()
