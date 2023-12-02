import os
from openai import api
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to the database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

def transcribe_mp3_and_save(records):
    for record in records:
        audio_file = f"{record['id']}.mp3"
        result = api.transcription.create(
            audio=audio_file,
            model="whisper",
            token=OPENAI_API_KEY
        )
        save_transcription(record['id'], result['text'])

def save_transcription(docket_id, transcription):
    cur = conn.cursor()
    cur.execute("INSERT INTO transcriptions (docket_id, transcription) VALUES (%s, %s)", 
                (docket_id, transcription))
    conn.commit()

if __name__ == "__main__":
    # Fetch records from the database
    cur = conn.cursor()
    cur.execute("SELECT * FROM dockets")
    records = cur.fetchall()
    transcribe_mp3_and_save(records)
