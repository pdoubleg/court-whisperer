import os
import requests
from bs4 import BeautifulSoup
import ffmpeg
from dotenv import load_dotenv
import psycopg2
from openai import api
import streamlit as st

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

# Fetch docket records and oral arguments from courtlistener.com
def fetch_records():
    url = "https://www.courtlistener.com/api/rest/v3/dockets/"
    response = requests.get(url)
    data = response.json()
    return data

# Update and maintain local postgreSQL database to store records
def update_database(records):
    cur = conn.cursor()
    for record in records:
        cur.execute("INSERT INTO dockets VALUES (%s, %s, %s, %s, %s)", 
                    (record['id'], record['absolute_url'], record['case_name'], record['date_argued'], record['date_reargued']))
    conn.commit()

# Download oral argument mp3 files from courtlistener.com
def download_mp3(records):
    for record in records:
        mp3_url = record['audio_files'][0]['absolute_url']
        response = requests.get(mp3_url)
        with open(f"{record['id']}.mp3", 'wb') as f:
            f.write(response.content)

# Transcribe mp3 to text using openai's whisper model
def transcribe_mp3(records):
    for record in records:
        audio_file = f"{record['id']}.mp3"
        result = api.transcription.create(
            audio=audio_file,
            model="whisper",
            token=OPENAI_API_KEY
        )
        with open(f"{record['id']}.txt", 'w') as f:
            f.write(result['text'])

# Simple Streamlit UI to execute core functionality
def main():
    st.title("Court Whisperer")
    if st.button("Fetch and Process New Records"):
        records = fetch_records()
        update_database(records)
        download_mp3(records)
        transcribe_mp3(records)
        st.success("Records processed successfully!")

if __name__ == "__main__":
    main()
