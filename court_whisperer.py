import os
import pandas as pd
import backoff
from tqdm import tqdm
import requests
import time
import whisper
from typing import List, Optional, Sequence
from pydantic import BaseModel, Field
from langchain.chains import create_structured_output_runnable
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CourtroomParty(BaseModel):
    """Identifying information about a person."""
    
    person: str = Field(..., description="The name of the person speaking")
    role: Optional[str] = Field(None, description="The role of the person speaking")
    background: Optional[str] = Field(None, description="A concise summary about the person")
    

class OralArgument(BaseModel):
    """Identifying information about a particular argument."""
    
    topic: str = Field(..., description="A concise topic title for the argument")
    side: Optional[str] = Field(None, description="The side the argument favors, plaintiff or defense")
    person: Optional[str] = Field(None, description="The name of the person presenting the argument")
    summary: Optional[str] = Field(None, description="A concise summary of key points")
    

class OralDocket(BaseModel):
    """Identifying information in a text."""

    parties: Sequence[CourtroomParty] = Field(..., description="The information of interest in the text")
    arguments: Sequence[OralArgument] = Field(..., description="The information of interest in the text")


class CourtListenerExtractor:
    def __init__(self, database_path, base_url="https://www.courtlistener.com/api/rest/v3/audio/"):
        """
        Initializes the CourtListenerExtractor with the path to the database file.

        Args:
        database_path (str): The path to the Excel file database.
        """
        self.database_path = database_path
        self.base_url=base_url

    def fetch_records(self, start_page, end_page):
        """
        Fetches records from Court Listener based on a page range.
        
        Args:
        start_page (int): court listener web search start
        end_page (int): court listener web search start

        Returns:
        pd.DataFrame: A DataFrame containing the fetched records.
        """
        all_results = []
        for page in tqdm(range(start_page, end_page + 1)):
            url = f"{self.base_url}?page={page}"
            data = self.fetch_page_(url)
            all_results.extend(data["results"])
            time.sleep(1)
        df = pd.DataFrame.from_records(all_results)
        return df

    def add_rows_to_excel(self, input_df):
        # Convert lists to strings in input_df
        input_df = input_df.applymap(lambda x: str(x) if isinstance(x, list) else x)
        
        existing_df = pd.DataFrame()
        if os.path.exists(self.database_path):
            # Load the existing data
            existing_df = pd.read_excel(self.database_path)

            # Convert lists to strings in existing_df
            existing_df = existing_df.applymap(lambda x: str(x) if isinstance(x, list) else x)

        # Concatenate the existing data with the new data
        combined_df = pd.concat([existing_df, input_df])

        # Remove the 'unique_index' column before removing duplicates
        combined_df = combined_df.drop(columns=['unique_index'])

        # Remove duplicates
        final_df = combined_df.drop_duplicates()

        # Reassign the 'unique_index' column
        final_df['unique_index'] = range(1, len(final_df) + 1)

        print(f"Total number of rows: {len(final_df)}")
        print(f"Number of new rows added: {len(final_df) - len(existing_df)}")

        # Write the data to the Excel file
        final_df.to_excel(self.database_path, index=False)

    def update_database(self, new_records):
        """
        Updates the database with new records.

        Args:
        new_records (pd.DataFrame): New records to be added to the database.
        """
        self.add_rows_to_excel(new_records)
        
    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, requests.exceptions.HTTPError), max_tries=8)
    def fetch_page_(self, url):
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        return response.json()
    
    def find_subset_with_keyword(self, keyword="insurance"):
        """
        Finds a subset of records containing a specific keyword.

        Args:
        keyword (str, optional): The keyword to search for. Defaults to "insurance".

        Returns:
        pd.DataFrame: A DataFrame containing the subset of records with the keyword.
        """
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database file {self.database_path} not found.")

        # Load the existing data
        df = pd.read_excel(self.database_path)

        # Finding the subset with the keyword
        subset_df = df[df.apply(lambda row: row.astype(str).str.contains(keyword).any(), axis=1)]
        print(f"Records with keyword: {keyword}: {len(subset_df)}")
        
        return subset_df

# Example usage:
# extractor = CourtListenerExtractor("database.xlsx")
# new_records = extractor.fetch_records(1, 6)
# extractor.update_database(new_records)


class AudioTranscriber:
    def __init__(self, download_path, text_output_path, model_name='base'):
        self.download_path = download_path
        self.text_output_path = text_output_path
        self.transcription_model = whisper.load_model(model_name)
        self._create_directory_if_not_exists(download_path)
        self._create_directory_if_not_exists(text_output_path)

    def _create_directory_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def download_mp3(self, url, file_name, unique_index):
        file_path = os.path.join(self.download_path, f"{file_name}_{unique_index}.mp3")
        if os.path.exists(file_path):
            print(f"File {file_name}.mp3 already exists.")
            return file_path

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"File downloaded and saved as {file_path}")
            return file_path
        except requests.RequestException as e:
            print(f"An error occurred: {e}")

    def transcribe_audio(self, file_path):
        result = self.transcription_model.transcribe(file_path)
        return result['text']
    
    def save_transcription(self, text, file_name, unique_index):
        text_file_path = os.path.join(self.text_output_path, f"{file_name}_{unique_index}.txt")
        with open(text_file_path, 'w') as file:
            file.write(text)
        print(f"Transcription saved as {text_file_path}")

    def read_transcription(self, file_name, unique_index):
        text_file_path = os.path.join(self.text_output_path, f"{file_name}_{unique_index}.txt")
        if not os.path.exists(text_file_path):
            print(f"No transcription file found for {file_name}")
            return None
        with open(text_file_path, 'r') as file:
            return file.read()

    def process_batch(self, df):
        for _, row in df.iterrows():
            url = row['download_url']
            name = re.sub(r'\W+', '', row['case_name'])
            unique_index = row['unique_index']
            file_path = self.download_mp3(url, name, unique_index)
            if file_path:
                transcription = self.transcribe_audio(file_path)
                self.save_transcription(transcription, name, unique_index)

# Example usage:
# audio_transcriber = AudioTranscriber('audio', 'transcriptions')
# extractor = CourtListenerExtractor("database.xlsx")
# df = extractor.fetch_records(1, 11)
# insurance_cases = extractor.find_subset_with_keyword()
# audio_transcriber.process_batch(insurance_cases)


class OralDocketExtractor:
    def __init__(self, model_name="gpt-4-1106-preview"):
        self.model = ChatOpenAI(model=model_name, temperature=0.1)

    def process_document(self, document_path: str) -> OralDocket:
        loader = TextLoader(document_path)
        doc = loader.load()
            
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information in structured formats.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {input}",
            ),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )

        runnable = create_structured_output_runnable(OralDocket, self.model, prompt)
        result = runnable.invoke({"input": doc})
        
        # Extract unique_index from the filename
        filename = os.path.basename(document_path)
        unique_index = filename.split("_")[-1].split(".")[0]
        
        return result, unique_index
    
    def save_to_excel(self, data: List[OralDocket], file_name: str, unique_index: int):
        data_dicts = [instance.dict() for instance in data]

        # Save 'parties' and 'arguments' to separate sheets
        for key in ['parties', 'arguments']:
            df_new = pd.concat([pd.json_normalize(data_dict[key]) for data_dict in data_dicts], ignore_index=True)
            df_new['unique_index'] = unique_index  # Add the unique_index column
            self.write_to_excel(df_new, key, file_name)

    @staticmethod
    def write_to_excel(df_new, sheet_name, file_name='court_whisperer_data.xlsx'):
        if os.path.exists(file_name):
            try:
                # Attempt to read the existing sheet
                df_old = pd.read_excel(file_name, sheet_name=sheet_name)
                df = pd.concat([df_old, df_new]).drop_duplicates()
            except ValueError:
                # If the sheet does not exist, use the new dataframe as is
                df = df_new

            # Open the existing workbook using 'a' mode (append)
            with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # If the sheet exists, it will be overwritten
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create a new file if it doesn't exist
            df_new.to_excel(file_name, sheet_name=sheet_name, index=False)

# Example Usage:
# llm_processor = OralDocketExtractor()
# result, unique_index = llm_processor.process_document(f"transcriptions/{text_file}")
# llm_processor.save_to_excel([result], "output_file_v2.xlsx", unique_index)
