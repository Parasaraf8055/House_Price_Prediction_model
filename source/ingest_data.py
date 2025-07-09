import os
from abc import ABC , abstractmethod
import zipfile

import pandas as pd

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a file"""
        pass

class ZipDataIngestor(DataIngestor):
    """extract a zip file and return content or data"""

    def ingest(self, file_path:str)->pd.DataFrame:

        if not file_path.endswith(".zip"):
            raise ValueError("file is not a zip file")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("Extracted_data")    

        extracted_files = os.listdir("Extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in the zip file")
        
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in the zip file")
        
        csv_file_path = os.path.join("Extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df
    

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"no ingestor available for file extention: {file_extension}")
        


if __name__ == "__main__":
    file_path = "./data/archive.zip"

    file_extension = os.path.splitext(file_path)[1]

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    df = data_ingestor.ingest(file_path)

    print(df.head())

    

