"""
CSV file handling utilities.
Separates file I/O concerns from business logic.
"""
import pandas as pd
from io import StringIO
from typing import List, Tuple
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CSVHandler:
    """Handles CSV file reading and validation."""

    @staticmethod
    def parse_csv(content: bytes) -> pd.DataFrame:
        """
        Parse CSV content from bytes.

        Args:
            content: Raw CSV file bytes

        Returns:
            Pandas DataFrame

        Raises:
            ValueError: If CSV is invalid or empty
        """
        try:
            csv_string = content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_string))
            return df
        except UnicodeDecodeError:
            raise ValueError("File encoding error. Please ensure CSV is UTF-8 encoded.")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")

    @staticmethod
    def extract_company_names(df: pd.DataFrame) -> Tuple[List[str], str]:
        """
        Extract company names from DataFrame.
        Uses first column and returns unique, non-null values.

        Args:
            df: DataFrame to extract from

        Returns:
            Tuple of (list of company names, column name used)

        Raises:
            ValueError: If DataFrame is invalid
        """
        if df.empty:
            raise ValueError("CSV file contains no data")

        if len(df.columns) == 0:
            raise ValueError("CSV has no columns")

        # Use first column
        column_name = df.columns[0]

        # Extract unique, non-null names
        company_names = df[column_name].dropna().astype(str).unique().tolist()

        if len(company_names) == 0:
            raise ValueError("No valid company names found in CSV")

        logger.info(f"Extracted {len(company_names)} unique company names from column '{column_name}'")

        return company_names, column_name

    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """
        Validate file has allowed extension.

        Args:
            filename: Name of file to validate
            allowed_extensions: List of allowed extensions (e.g., ['.csv'])

        Returns:
            True if valid, False otherwise
        """
        return any(filename.lower().endswith(ext) for ext in allowed_extensions)
