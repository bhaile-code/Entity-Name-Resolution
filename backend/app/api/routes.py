"""
API route handlers.
Separates endpoint logic from application setup.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.models import ProcessingResult
from app.services import NameMatcher
from app.utils import CSVHandler
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

# Create router
router = APIRouter()

# Initialize the name matcher (singleton for this module)
name_matcher = NameMatcher()


@router.get("/health")
async def health_check():
    """
    Detailed health check endpoint.

    Returns application status and version information.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "project_name": settings.PROJECT_NAME,
        "components": {
            "api": "operational",
            "name_matcher": "operational"
        }
    }


@router.post("/process", response_model=ProcessingResult)
async def process_companies(file: UploadFile = File(...)):
    """
    Process uploaded CSV file containing company names.

    Expected CSV format: Single column with header (e.g., 'company_name')
    The first column will be used regardless of header name.

    Args:
        file: Uploaded CSV file

    Returns:
        ProcessingResult with mappings, audit log, and summary statistics

    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Validate file type
        if not CSVHandler.validate_file_extension(file.filename, settings.ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Only {', '.join(settings.ALLOWED_EXTENSIONS)} files are supported"
            )

        # Read and parse CSV file
        contents = await file.read()
        df = CSVHandler.parse_csv(contents)

        # Extract company names
        company_names, column_name = CSVHandler.extract_company_names(df)

        logger.info(f"Processing {len(company_names)} unique company names from '{file.filename}'")

        # Process names through matcher
        result = name_matcher.process_names(company_names, filename=file.filename)

        logger.info(
            f"Successfully processed {len(result['mappings'])} companies into "
            f"{len(set(m['canonical_name'] for m in result['mappings']))} groups"
        )

        return result

    except ValueError as e:
        # Validation errors (from CSVHandler)
        logger.warning(f"Validation error for file '{file.filename}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"Error processing file '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
