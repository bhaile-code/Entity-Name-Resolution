"""
API route handlers.
Separates endpoint logic from application setup.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.models import ProcessingResult
from app.services import NameMatcher
from app.utils import CSVHandler
from app.utils.logger import setup_logger
from app.config import settings

logger = setup_logger(__name__)

# Create router
router = APIRouter()


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
async def process_companies(
    file: UploadFile = File(...),
    use_adaptive_threshold: bool = Query(default=False),
    embedding_mode: str = Query(default=None),
    clustering_mode: str = Query(default=None),
    hac_threshold: float = Query(default=None),
    hac_linkage: str = Query(default=None)
):
    """
    Process uploaded CSV file containing company names.

    Expected CSV format: Single column with header (e.g., 'company_name')
    The first column will be used regardless of header name.

    Args:
        file: Uploaded CSV file
        use_adaptive_threshold: (Deprecated) If True, use GMM-based adaptive thresholding
        embedding_mode: Embedding mode ('openai-small', 'openai-large', 'local', 'disabled')
                       If None, uses DEFAULT_EMBEDDING_MODE from settings
        clustering_mode: Clustering mode ('fixed', 'adaptive_gmm', 'hac')
                        If None, uses CLUSTERING_MODE from settings or infers from use_adaptive_threshold
        hac_threshold: HAC distance threshold (0-1 range, e.g., 0.42 for 58% similarity)
                      If None, uses HAC_DISTANCE_THRESHOLD from settings
        hac_linkage: HAC linkage method ('average', 'single', 'complete', 'ward')
                    If None, uses HAC_LINKAGE_METHOD from settings

    Returns:
        ProcessingResult with mappings, audit log, summary statistics, and optional metadata
        (gmm_metadata for adaptive_gmm mode, hac_metadata for hac mode)

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

        # Determine clustering mode for logging
        effective_mode = clustering_mode or ('adaptive_gmm' if use_adaptive_threshold else settings.CLUSTERING_MODE)

        logger.info(
            f"Processing {len(company_names)} unique company names from '{file.filename}' "
            f"(mode={effective_mode}, embedding={embedding_mode or 'default'}, "
            f"hac_threshold={hac_threshold or 'default'}, hac_linkage={hac_linkage or 'default'})"
        )

        # Initialize matcher with appropriate mode
        matcher = NameMatcher(
            use_adaptive_threshold=use_adaptive_threshold,
            embedding_mode=embedding_mode,
            clustering_mode=clustering_mode,
            hac_threshold=hac_threshold,
            hac_linkage=hac_linkage
        )

        # Process names through matcher
        result = matcher.process_names(company_names, filename=file.filename)

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
