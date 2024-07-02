import logging


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def log_usage(request: Request):

    logger.info(f"Request received: {request.method} {request.url}")


def log_error(request: Request, exception: Exception):

    logger.error(f"Error processing request {request.method} {request.url}: {str(exception)}")
