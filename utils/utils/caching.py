from fastapi_cache import cache


@cache(ttl=60)  # Cache for 1 minute

def cache_model_output(text: str, task: str, model_name: str):

    # Implement caching logic to store model outputs

    pass
