from fastapi import FastAPI, HTTPException

from fastapi.responses import JSONResponse

from fastapi.requests import Request

from fastapi.middleware.cors import CORSMiddleware

from fastapi_cache import FastAPICache

from fastapi_cache.backends.inmemory import InMemoryBackend

from fastapi_cache.decorator import cache

from models import *

from tasks import *

from utils import *


app = FastAPI()


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)


FastAPICache.init(InMemoryBackend())


@app.get("/health")

async def health():

    return {"status": "ok", "usage_stats": {"requests": 0, "errors": 0}}


@app.post("/evaluate")

async def evaluate(text: str, task: str):

    try:

        task_func = getattr(tasks, task)

        results = {}

        for model in models:

            results[model.__name__] = task_func(text, model)

        return JSONResponse(content=results, media_type="application/json")

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark")

async def benchmark(dataset: List[str]):

    try:

        results = {}

        for model in models:

            results[model.__name__] = benchmarking.benchmark(model, dataset)

        return JSONResponse(content=results, media_type="application/json")

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_model")

async def upload_model(model_file: UploadFile):

    # Bonus feature: implement model uploading and registration

    pass


@app.get("/models")

async def get_models():

    return [{"name": model.__name__, "description": model.__doc__} for model in models]


@app.get("/tasks")

async def get_tasks():

    return [{"name": task, "description": tasks.__dict__[task].__doc__} for task in tasks.__dict__]


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
