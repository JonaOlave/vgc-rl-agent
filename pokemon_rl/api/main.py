from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pokemon_rl.api.presentation.training_router import router as training_router
from pokemon_rl.api.presentation.evaluation_router import router as evaluation_router

app = FastAPI(title="VGC RL Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_router)
app.include_router(evaluation_router)


@app.get("/health")
def health():
    return {"status": "ok"}
