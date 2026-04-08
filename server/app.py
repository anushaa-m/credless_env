# server/app.py
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import (
    CreditAction, 
    CreditObservation, 
    CreditState,
    StepResponse,
    ResetResponse,
    HealthResponse,
    StateResponse
)
from .environment import CreditAnalystEnvironment    # ✅ relative import
from .tasks import TASK_REGISTRY                     # ✅ relative import
from .data_generator import FIELD_RANGES             # ✅ relative import

env = CreditAnalystEnvironment()

app = FastAPI(
    title="CredLess-Env",
    description="OpenEnv RL environment for alternative credit scoring.",
    version="1.0.0",
)


class ResetRequest(BaseModel):
    task_name: str = "binary_decision"
    seed: Optional[int] = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/reset", response_model=ResetResponse)
def reset(body: ResetRequest = ResetRequest()):
    obs = env.reset(task_name=body.task_name, seed=body.seed)
    obs.done = False
    return ResetResponse(
        observation=obs,
        reward=0.0,          # ✅ top-level reward
        done=False,          # ✅ top-level done
        info={"task_name": obs.task_name},  # ✅ required info
    )


@app.post("/step", response_model=StepResponse)
def step(action: CreditAction):
    obs = env.step(action)
    return StepResponse(
        observation=obs,
        reward=obs.step_reward,          # ✅ top-level reward
        done=obs.done,                   # ✅ top-level done
        info={                           # ✅ required info dict
            "task_name":         obs.task_name,
            "cumulative_reward": obs.cumulative_reward,
            "episode_score":     obs.episode_score,
        },
    )


# server/app.py — state endpoint (remove .state, call .state())

@app.get("/state", response_model=StateResponse)
def state():
    return StateResponse(state=env.state())    # ✅ call as method, not property


def main():
    import uvicorn
    import os
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 7860)),
        workers=int(os.getenv("WORKERS", 2))
    )

if __name__ == "__main__":
    main()