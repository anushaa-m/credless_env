import json
import os
import subprocess
from typing import Any, Optional

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import FinVerseAction
from .environment import CreditAnalystEnvironment
from .tasks import TASK_REGISTRY

env = CreditAnalystEnvironment()

app = FastAPI(
    title="CredLess-Env",
    description="OpenEnv RL environment for alternative credit scoring.",
    version="2.0.0",
)


class ResetRequest(BaseModel):
    task_name: str = "binary_decision"
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    return env.reset(task_name=body.task_name, seed=body.seed)


@app.post("/step")
def step(action: Any = Body(..., embed=False)):
    return env.step(action)


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_REGISTRY}


@app.get("/grader")
def grader_info():
    s = env.state()
    return {
        "episode_id": s.episode_id,
        "session_id": s.session_id,
        "task_difficulty": s.task_difficulty,
        "steps_taken": s.steps_taken,
        "fraud_flags": s.fraud_flags,
        "market_state_visible": bool(s.market_state),
        "auditor_history_length": len(s.auditor_compliance_log),
        "note": "episode_score is returned in the observation when done=True",
    }


@app.get("/baseline")
def run_baseline():
    try:
        result = subprocess.run(
            ["python", "baseline.py", "--output-json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        scores = json.loads(result.stdout)
    except json.JSONDecodeError:
        scores = {"error": "Parse failed", "stderr": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        scores = {"error": "Baseline timed out (>300s)"}
    except Exception as exc:
        scores = {"error": str(exc)}
    return JSONResponse(content=scores)


def main():
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 7860)),
        workers=int(os.getenv("WORKERS", 2)),
    )


if __name__ == "__main__":
    main()
