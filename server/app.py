# server/app.py
import json
import os
import subprocess
import sys
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import CreditAction, CreditObservation
from .environment import CreditAnalystEnvironment    # ✅ relative import
from .tasks import TASK_REGISTRY                     # ✅ relative import
from .data_generator import FIELD_RANGES             # ✅ relative import

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

env = CreditAnalystEnvironment()

app = FastAPI(
    title="CredLess-Env",
    description="OpenEnv RL environment for alternative credit scoring.",
    version="1.0.0",
)


class ResetRequest(BaseModel):
    task_name: str = "binary_decision"
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    obs = env.reset(task_name=body.task_name, seed=body.seed)
    return {
        "observation": obs.dict(),
        "reward": 0.0,          # ✅ top-level reward
        "done": False,          # ✅ top-level done
        "info": {"task_name": obs.task_name},  # ✅ required info
    }


@app.post("/step")
def step(action: CreditAction):
    obs = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": obs.step_reward,          # ✅ top-level reward
        "done": obs.done,                   # ✅ top-level done
        "info": {                           # ✅ required info dict
            "task_name":         obs.task_name,
            "cumulative_reward": obs.cumulative_reward,
            "episode_score":     obs.episode_score,
        },
    }


# server/app.py — state endpoint (remove .state, call .state())

@app.get("/state")
def state():
    return env.state().dict()    # ✅ call as method, not property


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_REGISTRY}


@app.get("/grader")
def grader_info():
    s = env.state()
    return {
        "episode_id":        s.episode_id,
        "task_name":         s.task_name,
        "steps_taken":       s.step_count,
        "cumulative_reward": round(s.cumulative_reward, 4),
        "fields_requested":  s.fields_requested,
        "note":              "episode_score is in the observation when done=True",
    }


@app.get("/baseline")
def run_baseline():
    # ✅ Guard: check API key before launching
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={"error": "OPENAI_API_KEY environment variable not set"},
        )
    try:
        result = subprocess.run(
            ["python", "baseline.py", "--output-json"],
            capture_output=True, text=True, timeout=300,
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
    import os
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 7860)),
        workers=int(os.getenv("WORKERS", 2))
    )

if __name__ == "__main__":
    main()