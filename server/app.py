import json
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import FinVerseAction
from credless_model.oracle import CreditOracle
from .environment import CreditAnalystEnvironment
from .models import FinVerseResponse
from .tasks import TASK_REGISTRY


oracle: CreditOracle | None = None
env: CreditAnalystEnvironment | None = None
request_count = 0
total_latency_ms = 0.0


def _init_runtime() -> tuple[CreditAnalystEnvironment, CreditOracle]:
    global oracle, env
    if oracle is None:
        oracle = CreditOracle()
    if env is None:
        env = CreditAnalystEnvironment()
    return env, oracle


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    _init_runtime()
    yield

app = FastAPI(
    title="CredLess-Env",
    description="OpenEnv RL environment for alternative credit scoring.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global request_count, total_latency_ms
    started = time.perf_counter()
    response = await call_next(request)
    total_latency_ms += (time.perf_counter() - started) * 1000.0
    request_count += 1
    return response


class ResetRequest(BaseModel):
    task_name: str = "binary_decision"
    seed: Optional[int] = None


def _require_runtime() -> tuple[CreditAnalystEnvironment, CreditOracle]:
    return _init_runtime()


def _build_response(observation_payload: dict[str, Any], *, reward: float, done: bool, info: dict[str, Any]) -> FinVerseResponse:
    active_env, active_oracle = _require_runtime()
    observation = observation_payload.get("observation", observation_payload)
    revealed = active_env.oracle_features()

    oracle_risk = active_oracle.predict_risk(revealed) if revealed else 0.0
    oracle_confidence = active_oracle.get_confidence(revealed) if revealed else 0.0
    top_factors = active_oracle.get_top_factors(revealed) if revealed else []
    explanation = str(observation.get("message") or info.get("explanation") or "")

    return FinVerseResponse(
        observation=observation,
        reward=round(float(reward), 4),
        done=bool(done),
        explanation=explanation,
        top_factors=top_factors,
        oracle_risk=round(float(oracle_risk), 6),
        oracle_confidence=round(float(oracle_confidence), 6),
        info=dict(info),
    )


def _invalid_step_response(error: Exception) -> FinVerseResponse:
    active_env, _active_oracle = _require_runtime()
    fallback_observation = active_env._build_observation(  # type: ignore[attr-defined]
        step_reward=-0.1,
        done=False,
        message=f"invalid action: {str(error)}",
    )
    return FinVerseResponse(
        observation=fallback_observation,
        reward=-0.1,
        done=False,
        explanation=f"invalid action: {str(error)}",
        top_factors=[],
        oracle_risk=0.0,
        oracle_confidence=0.0,
        info={"error": str(error)},
    )


@app.get("/health")
def health():
    return {"status": "healthy", "oracle_loaded": oracle is not None, "environment_loaded": env is not None}


@app.get("/metrics")
def metrics():
    avg_latency = total_latency_ms / request_count if request_count else 0.0
    return {
        "requests": request_count,
        "avg_latency_ms": round(float(avg_latency), 3),
    }


@app.post("/reset", response_model=FinVerseResponse)
def reset(body: ResetRequest = ResetRequest()):
    active_env, _active_oracle = _require_runtime()
    observation = active_env.reset(task_name=body.task_name, seed=body.seed)
    return _build_response(
        observation,
        reward=0.0,
        done=False,
        info={"explanation": "New applicant loaded. Begin investigation."},
    )


@app.post("/step", response_model=FinVerseResponse)
def step(action: Any = Body(..., embed=False)):
    try:
        active_env, _active_oracle = _require_runtime()
        result = active_env.step(action)
        return _build_response(
            result,
            reward=float(result.get("reward", 0.0)),
            done=bool(result.get("done", False)),
            info=dict(result.get("info", {})),
        )
    except Exception as exc:
        return _invalid_step_response(exc)


@app.get("/state")
def state():
    active_env, _active_oracle = _require_runtime()
    return active_env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_REGISTRY}


@app.get("/grader")
def grader_info():
    active_env, _active_oracle = _require_runtime()
    s = active_env.state()
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
