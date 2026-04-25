import json
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models import FinVerseAction
from credless_model.oracle import CreditOracle
from .environment import CreditAnalystEnvironment
from .models import FinVerseResponse
from .tasks import TASK_REGISTRY


oracle: CreditOracle | None = None
request_count = 0
total_latency_ms = 0.0
SESSION_DIR = Path(__file__).parent.parent / ".runtime" / "sessions"


def _init_runtime() -> CreditOracle:
    global oracle
    if oracle is None:
        oracle = CreditOracle()
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return oracle


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
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    episode_id: str
    action: Any


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_session_id(self, session_id: str) -> str:
        value = str(session_id).strip()
        if not value:
            raise HTTPException(status_code=400, detail="session_id required")
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in value):
            raise HTTPException(status_code=400, detail="session_id contains invalid characters")
        return value

    def _session_path(self, session_id: str) -> Path:
        return self.root / f"{self._validate_session_id(session_id)}.json"

    def _lock_path(self, session_id: str) -> Path:
        return self.root / f"{self._validate_session_id(session_id)}.lock"

    def _acquire_lock(self, session_id: str, timeout_s: float = 5.0) -> tuple[int, Path]:
        lock_path = self._lock_path(session_id)
        deadline = time.time() + timeout_s
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return fd, lock_path
            except FileExistsError:
                if time.time() >= deadline:
                    raise HTTPException(status_code=503, detail="session busy")
                time.sleep(0.05)

    def _release_lock(self, fd: int, lock_path: Path) -> None:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def load(self, session_id: str) -> CreditAnalystEnvironment:
        path = self._session_path(session_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="session not found")
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        env = CreditAnalystEnvironment()
        env.restore_snapshot(snapshot)
        return env

    def save(self, env: CreditAnalystEnvironment) -> None:
        path = self._session_path(env.state().session_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(env.snapshot(), indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def create(self, session_id: Optional[str] = None) -> CreditAnalystEnvironment:
        env = CreditAnalystEnvironment()
        env.set_session_id(session_id or str(uuid.uuid4()))
        return env


store = SessionStore(SESSION_DIR)


def _require_runtime() -> CreditOracle:
    return _init_runtime()


def _build_response(
    active_env: CreditAnalystEnvironment,
    observation_payload: dict[str, Any],
    *,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> FinVerseResponse:
    active_oracle = _require_runtime()
    observation = dict(observation_payload.get("observation", observation_payload))
    revealed = active_env.oracle_features()

    oracle_risk = active_oracle.predict_risk(revealed) if revealed else 0.0
    oracle_confidence = active_oracle.get_confidence(revealed) if revealed else 0.0
    explanation = str(observation.get("message") or info.get("explanation") or "")
    observation["oracle_risk"] = round(float(oracle_risk), 6)
    observation["oracle_confidence"] = round(float(oracle_confidence), 6)
    state = active_env.state()
    response_info = dict(info)
    response_info["session_id"] = state.session_id
    response_info["episode_id"] = state.episode_id

    return FinVerseResponse(
        observation=observation,
        reward=round(float(reward), 4),
        done=bool(done),
        explanation=explanation,
        oracle_risk=round(float(oracle_risk), 6),
        oracle_confidence=round(float(oracle_confidence), 6),
        session_id=state.session_id,
        episode_id=state.episode_id,
        info=response_info,
    )


def _invalid_step_response(active_env: CreditAnalystEnvironment, error: Exception) -> FinVerseResponse:
    fallback_observation = active_env._build_observation(  # type: ignore[attr-defined]
        step_reward=-0.1,
        done=False,
        message=f"invalid action: {str(error)}",
    )
    state = active_env.state()
    return FinVerseResponse(
        observation=fallback_observation,
        reward=-0.1,
        done=False,
        explanation=f"invalid action: {str(error)}",
        oracle_risk=0.0,
        oracle_confidence=0.0,
        session_id=state.session_id,
        episode_id=state.episode_id,
        info={"error": str(error), "session_id": state.session_id, "episode_id": state.episode_id},
    )


@app.get("/health")
def health():
    return {"status": "healthy", "oracle_loaded": oracle is not None, "session_store": str(SESSION_DIR)}


@app.get("/metrics")
def metrics():
    avg_latency = total_latency_ms / request_count if request_count else 0.0
    return {
        "requests": request_count,
        "avg_latency_ms": round(float(avg_latency), 3),
    }


@app.post("/reset", response_model=FinVerseResponse)
def reset(body: ResetRequest = ResetRequest()):
    _require_runtime()
    session_id = body.session_id or str(uuid.uuid4())
    fd, lock_path = store._acquire_lock(session_id)
    try:
        try:
            active_env = store.load(session_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            active_env = store.create(session_id)
        active_env.set_session_id(session_id)
        observation = active_env.reset(task_name=body.task_name, seed=body.seed)
        store.save(active_env)
    finally:
        store._release_lock(fd, lock_path)
    return _build_response(
        active_env,
        observation,
        reward=0.0,
        done=False,
        info={"explanation": "New applicant loaded. Begin investigation."},
    )


@app.post("/step", response_model=FinVerseResponse)
def step(body: StepRequest):
    _require_runtime()
    fd, lock_path = store._acquire_lock(body.session_id)
    try:
        active_env = store.load(body.session_id)
        current_episode_id = active_env.state().episode_id
        if body.episode_id != current_episode_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "episode_id mismatch",
                    "session_id": body.session_id,
                    "expected_episode_id": current_episode_id,
                    "received_episode_id": body.episode_id,
                },
            )
        result = active_env.step(body.action)
        store.save(active_env)
        return _build_response(
            active_env,
            result,
            reward=float(result.get("reward", 0.0)),
            done=bool(result.get("done", False)),
            info=dict(result.get("info", {})),
        )
    except HTTPException:
        raise
    except Exception as exc:
        return _invalid_step_response(active_env, exc)
    finally:
        store._release_lock(fd, lock_path)


@app.get("/state")
def state(session_id: str):
    _require_runtime()
    active_env = store.load(session_id)
    return active_env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_REGISTRY}


@app.get("/grader")
def grader_info(session_id: str):
    _require_runtime()
    active_env = store.load(session_id)
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
