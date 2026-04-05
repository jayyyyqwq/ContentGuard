# app.py
from openenv.core.env_server import create_app
from models import ContentGuardAction, ContentGuardObservation
from .environment import ContentGuardEnvironment

app = create_app(
    ContentGuardEnvironment,
    ContentGuardAction,
    ContentGuardObservation,
    env_name="contentguard",
    max_concurrent_envs=64,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
