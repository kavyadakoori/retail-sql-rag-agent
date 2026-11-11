from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/api/v1/health")
def health_api_v1():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    return {"status": "ready"}


@router.get("/api/v1/ready")
def ready_api_v1():
    return {"status": "ready"}
