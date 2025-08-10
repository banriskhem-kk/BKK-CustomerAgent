from agents.OrchestratorAgent import CoreSystemOrchestrator
import logging
from agents.dataclass.dataclass import ChatRequest, ChatAgentResponse
from fastapi import FastAPI
from contextlib import asynccontextmanager
import sys, os
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")
    app.state.orchestrator = CoreSystemOrchestrator()
    yield
    await app.state.orchestrator.shutdown()
    logger.info("Shutting down application")


app.router.lifespan_context = lifespan


@app.post("/chat", response_model=ChatAgentResponse)
async def chat(request: ChatRequest):
    logger.info(f"Received chat request: {request}")
    result = await app.state.orchestrator.process_request(
        request.session_id, request.message
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
