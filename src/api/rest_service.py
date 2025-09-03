"""
Production REST API and WebSocket Service for REV System
Provides FastAPI-based endpoints with real-time analysis capabilities
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, File, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# GraphQL support
from strawberry.fastapi import GraphQLRouter
import strawberry

# Import REV components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.error_handling import (
    REVException, handle_exception, error_context,
    degradation_manager, CircuitBreaker, retry_with_backoff
)
from src.utils.logging_config import get_logger, default_config
from src.utils.reproducibility import ReproducibilityManager

# Import core REV functionality
from run_rev import REVUnified

logger = get_logger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class ModelAnalysisRequest(BaseModel):
    """Request for model analysis"""
    model_path: str = Field(..., description="Model path or identifier")
    challenges: int = Field(default=10, description="Number of challenges")
    enable_prompt_orchestration: bool = Field(default=False)
    enable_principled_features: bool = Field(default=True)
    enable_unified_fingerprints: bool = Field(default=False)
    build_reference: bool = Field(default=False)
    options: Dict[str, Any] = Field(default_factory=dict)


class ModelAnalysisResponse(BaseModel):
    """Response from model analysis"""
    request_id: str
    status: str
    model_info: Dict[str, Any]
    fingerprint: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    processing_time: float
    errors: List[str] = Field(default_factory=list)


class FingerprintComparisonRequest(BaseModel):
    """Request for fingerprint comparison"""
    fingerprint1: Dict[str, Any]
    fingerprint2: Dict[str, Any]
    method: str = Field(default="hamming", description="Comparison method")


class FingerprintComparisonResponse(BaseModel):
    """Response from fingerprint comparison"""
    similarity_score: float
    distance: float
    method: str
    details: Dict[str, Any]


class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime: float
    memory_usage_mb: float
    active_analyses: int
    error_rate: float


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str  # analysis_update, error, complete
    data: Dict[str, Any]
    timestamp: str


# ============================================================================
# GraphQL Schema
# ============================================================================

@strawberry.type
class ModelInfo:
    """GraphQL type for model information"""
    name: str
    family: Optional[str]
    size: Optional[str]
    architecture: Optional[str]


@strawberry.type
class AnalysisResult:
    """GraphQL type for analysis results"""
    request_id: str
    model_info: ModelInfo
    confidence: float
    processing_time: float
    fingerprint_hash: Optional[str]


@strawberry.type
class Query:
    """GraphQL queries"""
    
    @strawberry.field
    async def analysis_status(self, request_id: str) -> Optional[AnalysisResult]:
        """Get analysis status by request ID"""
        if request_id in app.state.analysis_cache:
            result = app.state.analysis_cache[request_id]
            return AnalysisResult(
                request_id=request_id,
                model_info=ModelInfo(
                    name=result.get("model_name", ""),
                    family=result.get("family"),
                    size=result.get("size"),
                    architecture=result.get("architecture")
                ),
                confidence=result.get("confidence", 0.0),
                processing_time=result.get("processing_time", 0.0),
                fingerprint_hash=result.get("fingerprint_hash")
            )
        return None
    
    @strawberry.field
    async def list_analyses(self, limit: int = 10) -> List[AnalysisResult]:
        """List recent analyses"""
        results = []
        for request_id, data in list(app.state.analysis_cache.items())[:limit]:
            results.append(AnalysisResult(
                request_id=request_id,
                model_info=ModelInfo(
                    name=data.get("model_name", ""),
                    family=data.get("family"),
                    size=data.get("size"),
                    architecture=data.get("architecture")
                ),
                confidence=data.get("confidence", 0.0),
                processing_time=data.get("processing_time", 0.0),
                fingerprint_hash=data.get("fingerprint_hash")
            ))
        return results


@strawberry.type
class Mutation:
    """GraphQL mutations"""
    
    @strawberry.mutation
    async def start_analysis(
        self,
        model_path: str,
        challenges: int = 10
    ) -> AnalysisResult:
        """Start a new analysis"""
        request_id = str(uuid.uuid4())
        
        # Start analysis in background
        asyncio.create_task(
            run_analysis_async(request_id, model_path, challenges)
        )
        
        return AnalysisResult(
            request_id=request_id,
            model_info=ModelInfo(
                name=model_path,
                family=None,
                size=None,
                architecture=None
            ),
            confidence=0.0,
            processing_time=0.0,
            fingerprint_hash=None
        )


# ============================================================================
# Middleware
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with structured logging"""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "Request received",
            fields={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None
            }
        )
        
        # Track timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            fields={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_seconds": duration
            }
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(duration)
        
        # Update metrics
        if default_config.metrics_collector:
            default_config.metrics_collector.increment_counter(
                default_config.metrics_collector.request_count,
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            )
            default_config.metrics_collector.request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
        
        return response


# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting REV API service")
    
    # Initialize application state
    app.state.rev_pipeline = None
    app.state.analysis_cache = {}
    app.state.websocket_connections = []
    app.state.start_time = time.time()
    app.state.circuit_breaker = CircuitBreaker()
    
    # Initialize REV pipeline
    try:
        app.state.rev_pipeline = REVUnified(
            debug=False,
            enable_behavioral_analysis=True,
            enable_principled_features=True
        )
        logger.info("REV pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize REV pipeline: {e}")
        degradation_manager.degrade_feature("rev_pipeline", e)
    
    # Register graceful degradation features
    degradation_manager.register_feature("fingerprinting", default_enabled=True)
    degradation_manager.register_feature("prompt_orchestration", default_enabled=True)
    degradation_manager.register_feature("websocket", default_enabled=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down REV API service")
    
    # Close WebSocket connections
    for ws in app.state.websocket_connections:
        await ws.close()
    
    # Cleanup REV pipeline
    if app.state.rev_pipeline:
        app.state.rev_pipeline.cleanup()


# Create FastAPI application
app = FastAPI(
    title="REV System API",
    description="Restriction Enzyme Verification System - Production API",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add GraphQL endpoint
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "REV System API",
        "version": "3.0.0",
        "documentation": "/docs",
        "graphql": "/graphql"
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app.state.start_time
    
    # Get memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # Get error statistics
    error_stats = error_context.get_error_stats()
    
    return HealthStatus(
        status="healthy" if app.state.rev_pipeline else "degraded",
        version="3.0.0",
        uptime=uptime,
        memory_usage_mb=memory_mb,
        active_analyses=len(app.state.analysis_cache),
        error_rate=error_stats.get("error_rate", 0.0)
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if app.state.rev_pipeline:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if default_config.metrics_collector:
        metrics = default_config.metrics_collector.get_metrics()
        return StreamingResponse(
            iter([metrics]),
            media_type="text/plain; version=0.0.4"
        )
    return {"error": "Metrics not available"}


@app.post("/analyze", response_model=ModelAnalysisResponse)
async def analyze_model(
    request: ModelAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze a model"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Check if pipeline is available
        if not app.state.rev_pipeline:
            raise REVException("REV pipeline not initialized", error_code="PIPELINE_ERROR")
        
        # Run analysis
        logger.info(f"Starting analysis for {request.model_path}")
        
        # Process model
        result = await asyncio.to_thread(
            app.state.rev_pipeline.process_model,
            request.model_path,
            challenges=request.challenges,
            use_local=False
        )
        
        # Extract relevant information
        model_info = {
            "model_name": result.get("model_name"),
            "family": result.get("identification", {}).get("family"),
            "confidence": result.get("identification", {}).get("confidence", 0.0)
        }
        
        fingerprint = result.get("stages", {}).get("behavioral_analysis", {}).get("metrics", {})
        metrics = result.get("stages", {}).get("processing", {})
        
        # Cache result
        app.state.analysis_cache[request_id] = {
            **model_info,
            "fingerprint": fingerprint,
            "metrics": metrics,
            "processing_time": time.time() - start_time
        }
        
        return ModelAnalysisResponse(
            request_id=request_id,
            status="completed",
            model_info=model_info,
            fingerprint=fingerprint,
            metrics=metrics,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Analysis failed: {error_msg}")
        error_context.record_error(e, {"request_id": request_id})
        
        return ModelAnalysisResponse(
            request_id=request_id,
            status="failed",
            model_info={},
            fingerprint=None,
            metrics=None,
            processing_time=time.time() - start_time,
            errors=[error_msg]
        )


@app.post("/compare", response_model=FingerprintComparisonResponse)
async def compare_fingerprints(request: FingerprintComparisonRequest):
    """Compare two fingerprints"""
    try:
        # Extract hypervectors
        hv1 = request.fingerprint1.get("hypervector", [])
        hv2 = request.fingerprint2.get("hypervector", [])
        
        if not hv1 or not hv2:
            raise ValidationError("Missing hypervector data")
        
        # Calculate similarity
        import numpy as np
        from src.hypervector.hamming import HammingDistanceOptimized
        
        hamming = HammingDistanceOptimized()
        distance = hamming.distance(np.array(hv1), np.array(hv2))
        similarity = 1.0 - (distance / len(hv1))
        
        return FingerprintComparisonResponse(
            similarity_score=similarity,
            distance=distance,
            method=request.method,
            details={
                "vector_dimension": len(hv1),
                "matching_bits": len(hv1) - distance
            }
        )
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file for analysis"""
    try:
        # Save uploaded file
        upload_dir = Path("/tmp/rev_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoints
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {len(self.active_connections)} active")
        
    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {len(self.active_connections)} active")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        """Broadcast message to all clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/analysis")
async def websocket_analysis(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis"""
    
    if not degradation_manager.features.get("websocket", {}).is_enabled():
        await websocket.close(code=1000, reason="WebSocket feature disabled")
        return
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "start_analysis":
                # Start analysis
                model_path = data.get("model_path")
                request_id = str(uuid.uuid4())
                
                # Send initial response
                await websocket.send_json({
                    "type": "analysis_started",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Run analysis with progress updates
                async def run_with_updates():
                    try:
                        # Simulate progress updates
                        for progress in [0.2, 0.4, 0.6, 0.8]:
                            await asyncio.sleep(1)
                            await websocket.send_json({
                                "type": "analysis_update",
                                "request_id": request_id,
                                "progress": progress,
                                "message": f"Processing... {int(progress*100)}%",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        
                        # Complete
                        await websocket.send_json({
                            "type": "analysis_complete",
                            "request_id": request_id,
                            "result": {"status": "success"},
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "request_id": request_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                asyncio.create_task(run_with_updates())
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        

# ============================================================================
# Background Tasks
# ============================================================================

async def run_analysis_async(request_id: str, model_path: str, challenges: int):
    """Run analysis in background"""
    try:
        logger.info(f"Background analysis started: {request_id}")
        
        if app.state.rev_pipeline:
            result = await asyncio.to_thread(
                app.state.rev_pipeline.process_model,
                model_path,
                challenges=challenges
            )
            
            # Store result
            app.state.analysis_cache[request_id] = result
            
            # Broadcast completion
            await manager.broadcast(json.dumps({
                "type": "analysis_complete",
                "request_id": request_id
            }))
            
    except Exception as e:
        logger.error(f"Background analysis failed: {e}")
        error_context.record_error(e, {"request_id": request_id})


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(REVException)
async def rev_exception_handler(request: Request, exc: REVException):
    """Handle REV-specific exceptions"""
    error_context.record_error(exc, {"path": request.url.path})
    
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    error_context.record_error(exc, {"path": request.url.path})
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "rest_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )