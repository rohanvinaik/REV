"""
REST API server for fingerprint attestation service.
Provides TEE integration and signed attestation reports.
"""

import hashlib
import json
import time
import uuid
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import secrets

from flask import Flask, request, jsonify, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest, Unauthorized, TooManyRequests
import jwt

from src.security.zk_attestation import ZKAttestationSystem, ZKProof
from src.security.rate_limiter import HierarchicalRateLimiter, RateLimitConfig
from src.crypto.merkle_tree import SparseMerkleTree, MerkleTree, HSMIntegratedMerkleTree

logger = logging.getLogger(__name__)


@dataclass
class AttestationReport:
    """Attestation report for fingerprint verification."""
    
    report_id: str
    fingerprint_hash: bytes
    timestamp: str
    model_id: str
    attestation_type: str  # "distance", "membership", "range"
    proof_data: Dict[str, Any]
    signature: bytes
    tee_quote: Optional[bytes] = None
    metadata: Dict[str, Any] = None


@dataclass
class TEEConfig:
    """Configuration for Trusted Execution Environment."""
    
    enabled: bool = False
    provider: str = "sgx"  # "sgx", "sev", "trustzone"
    attestation_service_url: Optional[str] = None
    enclave_info: Dict[str, Any] = None


class AttestationServer:
    """
    REST API server for REV attestation service.
    """
    
    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        tee_config: Optional[TEEConfig] = None,
        enable_hsm: bool = False,
        secret_key: Optional[str] = None
    ):
        """
        Initialize attestation server.
        
        Args:
            port: Server port
            host: Server host
            tee_config: TEE configuration
            enable_hsm: Enable HSM for signing
            secret_key: Secret key for JWT
        """
        self.port = port
        self.host = host
        self.app = Flask(__name__)
        
        # Security configuration
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.app.config['SECRET_KEY'] = self.secret_key
        
        # TEE configuration
        self.tee_config = tee_config or TEEConfig()
        self.tee_client = self._init_tee() if self.tee_config.enabled else None
        
        # Initialize components
        self.zk_system = ZKAttestationSystem()
        self.rate_limiter = self._init_rate_limiter()
        self.merkle_tree = self._init_merkle_tree(enable_hsm)
        self.sparse_tree = SparseMerkleTree()
        
        # Storage
        self.attestation_reports: Dict[str, AttestationReport] = {}
        self.fingerprint_registry: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup Flask-Limiter
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["100 per minute", "1000 per hour"]
        )
        
        logger.info(f"Attestation server initialized on {host}:{port}")
    
    def _init_tee(self):
        """Initialize TEE client."""
        if self.tee_config.provider == "sgx":
            return SGXClient(self.tee_config)
        elif self.tee_config.provider == "sev":
            return SEVClient(self.tee_config)
        else:
            logger.warning(f"Unknown TEE provider: {self.tee_config.provider}")
            return None
    
    def _init_rate_limiter(self) -> HierarchicalRateLimiter:
        """Initialize rate limiter."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=20,
            enable_distributed=False  # Can be enabled with Redis
        )
        return HierarchicalRateLimiter(config)
    
    def _init_merkle_tree(self, enable_hsm: bool):
        """Initialize Merkle tree."""
        if enable_hsm:
            return HSMIntegratedMerkleTree(
                hsm_config={"type": "softhsm"}  # For testing
            )
        else:
            return MerkleTree()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "tee_enabled": self.tee_config.enabled
            })
        
        @self.app.route('/attest/fingerprint', methods=['POST'])
        @self.limiter.limit("10 per minute")
        def attest_fingerprint():
            """Create attestation for fingerprint."""
            try:
                data = request.get_json()
                
                # Validate request
                if not data or 'fingerprint' not in data:
                    raise BadRequest("Missing fingerprint data")
                
                # Check rate limit
                client_id = request.headers.get('X-Client-ID', get_remote_address())
                rate_result = self.rate_limiter.check_rate_limit(
                    user_id=client_id,
                    operation="attestation"
                )
                
                if not rate_result.allowed:
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "retry_after": rate_result.retry_after
                    }), 429
                
                # Create attestation
                report = self._create_attestation(
                    fingerprint=data['fingerprint'],
                    model_id=data.get('model_id', 'unknown'),
                    attestation_type=data.get('type', 'membership')
                )
                
                # Store report
                self.attestation_reports[report.report_id] = report
                
                # Audit log
                self._audit_log("attestation_created", {
                    "report_id": report.report_id,
                    "client_id": client_id,
                    "model_id": report.model_id
                })
                
                return jsonify({
                    "report_id": report.report_id,
                    "timestamp": report.timestamp,
                    "signature": report.signature.hex() if report.signature else None,
                    "tee_quote": report.tee_quote.hex() if report.tee_quote else None
                })
                
            except Exception as e:
                logger.error(f"Attestation failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/verify/attestation/<report_id>', methods=['GET'])
        def verify_attestation(report_id: str):
            """Verify an attestation report."""
            try:
                if report_id not in self.attestation_reports:
                    return jsonify({"error": "Report not found"}), 404
                
                report = self.attestation_reports[report_id]
                
                # Verify signature
                is_valid = self._verify_report_signature(report)
                
                # Verify TEE quote if present
                tee_valid = True
                if report.tee_quote and self.tee_client:
                    tee_valid = self.tee_client.verify_quote(report.tee_quote)
                
                return jsonify({
                    "report_id": report_id,
                    "valid": is_valid and tee_valid,
                    "timestamp": report.timestamp,
                    "attestation_type": report.attestation_type,
                    "metadata": report.metadata
                })
                
            except Exception as e:
                logger.error(f"Verification failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/prove/distance', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def prove_distance():
            """Create zero-knowledge proof of distance."""
            try:
                data = request.get_json()
                
                # Create distance proof
                proof = self.zk_system.prove_distance_computation(
                    fingerprint1=bytes.fromhex(data['fingerprint1']),
                    fingerprint2=bytes.fromhex(data['fingerprint2']),
                    distance=data['distance']
                )
                
                return jsonify({
                    "proof": {
                        "type": proof.proof_type,
                        "commitment": proof.commitment.hex(),
                        "challenge": proof.challenge.hex(),
                        "response": proof.response.hex(),
                        "public_inputs": proof.public_inputs
                    }
                })
                
            except Exception as e:
                logger.error(f"Distance proof failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/prove/range', methods=['POST'])
        @self.limiter.limit("5 per minute")
        def prove_range():
            """Create range proof for similarity score."""
            try:
                data = request.get_json()
                
                # Create range proof
                proof = self.zk_system.create_range_proof(
                    value=data['value'],
                    range_min=data.get('min', 0.0),
                    range_max=data.get('max', 1.0)
                )
                
                return jsonify({
                    "proof": {
                        "commitment": proof.commitment.hex(),
                        "proof_data": proof.proof_data.decode('utf-8'),
                        "range": [proof.range_min, proof.range_max]
                    }
                })
                
            except Exception as e:
                logger.error(f"Range proof failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/register/fingerprint', methods=['POST'])
        @self.limiter.limit("10 per hour")
        def register_fingerprint():
            """Register a fingerprint in the system."""
            try:
                data = request.get_json()
                auth_token = request.headers.get('Authorization')
                
                # Verify authorization
                if not self._verify_auth(auth_token):
                    raise Unauthorized("Invalid authorization")
                
                # Register fingerprint
                fingerprint_id = self._register_fingerprint(
                    fingerprint=data['fingerprint'],
                    model_id=data['model_id'],
                    metadata=data.get('metadata', {})
                )
                
                return jsonify({
                    "fingerprint_id": fingerprint_id,
                    "registered": True,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/audit/log', methods=['GET'])
        def get_audit_log():
            """Get audit log (requires admin auth)."""
            try:
                auth_token = request.headers.get('Authorization')
                
                if not self._verify_admin_auth(auth_token):
                    raise Unauthorized("Admin authorization required")
                
                # Return recent audit entries
                limit = min(int(request.args.get('limit', 100)), 1000)
                
                return jsonify({
                    "entries": self.audit_log[-limit:],
                    "total": len(self.audit_log)
                })
                
            except Exception as e:
                logger.error(f"Audit log retrieval failed: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _create_attestation(
        self,
        fingerprint: Any,
        model_id: str,
        attestation_type: str
    ) -> AttestationReport:
        """
        Create attestation report for fingerprint.
        
        Args:
            fingerprint: Fingerprint data
            model_id: Model identifier
            attestation_type: Type of attestation
            
        Returns:
            AttestationReport
        """
        # Generate report ID
        report_id = str(uuid.uuid4())
        
        # Hash fingerprint
        if isinstance(fingerprint, str):
            fingerprint_bytes = bytes.fromhex(fingerprint)
        else:
            fingerprint_bytes = json.dumps(fingerprint).encode()
        
        fingerprint_hash = hashlib.sha256(fingerprint_bytes).digest()
        
        # Create proof based on type
        proof_data = {}
        
        if attestation_type == "membership":
            # Add to Merkle tree and get proof
            self.merkle_tree.build([fingerprint_hash])
            proof = self.merkle_tree.get_proof(0)
            proof_data = {
                "merkle_root": self.merkle_tree.root.hash.hex() if self.merkle_tree.root else "",
                "proof": {
                    "leaf_index": proof.leaf_index if proof else 0,
                    "siblings": [(s[0].hex(), s[1]) for s in proof.siblings] if proof else []
                }
            }
        
        elif attestation_type == "distance":
            # Create ZK distance proof
            proof_data = {"type": "distance", "status": "pending"}
        
        # Get TEE quote if available
        tee_quote = None
        if self.tee_client:
            tee_quote = self.tee_client.generate_quote(fingerprint_hash)
        
        # Sign report
        report_data = {
            "report_id": report_id,
            "fingerprint_hash": fingerprint_hash.hex(),
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "attestation_type": attestation_type,
            "proof_data": proof_data
        }
        
        signature = self._sign_report(report_data)
        
        return AttestationReport(
            report_id=report_id,
            fingerprint_hash=fingerprint_hash,
            timestamp=report_data["timestamp"],
            model_id=model_id,
            attestation_type=attestation_type,
            proof_data=proof_data,
            signature=signature,
            tee_quote=tee_quote,
            metadata={"version": "1.0"}
        )
    
    def _sign_report(self, report_data: Dict[str, Any]) -> bytes:
        """Sign attestation report."""
        # Serialize report data
        serialized = json.dumps(report_data, sort_keys=True)
        
        # Sign with HMAC (in production, use proper signing key)
        import hmac
        signature = hmac.new(
            self.secret_key.encode(),
            serialized.encode(),
            hashlib.sha256
        ).digest()
        
        return signature
    
    def _verify_report_signature(self, report: AttestationReport) -> bool:
        """Verify report signature."""
        # Reconstruct report data
        report_data = {
            "report_id": report.report_id,
            "fingerprint_hash": report.fingerprint_hash.hex(),
            "timestamp": report.timestamp,
            "model_id": report.model_id,
            "attestation_type": report.attestation_type,
            "proof_data": report.proof_data
        }
        
        # Verify signature
        expected_signature = self._sign_report(report_data)
        
        import hmac
        return hmac.compare_digest(report.signature, expected_signature)
    
    def _register_fingerprint(
        self,
        fingerprint: Any,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Register fingerprint in the system."""
        # Generate fingerprint ID
        fingerprint_id = str(uuid.uuid4())
        
        # Hash fingerprint
        if isinstance(fingerprint, str):
            fingerprint_bytes = bytes.fromhex(fingerprint)
        else:
            fingerprint_bytes = json.dumps(fingerprint).encode()
        
        fingerprint_hash = hashlib.sha256(fingerprint_bytes).digest()
        
        # Store in registry
        self.fingerprint_registry[fingerprint_id] = {
            "hash": fingerprint_hash.hex(),
            "model_id": model_id,
            "metadata": metadata,
            "registered_at": datetime.utcnow().isoformat()
        }
        
        # Add to sparse Merkle tree
        self.sparse_tree.update(fingerprint_hash, fingerprint_bytes)
        
        return fingerprint_id
    
    def _verify_auth(self, auth_token: Optional[str]) -> bool:
        """Verify authorization token."""
        if not auth_token:
            return False
        
        try:
            # Extract bearer token
            if auth_token.startswith("Bearer "):
                token = auth_token[7:]
            else:
                token = auth_token
            
            # Verify JWT
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            
            # Check expiration
            if "exp" in payload:
                if datetime.fromtimestamp(payload["exp"]) < datetime.utcnow():
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Auth verification failed: {e}")
            return False
    
    def _verify_admin_auth(self, auth_token: Optional[str]) -> bool:
        """Verify admin authorization."""
        if not self._verify_auth(auth_token):
            return False
        
        try:
            token = auth_token[7:] if auth_token.startswith("Bearer ") else auth_token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check admin role
            return payload.get("role") == "admin"
            
        except Exception:
            return False
    
    def _audit_log(self, event_type: str, details: Dict[str, Any]):
        """Add entry to audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.audit_log.append(entry)
        
        # Trim log if too large
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def generate_auth_token(
        self,
        user_id: str,
        role: str = "user",
        expires_in: int = 3600
    ) -> str:
        """
        Generate JWT authentication token.
        
        Args:
            user_id: User identifier
            role: User role
            expires_in: Token expiry in seconds
            
        Returns:
            JWT token
        """
        payload = {
            "user_id": user_id,
            "role": role,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def run(self, debug: bool = False):
        """Run the attestation server."""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=True
        )


class SGXClient:
    """Mock Intel SGX client for TEE attestation."""
    
    def __init__(self, config: TEEConfig):
        self.config = config
        logger.info("Initialized SGX client (mock)")
    
    def generate_quote(self, data: bytes) -> bytes:
        """Generate SGX quote."""
        # Mock implementation
        quote = hashlib.sha256(b"SGX_QUOTE:" + data).digest()
        return quote
    
    def verify_quote(self, quote: bytes) -> bool:
        """Verify SGX quote."""
        # Mock implementation - always valid in testing
        return True


class SEVClient:
    """Mock AMD SEV client for TEE attestation."""
    
    def __init__(self, config: TEEConfig):
        self.config = config
        logger.info("Initialized SEV client (mock)")
    
    def generate_quote(self, data: bytes) -> bytes:
        """Generate SEV attestation report."""
        # Mock implementation
        report = hashlib.sha256(b"SEV_REPORT:" + data).digest()
        return report
    
    def verify_quote(self, report: bytes) -> bool:
        """Verify SEV attestation report."""
        # Mock implementation
        return True


def create_attestation_server(
    config: Optional[Dict[str, Any]] = None
) -> AttestationServer:
    """
    Factory function to create attestation server.
    
    Args:
        config: Server configuration
        
    Returns:
        AttestationServer instance
    """
    config = config or {}
    
    tee_config = None
    if config.get("enable_tee"):
        tee_config = TEEConfig(
            enabled=True,
            provider=config.get("tee_provider", "sgx")
        )
    
    return AttestationServer(
        port=config.get("port", 8080),
        host=config.get("host", "0.0.0.0"),
        tee_config=tee_config,
        enable_hsm=config.get("enable_hsm", False),
        secret_key=config.get("secret_key")
    )