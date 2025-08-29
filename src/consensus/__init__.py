"""
Byzantine consensus module for REV verification.

Provides fault-tolerant consensus mechanisms for distributed LLM verification.
"""

from .byzantine import (
    ByzantineValidator,
    ConsensusNetwork,
    ConsensusResult,
    Vote,
    VoteType
)

__all__ = [
    'ByzantineValidator',
    'ConsensusNetwork', 
    'ConsensusResult',
    'Vote',
    'VoteType'
]