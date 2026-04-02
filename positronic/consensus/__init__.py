"""
Positronic - Consensus Module
Delegated Proof of Stake (DPoS) with BFT finality and Neural Validator Nodes.

Components:
    - validator:  Validator data model and registry
    - dpos:       Core DPoS engine with proposer selection
    - slot:       Slot and epoch timing (3s slots, 32 per epoch)
    - staking:    Staking and delegation management
    - finality:   BFT finality (2/3 supermajority attestation)
    - slashing:   Slashing conditions for validator misbehavior
    - election:   Validator election per epoch (top-21 by stake)
    - censorship_detector: Phase 15 validator TX inclusion monitoring
    - ai_judge:  Phase 4 AI-in-Consensus judge verification
"""
