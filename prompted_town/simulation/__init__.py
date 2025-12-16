"""
Simulation module: World rules, actions, and state transitions.

This module provides the deterministic simulation logic for Prompted Town.
Given a state and actions, it produces a new state.

Key components:
- Actions: What agents want to do
- Rules: How the world transitions
- Laws: Curfew, quota, and enforcement
- Social: Conversation outcome integration
"""

# Actions
from .actions import (
    Action,
    ActionValidation,
    validate_action,
    get_valid_actions,
    get_action_space_size,
    # Factories
    move_action,
    work_action,
    rest_action,
    deliver_quota_action,
    conversation_action,
    wait_action,
)

# Rules
from .rules import (
    StepResult,
    PendingConversation,
    step_world,
    get_agents_who_can_see,
    check_conversation_witnesses,
    # Constants
    ENERGY_COST_MOVE,
    ENERGY_COST_WORK,
    ENERGY_COST_CONVERSATION,
    ENERGY_RECOVERY_REST_HOME,
    ENERGY_RECOVERY_REST_OTHER,
    ENERGY_RECOVERY_WAIT,
    WORK_PRODUCTION_FARM,
)

# Laws
from .laws import (
    Violation,
    EnforcementResult,
    enforce_laws,
    check_curfew_violations,
    check_quota_compliance,
    check_arrest_threshold,
    check_interrogation_threshold,
    decay_suspicion,
    apply_arrest,
)

# Social
from .social import (
    DialogueTurn,
    ConversationOutcome,
    apply_conversation_outcome,
    can_recruit,
    estimate_recruitment_success,
)

__all__ = [
    # Actions
    "Action",
    "ActionValidation",
    "validate_action",
    "get_valid_actions",
    "get_action_space_size",
    "move_action",
    "work_action",
    "rest_action",
    "deliver_quota_action",
    "conversation_action",
    "wait_action",
    # Rules
    "StepResult",
    "PendingConversation",
    "step_world",
    "get_agents_who_can_see",
    "check_conversation_witnesses",
    "ENERGY_COST_MOVE",
    "ENERGY_COST_WORK",
    "ENERGY_COST_CONVERSATION",
    "ENERGY_RECOVERY_REST_HOME",
    "ENERGY_RECOVERY_REST_OTHER",
    "ENERGY_RECOVERY_WAIT",
    "WORK_PRODUCTION_FARM",
    # Laws
    "Violation",
    "EnforcementResult",
    "enforce_laws",
    "check_curfew_violations",
    "check_quota_compliance",
    "check_arrest_threshold",
    "check_interrogation_threshold",
    "decay_suspicion",
    "apply_arrest",
    # Social
    "DialogueTurn",
    "ConversationOutcome",
    "apply_conversation_outcome",
    "can_recruit",
    "estimate_recruitment_success",
]
