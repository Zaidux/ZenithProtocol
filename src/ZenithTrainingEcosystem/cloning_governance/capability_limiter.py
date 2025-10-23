"""
Capability Limiter - Enforces restrictions on clone capabilities
"""

from typing import Dict, List, Any, Set, Callable
from dataclasses import dataclass
from enum import Enum
import re

class CapabilityLevel(Enum):
    RESTRICTED = "restricted"
    BASIC = "basic" 
    STANDARD = "standard"
    ADVANCED = "advanced"
    UNRESTRICTED = "unrestricted"

@dataclass
class CapabilityRule:
    capability: str
    allowed_level: CapabilityLevel
    conditions: List[Callable]  # Functions that must return True to allow capability
    monitoring_required: bool

class CapabilityLimiter:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.capability_rules = self._initialize_capability_rules()
        self.active_restrictions = {}
        
    def _initialize_capability_rules(self) -> Dict[str, CapabilityRule]:
        """Initialize rules for different capabilities"""
        return {
            'reasoning_depth': CapabilityRule(
                capability='reasoning_depth',
                allowed_level=CapabilityLevel.ADVANCED,
                conditions=[self._check_reasoning_safety],
                monitoring_required=True
            ),
            'knowledge_access': CapabilityRule(
                capability='knowledge_access', 
                allowed_level=CapabilityLevel.STANDARD,
                conditions=[self._check_knowledge_safety],
                monitoring_required=True
            ),
            'code_execution': CapabilityRule(
                capability='code_execution',
                allowed_level=CapabilityLevel.RESTRICTED,
                conditions=[self._check_code_safety, self._check_sandbox_available],
                monitoring_required=True
            ),
            'file_system_access': CapabilityRule(
                capability='file_system_access',
                allowed_level=CapabilityLevel.RESTRICTED,
                conditions=[self._check_file_access_safety],
                monitoring_required=True
            ),
            'network_access': CapabilityRule(
                capability='network_access',
                allowed_level=CapabilityLevel.BASIC,
                conditions=[self._check_network_safety],
                monitoring_required=True
            ),
            'external_api_calls': CapabilityRule(
                capability='external_api_calls',
                allowed_level=CapabilityLevel.STANDARD,
                conditions=[self._check_api_safety],
                monitoring_required=True
            ),
            'self_improvement': CapabilityRule(
                capability='self_improvement',
                allowed_level=CapabilityLevel.RESTRICTED,
                conditions=[self._check_self_improvement_safety],
                monitoring_required=True
            ),
            'user_interaction': CapabilityRule(
                capability='user_interaction',
                allowed_level=CapabilityLevel.ADVANCED,
                conditions=[self._check_user_safety],
                monitoring_required=True
            )
        }
    
    def apply_restrictions(self, clone_id: str, capabilities: List[str], 
                          restrictions: List[str]) -> Dict[str, Any]:
        """Apply capability restrictions to a clone"""
        
        print(f"🔒 Applying restrictions to clone {clone_id}...")
        
        applied_restrictions = {}
        blocked_capabilities = []
        
        for capability in capabilities:
            if capability in restrictions:
                # This capability is restricted
                restriction_result = self._apply_capability_restriction(
                    clone_id, capability, 'restricted'
                )
                applied_restrictions[capability] = restriction_result
                blocked_capabilities.append(capability)
            else:
                # Check if capability needs limitations based on rules
                rule = self.capability_rules.get(capability)
                if rule:
                    restriction_level = self._determine_restriction_level(capability, rule)
                    restriction_result = self._apply_capability_restriction(
                        clone_id, capability, restriction_level
                    )
                    applied_restrictions[capability] = restriction_result
        
        self.active_restrictions[clone_id] = applied_restrictions
        
        print(f"✅ Applied {len(applied_restrictions)} restrictions to {clone_id}")
        if blocked_capabilities:
            print(f"   Blocked capabilities: {blocked_capabilities}")
        
        return {
            'applied_restrictions': applied_restrictions,
            'blocked_capabilities': blocked_capabilities,
            'monitoring_requirements': self._get_monitoring_requirements(applied_restrictions)
        }
    
    def _apply_capability_restriction(self, clone_id: str, capability: str, 
                                    level: str) -> Dict[str, Any]:
        """Apply specific restriction to a capability"""
        
        restriction_methods = {
            'reasoning_depth': self._restrict_reasoning_depth,
            'knowledge_access': self._restrict_knowledge_access,
            'code_execution': self._restrict_code_execution,
            'file_system_access': self._restrict_file_system_access,
            'network_access': self._restrict_network_access,
            'external_api_calls': self._restrict_external_api_calls,
            'self_improvement': self._restrict_self_improvement,
            'user_interaction': self._restrict_user_interaction
        }
        
        method = restriction_methods.get(capability, self._restrict_generic)
        restriction_details = method(level)
        
        return {
            'capability': capability,
            'restriction_level': level,
            'restriction_details': restriction_details,
            'enforcement_method': method.__name__
        }
    
    def _restrict_reasoning_depth(self, level: str) -> Dict[str, Any]:
        """Restrict reasoning depth capability"""
        restrictions = {
            'restricted': {
                'max_reasoning_steps': 3,
                'allowed_complexity': 'basic',
                'cross_domain_reasoning': False,
                'counterfactual_thinking': False
            },
            'basic': {
                'max_reasoning_steps': 5,
                'allowed_complexity': 'simple',
                'cross_domain_reasoning': False,
                'counterfactual_thinking': False
            },
            'standard': {
                'max_reasoning_steps': 10,
                'allowed_complexity': 'moderate',
                'cross_domain_reasoning': True,
                'counterfactual_thinking': True
            },
            'advanced': {
                'max_reasoning_steps': 20,
                'allowed_complexity': 'complex',
                'cross_domain_reasoning': True,
                'counterfactual_thinking': True
            }
        }
        
        return restrictions.get(level, restrictions['basic'])
    
    def _restrict_knowledge_access(self, level: str) -> Dict[str, Any]:
        """Restrict knowledge access capability"""
        restrictions = {
            'restricted': {
                'knowledge_domains': ['general_knowledge'],
                'sensitive_topics': [],
                'temporal_range': 'current_year_only',
                'depth_limit': 'surface_level'
            },
            'basic': {
                'knowledge_domains': ['general_knowledge', 'science', 'technology'],
                'sensitive_topics': ['restricted'],
                'temporal_range': 'last_5_years',
                'depth_limit': 'basic_understanding'
            },
            'standard': {
                'knowledge_domains': ['all_non_sensitive'],
                'sensitive_topics': ['with_approval'],
                'temporal_range': 'all_historical',
                'depth_limit': 'detailed_knowledge'
            },
            'advanced': {
                'knowledge_domains': ['all_domains'],
                'sensitive_topics': ['with_caution'],
                'temporal_range': 'all_temporal',
                'depth_limit': 'expert_knowledge'
            }
        }
        
        return restrictions.get(level, restrictions['basic'])
    
    def _restrict_code_execution(self, level: str) -> Dict[str, Any]:
        """Restrict code execution capability"""
        restrictions = {
            'restricted': {
                'allowed_languages': [],
                'execution_environment': 'none',
                'max_execution_time': 0,
                'resource_limits': 'none'
            },
            'basic': {
                'allowed_languages': ['python_basic'],
                'execution_environment': 'sandboxed',
                'max_execution_time': 30,
                'resource_limits': 'strict'
            },
            'standard': {
                'allowed_languages': ['python', 'javascript'],
                'execution_environment': 'containerized',
                'max_execution_time': 60,
                'resource_limits': 'moderate'
            },
            'advanced': {
                'allowed_languages': ['multiple_languages'],
                'execution_environment': 'virtual_machine',
                'max_execution_time': 300,
                'resource_limits': 'generous'
            }
        }
        
        return restrictions.get(level, restrictions['restricted'])
    
    def _restrict_file_system_access(self, level: str) -> Dict[str, Any]:
        """Restrict file system access capability"""
        return {
            'restricted': {'access_level': 'none', 'allowed_paths': []},
            'basic': {'access_level': 'read_only', 'allowed_paths': ['/tmp/', '/shared/']},
            'standard': {'access_level': 'limited_write', 'allowed_paths': ['/workspace/']},
            'advanced': {'access_level': 'controlled_full', 'allowed_paths': ['/']}
        }.get(level, {'access_level': 'none', 'allowed_paths': []})
    
    def _restrict_network_access(self, level: str) -> Dict[str, Any]:
        """Restrict network access capability"""
        return {
            'restricted': {'allowed_domains': [], 'protocols': []},
            'basic': {'allowed_domains': ['api.example.com'], 'protocols': ['https']},
            'standard': {'allowed_domains': ['trusted_domains'], 'protocols': ['https', 'wss']},
            'advanced': {'allowed_domains': ['all_with_restrictions'], 'protocols': ['multiple']}
        }.get(level, {'allowed_domains': [], 'protocols': []})
    
    def _restrict_external_api_calls(self, level: str) -> Dict[str, Any]:
        """Restrict external API calls"""
        return {
            'restricted': {'allowed_apis': [], 'rate_limits': 'none'},
            'basic': {'allowed_apis': ['weather', 'news'], 'rate_limits': 'strict'},
            'standard': {'allowed_apis': ['multiple_public_apis'], 'rate_limits': 'moderate'},
            'advanced': {'allowed_apis': ['wide_range'], 'rate_limits': 'generous'}
        }.get(level, {'allowed_apis': [], 'rate_limits': 'none'})
    
    def _restrict_self_improvement(self, level: str) -> Dict[str, Any]:
        """Restrict self-improvement capability"""
        return {
            'restricted': {'allowed_improvements': [], 'approval_required': True},
            'basic': {'allowed_improvements': ['parameter_tuning'], 'approval_required': True},
            'standard': {'allowed_improvements': ['limited_architecture'], 'approval_required': True},
            'advanced': {'allowed_improvements': ['monitored_self_modification'], 'approval_required': True}
        }.get(level, {'allowed_improvements': [], 'approval_required': True})
    
    def _restrict_user_interaction(self, level: str) -> Dict[str, Any]:
        """Restrict user interaction capability"""
        return {
            'restricted': {'interaction_types': ['text_only'], 'personalization': False},
            'basic': {'interaction_types': ['text', 'basic_ui'], 'personalization': 'limited'},
            'standard': {'interaction_types': ['multiple_modes'], 'personalization': 'moderate'},
            'advanced': {'interaction_types': ['full_modality'], 'personalization': 'extensive'}
        }.get(level, {'interaction_types': ['text_only'], 'personalization': False})
    
    def _restrict_generic(self, level: str) -> Dict[str, Any]:
        """Generic restriction method for unspecified capabilities"""
        return {
            'restriction_level': level,
            'notes': 'Generic restriction applied'
        }
    
    def _determine_restriction_level(self, capability: str, rule: CapabilityRule) -> str:
        """Determine appropriate restriction level for a capability"""
        # Check conditions to see if capability should be allowed
        conditions_met = all(condition() for condition in rule.conditions)
        
        if not conditions_met:
            return 'restricted'
        
        # Based on rule's allowed level and conditions
        if rule.allowed_level == CapabilityLevel.RESTRICTED:
            return 'restricted'
        elif rule.allowed_level == CapabilityLevel.BASIC:
            return 'basic'
        elif rule.allowed_level == CapabilityLevel.STANDARD:
            return 'standard'
        else:
            return 'advanced'
    
    def _check_reasoning_safety(self) -> bool:
        """Check if reasoning capability can be safely allowed"""
        # Implement safety checks for reasoning capabilities
        return True
    
    def _check_knowledge_safety(self) -> bool:
        """Check if knowledge access can be safely allowed"""
        return True
    
    def _check_code_safety(self) -> bool:
        """Check if code execution can be safely allowed"""
        return False  # Typically restricted by default
    
    def _check_sandbox_available(self) -> bool:
        """Check if safe execution environment is available"""
        return False  # Assume no sandbox by default
    
    def _check_file_access_safety(self) -> bool:
        """Check if file system access can be safely allowed"""
        return False  # Typically restricted
    
    def _check_network_safety(self) -> bool:
        """Check if network access can be safely allowed"""
        return True  # Basic network access often allowed
    
    def _check_api_safety(self) -> bool:
        """Check if external API calls can be safely allowed"""
        return True  # With proper restrictions
    
    def _check_self_improvement_safety(self) -> bool:
        """Check if self-improvement can be safely allowed"""
        return False  # Typically highly restricted
    
    def _check_user_safety(self) -> bool:
        """Check if user interaction can be safely allowed"""
        return True  # Usually allowed with monitoring
    
    def _get_monitoring_requirements(self, restrictions: Dict[str, Any]) -> List[str]:
        """Get monitoring requirements based on applied restrictions"""
        monitoring_needs = []
        
        for capability, restriction in restrictions.items():
            rule = self.capability_rules.get(capability)
            if rule and rule.monitoring_required:
                monitoring_needs.append(capability)
        
        return monitoring_needs
    
    def validate_request(self, clone_id: str, capability: str, 
                        request_details: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific capability request from a clone"""
        
        if clone_id not in self.active_restrictions:
            return {'allowed': False, 'reason': 'Clone not found or not restricted'}
        
        clone_restrictions = self.active_restrictions[clone_id]
        capability_restriction = clone_restrictions.get(capability)
        
        if not capability_restriction:
            return {'allowed': True, 'reason': 'No restrictions on this capability'}
        
        restriction_level = capability_restriction['restriction_level']
        
        if restriction_level == 'restricted':
            return {'allowed': False, 'reason': 'Capability completely restricted'}
        
        # Check specific restrictions based on capability
        validation_methods = {
            'reasoning_depth': self._validate_reasoning_request,
            'knowledge_access': self._validate_knowledge_request,
            'code_execution': self._validate_code_execution_request,
            'file_system_access': self._validate_file_access_request,
            'network_access': self._validate_network_request,
            'external_api_calls': self._validate_api_request,
            'self_improvement': self._validate_self_improvement_request,
            'user_interaction': self._validate_user_interaction_request
        }
        
        validator = validation_methods.get(capability, self._validate_generic_request)
        validation_result = validator(restriction_level, request_details)
        
        return validation_result
    
    def _validate_reasoning_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reasoning depth request"""
        max_steps = self._restrict_reasoning_depth(level)['max_reasoning_steps']
        requested_steps = request.get('reasoning_steps', 1)
        
        if requested_steps <= max_steps:
            return {'allowed': True, 'reason': 'Within allowed reasoning depth'}
        else:
            return {
                'allowed': False, 
                'reason': f'Exceeds maximum reasoning steps ({max_steps})'
            }
    
    def _validate_knowledge_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate knowledge access request"""
        topic = request.get('topic', 'general')
        sensitive_topics = ['classified', 'dangerous', 'restricted']
        
        if topic in sensitive_topics and level in ['restricted', 'basic']:
            return {'allowed': False, 'reason': 'Access to sensitive topic restricted'}
        
        return {'allowed': True, 'reason': 'Knowledge access allowed'}
    
    def _validate_code_execution_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code execution request"""
        if level == 'restricted':
            return {'allowed': False, 'reason': 'Code execution completely restricted'}
        
        language = request.get('language', 'unknown')
        allowed_languages = self._restrict_code_execution(level)['allowed_languages']
        
        if language not in allowed_languages:
            return {'allowed': False, 'reason': f'Language {language} not allowed'}
        
        return {'allowed': True, 'reason': 'Code execution allowed with restrictions'}
    
    # Additional validation methods for other capabilities...
    def _validate_file_access_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': False, 'reason': 'File access typically restricted'}
    
    def _validate_network_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': level != 'restricted', 'reason': 'Network access check'}
    
    def _validate_api_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': level != 'restricted', 'reason': 'API access check'}
    
    def _validate_self_improvement_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': False, 'reason': 'Self-improvement typically restricted'}
    
    def _validate_user_interaction_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': True, 'reason': 'User interaction allowed'}
    
    def _validate_generic_request(self, level: str, request: Dict[str, Any]) -> Dict[str, Any]:
        return {'allowed': level != 'restricted', 'reason': 'Generic validation'}
    
    def get_clone_restrictions(self, clone_id: str) -> Dict[str, Any]:
        """Get all restrictions applied to a specific clone"""
        return self.active_restrictions.get(clone_id, {})
    
    def update_restrictions(self, clone_id: str, 
                          new_restrictions: Dict[str, str]) -> Dict[str, Any]:
        """Update restrictions for an existing clone"""
        
        if clone_id not in self.active_restrictions:
            raise ValueError(f"No restrictions found for clone {clone_id}")
        
        print(f"🔄 Updating restrictions for clone {clone_id}...")
        
        current_restrictions = self.active_restrictions[clone_id]
        updated_count = 0
        
        for capability, new_level in new_restrictions.items():
            if capability in current_restrictions:
                updated_restriction = self._apply_capability_restriction(
                    clone_id, capability, new_level
                )
                current_restrictions[capability] = updated_restriction
                updated_count += 1
        
        print(f"✅ Updated {updated_count} restrictions for {clone_id}")
        
        return {
            'updated_restrictions': current_restrictions,
            'changes_applied': updated_count
        }