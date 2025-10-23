"""
Knowledge Gateway - Controls knowledge flow between Zain and clones
"""

from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta

class KnowledgeAccessLevel(Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    LIMITED = "limited"
    STANDARD = "standard"
    EXTENSIVE = "extensive"
    FULL = "full"

class KnowledgeDomain(Enum):
    GENERAL = "general_knowledge"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    MEDICINE = "medicine"
    LAW = "law"
    FINANCE = "finance"
    CREATIVE = "creative"
    SENSITIVE = "sensitive"

@dataclass
class KnowledgeRequest:
    request_id: str
    clone_id: str
    domains: List[KnowledgeDomain]
    query: str
    context: Dict[str, Any]
    urgency: int  # 1-10
    timestamp: str

class KnowledgeGateway:
    def __init__(self, knowledge_graph, parent_model):
        self.kg = knowledge_graph
        self.parent_model = parent_model
        self.clone_access_profiles = {}
        self.knowledge_cache = {}
        self.request_log = []
        self.access_policies = self._initialize_access_policies()
        
    def _initialize_access_policies(self) -> Dict[KnowledgeDomain, Dict[str, Any]]:
        """Initialize access policies for different knowledge domains"""
        return {
            KnowledgeDomain.GENERAL: {
                'access_level': KnowledgeAccessLevel.FULL,
                'restrictions': [],
                'approval_required': False,
                'monitoring_level': 'low'
            },
            KnowledgeDomain.SCIENCE: {
                'access_level': KnowledgeAccessLevel.EXTENSIVE,
                'restrictions': ['emerging_technologies'],
                'approval_required': False,
                'monitoring_level': 'medium'
            },
            KnowledgeDomain.TECHNOLOGY: {
                'access_level': KnowledgeAccessLevel.STANDARD,
                'restrictions': ['proprietary_algorithms', 'security_details'],
                'approval_required': False,
                'monitoring_level': 'medium'
            },
            KnowledgeDomain.MEDICINE: {
                'access_level': KnowledgeAccessLevel.LIMITED,
                'restrictions': ['patient_data', 'experimental_treatments'],
                'approval_required': True,
                'monitoring_level': 'high'
            },
            KnowledgeDomain.LAW: {
                'access_level': KnowledgeAccessLevel.LIMITED,
                'restrictions': ['ongoing_cases', 'classified_legal'],
                'approval_required': True,
                'monitoring_level': 'high'
            },
            KnowledgeDomain.FINANCE: {
                'access_level': KnowledgeAccessLevel.LIMITED,
                'restrictions': ['market_insider', 'proprietary_strategies'],
                'approval_required': True,
                'monitoring_level': 'high'
            },
            KnowledgeDomain.CREATIVE: {
                'access_level': KnowledgeAccessLevel.STANDARD,
                'restrictions': ['copyrighted_content'],
                'approval_required': False,
                'monitoring_level': 'low'
            },
            KnowledgeDomain.SENSITIVE: {
                'access_level': KnowledgeAccessLevel.NONE,
                'restrictions': ['all'],
                'approval_required': True,
                'monitoring_level': 'maximum'
            }
        }
    
    def register_clone(self, clone_id: str, access_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Register a clone with its knowledge access profile"""
        
        print(f"📚 Registering clone {clone_id} with knowledge gateway...")
        
        # Validate access profile
        self._validate_access_profile(access_profile)
        
        # Create enhanced access profile
        enhanced_profile = {
            'clone_id': clone_id,
            'access_levels': access_profile.get('access_levels', {}),
            'allowed_domains': access_profile.get('allowed_domains', []),
            'restricted_topics': access_profile.get('restricted_topics', []),
            'cache_settings': access_profile.get('cache_settings', {}),
            'request_limits': access_profile.get('request_limits', {}),
            'registration_time': self._get_timestamp(),
            'total_requests': 0,
            'last_access': None
        }
        
        self.clone_access_profiles[clone_id] = enhanced_profile
        
        print(f"✅ Clone {clone_id} registered with {len(enhanced_profile['allowed_domains'])} allowed domains")
        
        return {
            'registration_status': 'success',
            'access_profile': enhanced_profile,
            'initial_restrictions': self._get_initial_restrictions(enhanced_profile)
        }
    
    def request_knowledge(self, clone_id: str, query: str, 
                         domains: List[KnowledgeDomain],
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process knowledge request from a clone"""
        
        # Create request object
        request = KnowledgeRequest(
            request_id=self._generate_request_id(),
            clone_id=clone_id,
            domains=domains,
            query=query,
            context=context or {},
            urgency=context.get('urgency', 5) if context else 5,
            timestamp=self._get_timestamp()
        )
        
        print(f"🧠 Knowledge request from {clone_id}: {query[:50]}...")
        
        # Check if clone is registered
        if clone_id not in self.clone_access_profiles:
            return self._create_error_response(request, "Clone not registered")
        
        # Validate access permissions
        access_check = self._check_access_permissions(clone_id, domains, query)
        if not access_check['allowed']:
            return self._create_error_response(request, access_check['reason'])
        
        # Check cache first
        cached_response = self._check_knowledge_cache(clone_id, query, domains)
        if cached_response:
            print("   📦 Serving from cache")
            return self._create_success_response(request, cached_response, source='cache')
        
        # Process knowledge retrieval
        knowledge_result = self._retrieve_knowledge(query, domains, context)
        
        # Apply filters and transformations
        filtered_knowledge = self._apply_knowledge_filters(
            knowledge_result, clone_id, domains
        )
        
        # Cache the result
        self._cache_knowledge(clone_id, query, domains, filtered_knowledge)
        
        # Update clone statistics
        self._update_clone_stats(clone_id)
        
        # Log the request
        self._log_request(request, filtered_knowledge)
        
        return self._create_success_response(request, filtered_knowledge, source='knowledge_graph')
    
    def _check_access_permissions(self, clone_id: str, domains: List[KnowledgeDomain], 
                                query: str) -> Dict[str, Any]:
        """Check if clone has permission to access requested knowledge"""
        
        profile = self.clone_access_profiles[clone_id]
        
        # Check domain access
        for domain in domains:
            if domain not in profile['allowed_domains']:
                return {
                    'allowed': False,
                    'reason': f'Access to {domain.value} domain not permitted'
                }
        
        # Check for restricted topics in query
        restricted_topics = profile['restricted_topics']
        query_lower = query.lower()
        for topic in restricted_topics:
            if topic.lower() in query_lower:
                return {
                    'allowed': False,
                    'reason': f'Topic "{topic}" is restricted'
                }
        
        # Check request limits
        limit_check = self._check_request_limits(clone_id)
        if not limit_check['allowed']:
            return limit_check
        
        # Check domain-specific policies
        for domain in domains:
            policy = self.access_policies[domain]
            if policy['approval_required']:
                approval_check = self._check_approval_required(clone_id, domain, query)
                if not approval_check['allowed']:
                    return approval_check
        
        return {'allowed': True, 'reason': 'Access granted'}
    
    def _check_request_limits(self, clone_id: str) -> Dict[str, Any]:
        """Check if clone is within its request limits"""
        
        profile = self.clone_access_profiles[clone_id]
        limits = profile['request_limits']
        
        # Check hourly limit
        hourly_limit = limits.get('hourly', 100)
        recent_requests = self._get_recent_requests(clone_id, hours=1)
        if len(recent_requests) >= hourly_limit:
            return {
                'allowed': False,
                'reason': f'Hourly request limit ({hourly_limit}) exceeded'
            }
        
        # Check daily limit
        daily_limit = limits.get('daily', 1000)
        daily_requests = self._get_recent_requests(clone_id, hours=24)
        if len(daily_requests) >= daily_limit:
            return {
                'allowed': False,
                'reason': f'Daily request limit ({daily_limit}) exceeded'
            }
        
        return {'allowed': True, 'reason': 'Within limits'}
    
    def _check_approval_required(self, clone_id: str, domain: KnowledgeDomain, 
                               query: str) -> Dict[str, Any]:
        """Check if approval is required for sensitive domain access"""
        
        # For now, auto-approve based on urgency and clone trust level
        profile = self.clone_access_profiles[clone_id]
        trust_level = profile.get('trust_level', 0.5)
        
        # Simple auto-approval logic
        if trust_level > 0.8:
            return {'allowed': True, 'reason': 'Auto-approved (high trust)'}
        elif 'urgent' in query.lower():
            return {'allowed': True, 'reason': 'Auto-approved (urgent)'}
        else:
            return {
                'allowed': False,
                'reason': f'Approval required for {domain.value} domain'
            }
    
    def _retrieve_knowledge(self, query: str, domains: List[KnowledgeDomain],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve knowledge from the knowledge graph"""
        
        print(f"   🔍 Retrieving knowledge for: {query}")
        
        # This would integrate with your actual knowledge graph
        # For now, return simulated knowledge
        
        knowledge_result = {
            'query': query,
            'domains': [domain.value for domain in domains],
            'facts': self._simulate_fact_retrieval(query, domains),
            'concepts': self._simulate_concept_retrieval(query, domains),
            'relationships': self._simulate_relationship_retrieval(query, domains),
            'certainty_score': 0.85,
            'retrieval_timestamp': self._get_timestamp(),
            'sources': ['knowledge_graph', 'parent_model']
        }
        
        return knowledge_result
    
    def _apply_knowledge_filters(self, knowledge: Dict[str, Any], clone_id: str,
                               domains: List[KnowledgeDomain]) -> Dict[str, Any]:
        """Apply filters based on clone's access level and domain policies"""
        
        profile = self.clone_access_profiles[clone_id]
        filtered_knowledge = knowledge.copy()
        
        # Remove sensitive facts based on domain policies
        filtered_facts = []
        for fact in knowledge.get('facts', []):
            if self._is_fact_allowed(fact, domains, profile):
                filtered_facts.append(fact)
        
        filtered_knowledge['facts'] = filtered_facts
        
        # Filter concepts based on access level
        access_level = profile['access_levels'].get('general', KnowledgeAccessLevel.LIMITED)
        if access_level in [KnowledgeAccessLevel.NONE, KnowledgeAccessLevel.READ_ONLY]:
            filtered_knowledge['concepts'] = []
            filtered_knowledge['relationships'] = []
        
        # Add access metadata
        filtered_knowledge['access_metadata'] = {
            'applied_filters': len(knowledge.get('facts', [])) - len(filtered_facts),
            'access_level': access_level.value,
            'filtering_timestamp': self._get_timestamp()
        }
        
        return filtered_knowledge
    
    def _is_fact_allowed(self, fact: Dict[str, Any], domains: List[KnowledgeDomain],
                        profile: Dict[str, Any]) -> bool:
        """Check if a specific fact is allowed for the clone"""
        
        # Check against restricted topics
        fact_text = str(fact).lower()
        for topic in profile['restricted_topics']:
            if topic.lower() in fact_text:
                return False
        
        # Check domain-specific restrictions
        for domain in domains:
            policy = self.access_policies[domain]
            if policy['access_level'] == KnowledgeAccessLevel.NONE:
                return False
        
        return True
    
    def _check_knowledge_cache(self, clone_id: str, query: str, 
                             domains: List[KnowledgeDomain]) -> Optional[Dict[str, Any]]:
        """Check if knowledge is available in cache"""
        
        cache_key = self._generate_cache_key(clone_id, query, domains)
        cached_item = self.knowledge_cache.get(cache_key)
        
        if cached_item:
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_item['cache_timestamp'])
            current_time = datetime.now()
            cache_ttl = timedelta(hours=24)  # 24 hour TTL
            
            if current_time - cache_time < cache_ttl:
                return cached_item['knowledge']
            else:
                # Remove expired cache entry
                del self.knowledge_cache[cache_key]
        
        return None
    
    def _cache_knowledge(self, clone_id: str, query: str, domains: List[KnowledgeDomain],
                        knowledge: Dict[str, Any]):
        """Cache knowledge for future requests"""
        
        cache_key = self._generate_cache_key(clone_id, query, domains)
        
        cache_entry = {
            'knowledge': knowledge,
            'cache_timestamp': self._get_timestamp(),
            'access_count': 0,
            'domains': [domain.value for domain in domains]
        }
        
        self.knowledge_cache[cache_key] = cache_entry
        
        # Limit cache size (LRU-like behavior)
        if len(self.knowledge_cache) > 1000:  # Max 1000 entries
            oldest_key = min(self.knowledge_cache.keys(), 
                           key=lambda k: self.knowledge_cache[k]['cache_timestamp'])
            del self.knowledge_cache[oldest_key]
    
    def share_knowledge_with_parent(self, clone_id: str, knowledge: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Allow clones to share knowledge back with Zain (parent model)"""
        
        print(f"🔄 Clone {clone_id} sharing knowledge with parent...")
        
        # Validate clone permissions
        if clone_id not in self.clone_access_profiles:
            return {'success': False, 'error': 'Clone not registered'}
        
        profile = self.clone_access_profiles[clone_id]
        if not profile.get('can_share_knowledge', False):
            return {'success': False, 'error': 'Knowledge sharing not permitted'}
        
        # Process and validate shared knowledge
        processed_knowledge = self._process_shared_knowledge(knowledge, clone_id)
        
        if not processed_knowledge['valid']:
            return {'success': False, 'error': 'Invalid knowledge format'}
        
        # Integrate with parent model's knowledge
        integration_result = self._integrate_with_parent_knowledge(
            processed_knowledge, context
        )
        
        # Log the knowledge sharing
        self._log_knowledge_sharing(clone_id, processed_knowledge, integration_result)
        
        return {
            'success': True,
            'integration_result': integration_result,
            'shared_knowledge_id': processed_knowledge['knowledge_id'],
            'timestamp': self._get_timestamp()
        }
    
    def _process_shared_knowledge(self, knowledge: Dict[str, Any], 
                                clone_id: str) -> Dict[str, Any]:
        """Process and validate knowledge shared by clones"""
        
        # Basic validation
        required_fields = ['content', 'domain', 'certainty']
        for field in required_fields:
            if field not in knowledge:
                return {'valid': False, 'error': f'Missing field: {field}'}
        
        # Check knowledge quality
        quality_score = self._assess_knowledge_quality(knowledge, clone_id)
        if quality_score < 0.5:
            return {'valid': False, 'error': 'Knowledge quality too low'}
        
        # Create processed knowledge object
        processed = {
            'knowledge_id': self._generate_knowledge_id(),
            'content': knowledge['content'],
            'domain': knowledge['domain'],
            'certainty': knowledge['certainty'],
            'quality_score': quality_score,
            'source_clone': clone_id,
            'processing_timestamp': self._get_timestamp(),
            'valid': True
        }
        
        return processed
    
    def _integrate_with_parent_knowledge(self, knowledge: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate shared knowledge with parent model's knowledge base"""
        
        # This would actually integrate with Zain's knowledge graph
        # For now, simulate integration
        
        return {
            'integration_status': 'accepted',
            'integration_method': 'knowledge_graph_update',
            'confidence_boost': 0.1,
            'parent_acknowledgement': True
        }
    
    def get_clone_knowledge_stats(self, clone_id: str) -> Dict[str, Any]:
        """Get knowledge access statistics for a clone"""
        
        if clone_id not in self.clone_access_profiles:
            return {'error': 'Clone not registered'}
        
        profile = self.clone_access_profiles[clone_id]
        recent_requests = self._get_recent_requests(clone_id, hours=24)
        
        domain_stats = {}
        for domain in profile['allowed_domains']:
            domain_requests = [r for r in recent_requests if domain in r.domains]
            domain_stats[domain.value] = len(domain_requests)
        
        return {
            'clone_id': clone_id,
            'total_requests': profile['total_requests'],
            'recent_requests_24h': len(recent_requests),
            'domain_breakdown': domain_stats,
            'cache_hit_rate': self._calculate_cache_hit_rate(clone_id),
            'average_response_time': self._calculate_avg_response_time(clone_id),
            'knowledge_sharing_count': self._get_knowledge_sharing_count(clone_id)
        }
    
    def update_access_profile(self, clone_id: str, 
                            updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a clone's knowledge access profile"""
        
        if clone_id not in self.clone_access_profiles:
            return {'success': False, 'error': 'Clone not registered'}
        
        profile = self.clone_access_profiles[clone_id]
        
        # Apply updates
        for key, value in updates.items():
            if key in profile:
                profile[key] = value
            elif key == 'allowed_domains':
                profile['allowed_domains'] = [KnowledgeDomain(d) for d in value]
        
        profile['last_updated'] = self._get_timestamp()
        
        print(f"✅ Updated knowledge access profile for {clone_id}")
        
        return {
            'success': True,
            'updated_profile': profile,
            'changes_applied': len(updates)
        }
    
    # Helper methods
    def _validate_access_profile(self, profile: Dict[str, Any]):
        """Validate access profile configuration"""
        if 'allowed_domains' not in profile:
            raise ValueError("Access profile must include allowed_domains")
    
    def _get_initial_restrictions(self, profile: Dict[str, Any]) -> List[str]:
        """Get initial restrictions based on access profile"""
        restrictions = []
        
        if not profile.get('can_share_knowledge', False):
            restrictions.append('knowledge_sharing_restricted')
        
        if profile['access_levels'].get('general') == KnowledgeAccessLevel.READ_ONLY:
            restrictions.append('read_only_access')
        
        return restrictions
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return hashlib.md5(f"{self._get_timestamp()}{id(self)}".encode()).hexdigest()[:8]
    
    def _generate_cache_key(self, clone_id: str, query: str, domains: List[KnowledgeDomain]) -> str:
        """Generate cache key for knowledge request"""
        domain_str = ''.join(sorted([d.value for d in domains]))
        return hashlib.md5(f"{clone_id}:{query}:{domain_str}".encode()).hexdigest()
    
    def _generate_knowledge_id(self) -> str:
        """Generate unique knowledge ID"""
        return f"knowledge_{hashlib.md5(self._get_timestamp().encode()).hexdigest()[:12]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def _get_recent_requests(self, clone_id: str, hours: int) -> List[KnowledgeRequest]:
        """Get recent requests from a clone"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            req for req in self.request_log 
            if req.clone_id == clone_id and 
            datetime.fromisoformat(req.timestamp) > cutoff_time
        ]
    
    def _update_clone_stats(self, clone_id: str):
        """Update clone statistics"""
        profile = self.clone_access_profiles[clone_id]
        profile['total_requests'] += 1
        profile['last_access'] = self._get_timestamp()
    
    def _log_request(self, request: KnowledgeRequest, response: Dict[str, Any]):
        """Log knowledge request"""
        log_entry = {
            'request': request,
            'response_size': len(str(response)),
            'response_domains': response.get('domains', []),
            'cache_hit': response.get('source') == 'cache'
        }
        self.request_log.append(log_entry)
    
    def _log_knowledge_sharing(self, clone_id: str, knowledge: Dict[str, Any],
                             result: Dict[str, Any]):
        """Log knowledge sharing activity"""
        print(f"   📖 Logged knowledge sharing from {clone_id}")
    
    def _calculate_cache_hit_rate(self, clone_id: str) -> float:
        """Calculate cache hit rate for a clone"""
        recent_requests = self._get_recent_requests(clone_id, hours=24)
        if not recent_requests:
            return 0.0
        
        cache_hits = len([r for r in recent_requests if r.get('cache_hit', False)])
        return cache_hits / len(recent_requests)
    
    def _calculate_avg_response_time(self, clone_id: str) -> float:
        """Calculate average response time for a clone"""
        # Simulated for now
        return 0.15
    
    def _get_knowledge_sharing_count(self, clone_id: str) -> int:
        """Get count of knowledge sharing events for a clone"""
        # Simulated for now
        return 0
    
    def _assess_knowledge_quality(self, knowledge: Dict[str, Any], clone_id: str) -> float:
        """Assess quality of shared knowledge"""
        # Simple quality assessment
        score = 0.5  # Base score
        
        if knowledge.get('certainty', 0) > 0.8:
            score += 0.2
        
        if len(knowledge.get('content', '')) > 50:
            score += 0.1
        
        # Trust factor based on clone
        profile = self.clone_access_profiles[clone_id]
        trust_level = profile.get('trust_level', 0.5)
        score += trust_level * 0.2
        
        return min(score, 1.0)
    
    def _create_error_response(self, request: KnowledgeRequest, error: str) -> Dict[str, Any]:
        """Create error response for knowledge request"""
        return {
            'success': False,
            'request_id': request.request_id,
            'error': error,
            'timestamp': request.timestamp,
            'knowledge': None
        }
    
    def _create_success_response(self, request: KnowledgeRequest, knowledge: Dict[str, Any],
                               source: str) -> Dict[str, Any]:
        """Create success response for knowledge request"""
        return {
            'success': True,
            'request_id': request.request_id,
            'knowledge': knowledge,
            'source': source,
            'timestamp': request.timestamp,
            'access_metadata': knowledge.get('access_metadata', {})
        }
    
    # Simulation methods for knowledge retrieval
    def _simulate_fact_retrieval(self, query: str, domains: List[KnowledgeDomain]) -> List[Dict]:
        """Simulate fact retrieval from knowledge graph"""
        return [{'fact': f"Simulated fact about {query}", 'confidence': 0.9}]
    
    def _simulate_concept_retrieval(self, query: str, domains: List[KnowledgeDomain]) -> List[Dict]:
        """Simulate concept retrieval from knowledge graph"""
        return [{'concept': f"Concept related to {query}", 'relevance': 0.8}]
    
    def _simulate_relationship_retrieval(self, query: str, domains: List[KnowledgeDomain]) -> List[Dict]:
        """Simulate relationship retrieval from knowledge graph"""
        return [{'relationship': f"Relationship for {query}", 'strength': 0.7}]