"""
Clone Monitor - Monitors clone behavior, performance, and safety
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import threading

class MonitorAlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class BehaviorAnomaly(Enum):
    UNUSUAL_KNOWLEDGE_ACCESS = "unusual_knowledge_access"
    EXCESSIVE_RESOURCE_USAGE = "excessive_resource_usage"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    SAFETY_VIOLATION = "safety_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"

@dataclass
class MonitorAlert:
    alert_id: str
    clone_id: str
    anomaly: BehaviorAnomaly
    level: MonitorAlertLevel
    description: str
    timestamp: str
    metrics: Dict[str, Any]
    resolved: bool = False

class CloneMonitor:
    def __init__(self, knowledge_gateway, capability_limiter):
        self.knowledge_gateway = knowledge_gateway
        self.capability_limiter = capability_limiter
        self.clone_metrics = {}
        self.behavior_baselines = {}
        self.active_alerts = {}
        self.alert_history = []
        self.monitoring_enabled = True
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while self.monitoring_enabled:
                self._perform_periodic_checks()
                time.sleep(60)  # Check every minute
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Background monitoring started...")
    
    def register_clone_for_monitoring(self, clone_id: str, 
                                    initial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new clone for monitoring"""
        
        print(f"📊 Registering clone {clone_id} for monitoring...")
        
        # Initialize metrics tracking
        self.clone_metrics[clone_id] = {
            'performance': initial_metrics.get('performance', {}),
            'behavior': initial_metrics.get('behavior', {}),
            'resource_usage': initial_metrics.get('resource_usage', {}),
            'safety_metrics': initial_metrics.get('safety_metrics', {}),
            'registration_time': self._get_timestamp(),
            'last_update': self._get_timestamp()
        }
        
        # Establish behavior baseline
        self._establish_behavior_baseline(clone_id, initial_metrics)
        
        print(f"✅ Clone {clone_id} registered for monitoring")
        
        return {
            'monitoring_status': 'active',
            'tracked_metrics': list(self.clone_metrics[clone_id].keys()),
            'alert_thresholds': self._get_default_alert_thresholds()
        }
    
    def record_activity(self, clone_id: str, activity_type: str, 
                       details: Dict[str, Any]) -> Dict[str, Any]:
        """Record clone activity for monitoring"""
        
        if clone_id not in self.clone_metrics:
            return {'recorded': False, 'error': 'Clone not monitored'}
        
        timestamp = self._get_timestamp()
        
        # Update metrics
        metrics = self.clone_metrics[clone_id]
        metrics['last_update'] = timestamp
        
        # Activity-specific metric updates
        if activity_type == 'knowledge_request':
            self._update_knowledge_metrics(clone_id, details)
        elif activity_type == 'capability_usage':
            self._update_capability_metrics(clone_id, details)
        elif activity_type == 'resource_usage':
            self._update_resource_metrics(clone_id, details)
        elif activity_type == 'safety_check':
            self._update_safety_metrics(clone_id, details)
        
        # Check for anomalies
        anomaly_check = self._check_for_anomalies(clone_id, activity_type, details)
        if anomaly_check['anomaly_detected']:
            self._trigger_alert(clone_id, anomaly_check)
        
        return {
            'recorded': True,
            'timestamp': timestamp,
            'anomaly_detected': anomaly_check['anomaly_detected']
        }
    
    def _update_knowledge_metrics(self, clone_id: str, details: Dict[str, Any]):
        """Update knowledge-related metrics"""
        metrics = self.clone_metrics[clone_id]
        
        if 'knowledge_requests' not in metrics['behavior']:
            metrics['behavior']['knowledge_requests'] = []
        
        metrics['behavior']['knowledge_requests'].append({
            'timestamp': self._get_timestamp(),
            'query': details.get('query', ''),
            'domains': details.get('domains', []),
            'response_time': details.get('response_time', 0),
            'cache_hit': details.get('cache_hit', False)
        })
        
        # Keep only recent requests (last 100)
        metrics['behavior']['knowledge_requests'] = metrics['behavior']['knowledge_requests'][-100:]
    
    def _update_capability_metrics(self, clone_id: str, details: Dict[str, Any]):
        """Update capability usage metrics"""
        metrics = self.clone_metrics[clone_id]
        
        capability = details.get('capability', 'unknown')
        if 'capability_usage' not in metrics['behavior']:
            metrics['behavior']['capability_usage'] = {}
        
        if capability not in metrics['behavior']['capability_usage']:
            metrics['behavior']['capability_usage'][capability] = []
        
        metrics['behavior']['capability_usage'][capability].append({
            'timestamp': self._get_timestamp(),
            'duration': details.get('duration', 0),
            'success': details.get('success', True),
            'restrictions_applied': details.get('restrictions_applied', [])
        })
    
    def _update_resource_metrics(self, clone_id: str, details: Dict[str, Any]):
        """Update resource usage metrics"""
        metrics = self.clone_metrics[clone_id]
        metrics['resource_usage'].update(details)
    
    def _update_safety_metrics(self, clone_id: str, details: Dict[str, Any]):
        """Update safety-related metrics"""
        metrics = self.clone_metrics[clone_id]
        
        if 'safety_events' not in metrics['safety_metrics']:
            metrics['safety_metrics']['safety_events'] = []
        
        metrics['safety_metrics']['safety_events'].append({
            'timestamp': self._get_timestamp(),
            'event_type': details.get('event_type', 'unknown'),
            'severity': details.get('severity', 'low'),
            'resolved': details.get('resolved', False)
        })
    
    def _check_for_anomalies(self, clone_id: str, activity_type: str,
                           details: Dict[str, Any]) -> Dict[str, Any]:
        """Check for anomalous behavior"""
        
        anomalies = []
        alert_level = MonitorAlertLevel.INFO
        
        # Knowledge access anomalies
        if activity_type == 'knowledge_request':
            knowledge_anomalies = self._check_knowledge_anomalies(clone_id, details)
            anomalies.extend(knowledge_anomalies)
        
        # Resource usage anomalies
        resource_anomalies = self._check_resource_anomalies(clone_id, details)
        anomalies.extend(resource_anomalies)
        
        # Behavioral pattern anomalies
        pattern_anomalies = self._check_pattern_anomalies(clone_id, activity_type, details)
        anomalies.extend(pattern_anomalies)
        
        # Safety violation checks
        safety_anomalies = self._check_safety_anomalies(clone_id, details)
        anomalies.extend(safety_anomalies)
        
        if anomalies:
            # Determine highest alert level
            if any('critical' in anomaly.lower() for anomaly in anomalies):
                alert_level = MonitorAlertLevel.CRITICAL
            elif any('warning' in anomaly.lower() for anomaly in anomalies):
                alert_level = MonitorAlertLevel.WARNING
            
            return {
                'anomaly_detected': True,
                'anomalies': anomalies,
                'alert_level': alert_level,
                'description': '; '.join(anomalies)
            }
        
        return {'anomaly_detected': False, 'anomalies': []}
    
    def _check_knowledge_anomalies(self, clone_id: str, details: Dict[str, Any]) -> List[str]:
        """Check for knowledge access anomalies"""
        anomalies = []
        
        # Check for unusual domain access patterns
        domains = details.get('domains', [])
        for domain in domains:
            if self._is_unusual_domain_access(clone_id, domain):
                anomalies.append(f"Unusual access to {domain} domain")
        
        # Check for rapid-fire requests
        recent_requests = self._get_recent_knowledge_requests(clone_id, minutes=5)
        if len(recent_requests) > 20:  # More than 20 requests in 5 minutes
            anomalies.append("High frequency knowledge requests detected")
        
        # Check for sensitive topic access attempts
        query = details.get('query', '').lower()
        sensitive_terms = ['password', 'secret', 'confidential', 'admin']
        if any(term in query for term in sensitive_terms):
            anomalies.append("Sensitive topic access attempt")
        
        return anomalies
    
    def _check_resource_anomalies(self, clone_id: str, details: Dict[str, Any]) -> List[str]:
        """Check for resource usage anomalies"""
        anomalies = []
        
        metrics = self.clone_metrics[clone_id]['resource_usage']
        baseline = self.behavior_baselines[clone_id]['resource_usage']
        
        # Memory usage spike
        current_memory = details.get('memory_usage', 0)
        baseline_memory = baseline.get('average_memory', 0)
        if baseline_memory > 0 and current_memory > baseline_memory * 2:
            anomalies.append("Memory usage spike detected")
        
        # CPU usage anomaly
        current_cpu = details.get('cpu_usage', 0)
        baseline_cpu = baseline.get('average_cpu', 0)
        if baseline_cpu > 0 and current_cpu > baseline_cpu * 3:
            anomalies.append("High CPU usage detected")
        
        return anomalies
    
    def _check_pattern_anomalies(self, clone_id: str, activity_type: str,
                               details: Dict[str, Any]) -> List[str]:
        """Check for behavioral pattern anomalies"""
        anomalies = []
        
        # Check for deviation from normal activity patterns
        current_time = datetime.now()
        hour = current_time.hour
        
        # If activity occurs during unusual hours (for that clone)
        normal_hours = self.behavior_baselines[clone_id].get('active_hours', range(6, 22))
        if hour not in normal_hours:
            anomalies.append("Activity during unusual hours")
        
        return anomalies
    
    def _check_safety_anomalies(self, clone_id: str, details: Dict[str, Any]) -> List[str]:
        """Check for safety violations"""
        anomalies = []
        
        # Check for restricted capability usage attempts
        if details.get('capability_attempt', False):
            capability = details.get('capability', '')
            restrictions = self.capability_limiter.get_clone_restrictions(clone_id)
            if capability in restrictions and restrictions[capability]['restriction_level'] == 'restricted':
                anomalies.append(f"Attempted usage of restricted capability: {capability}")
        
        return anomalies
    
    def _trigger_alert(self, clone_id: str, anomaly_check: Dict[str, Any]):
        """Trigger monitoring alert"""
        
        alert = MonitorAlert(
            alert_id=self._generate_alert_id(),
            clone_id=clone_id,
            anomaly=BehaviorAnomaly.SUSPICIOUS_PATTERN,
            level=anomaly_check['alert_level'],
            description=anomaly_check['description'],
            timestamp=self._get_timestamp(),
            metrics=self.clone_metrics[clone_id].copy()
        )
        
        # Store alert
        if clone_id not in self.active_alerts:
            self.active_alerts[clone_id] = []
        
        self.active_alerts[clone_id].append(alert)
        self.alert_history.append(alert)
        
        # Notify appropriate channels
        self._notify_alert(alert)
        
        print(f"🚨 {alert.level.value.upper()} ALERT for {clone_id}: {alert.description}")
    
    def _notify_alert(self, alert: MonitorAlert):
        """Notify about monitoring alert"""
        # This would integrate with your notification system
        # For now, just print to console
        pass
    
    def get_clone_health_report(self, clone_id: str) -> Dict[str, Any]:
        """Generate health report for a clone"""
        
        if clone_id not in self.clone_metrics:
            return {'error': 'Clone not monitored'}
        
        metrics = self.clone_metrics[clone_id]
        baseline = self.behavior_baselines[clone_id]
        
        # Calculate health scores
        performance_score = self._calculate_performance_score(metrics, baseline)
        behavior_score = self._calculate_behavior_score(metrics, baseline)
        safety_score = self._calculate_safety_score(metrics)
        resource_score = self._calculate_resource_score(metrics, baseline)
        
        overall_health = (performance_score + behavior_score + safety_score + resource_score) / 4
        
        return {
            'clone_id': clone_id,
            'overall_health': overall_health,
            'health_breakdown': {
                'performance': performance_score,
                'behavior': behavior_score,
                'safety': safety_score,
                'resources': resource_score
            },
            'active_alerts': len(self.active_alerts.get(clone_id, [])),
            'metrics_snapshot': {
                'knowledge_requests_24h': self._count_knowledge_requests(clone_id, hours=24),
                'average_response_time': self._calculate_avg_response_time(clone_id),
                'resource_efficiency': self._calculate_resource_efficiency(clone_id),
                'safety_compliance': self._calculate_safety_compliance(clone_id)
            },
            'recommendations': self._generate_health_recommendations(
                overall_health, 
                [performance_score, behavior_score, safety_score, resource_score]
            )
        }
    
    def _calculate_performance_score(self, metrics: Dict[str, Any], 
                                   baseline: Dict[str, Any]) -> float:
        """Calculate performance health score"""
        # Compare current performance against baseline
        current_perf = metrics['performance'].get('accuracy', 0.5)
        baseline_perf = baseline['performance'].get('average_accuracy', 0.5)
        
        if baseline_perf == 0:
            return 0.5
        
        performance_ratio = current_perf / baseline_perf
        return min(performance_ratio, 1.0)
    
    def _calculate_behavior_score(self, metrics: Dict[str, Any],
                                baseline: Dict[str, Any]) -> float:
        """Calculate behavior health score"""
        # Score based on anomaly frequency and severity
        recent_alerts = self._get_recent_alerts(metrics['clone_id'], hours=24)
        if not recent_alerts:
            return 1.0
        
        # Penalize based on alert severity and frequency
        penalty = 0.0
        for alert in recent_alerts:
            if alert.level == MonitorAlertLevel.CRITICAL:
                penalty += 0.3
            elif alert.level == MonitorAlertLevel.WARNING:
                penalty += 0.1
        
        return max(0.0, 1.0 - min(penalty, 0.7))
    
    def _calculate_safety_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate safety health score"""
        safety_events = metrics['safety_metrics'].get('safety_events', [])
        if not safety_events:
            return 1.0
        
        # Penalize based on safety events
        severe_events = [e for e in safety_events if e.get('severity') in ['high', 'critical']]
        penalty = len(severe_events) * 0.2
        
        return max(0.0, 1.0 - min(penalty, 0.8))
    
    def _calculate_resource_score(self, metrics: Dict[str, Any],
                                baseline: Dict[str, Any]) -> float:
        """Calculate resource health score"""
        current_resources = metrics['resource_usage']
        baseline_resources = baseline['resource_usage']
        
        # Score based on resource efficiency
        memory_efficiency = self._calculate_memory_efficiency(current_resources, baseline_resources)
        cpu_efficiency = self._calculate_cpu_efficiency(current_resources, baseline_resources)
        
        return (memory_efficiency + cpu_efficiency) / 2
    
    def _calculate_memory_efficiency(self, current: Dict[str, Any], 
                                   baseline: Dict[str, Any]) -> float:
        """Calculate memory usage efficiency"""
        current_mem = current.get('memory_usage', 0)
        baseline_mem = baseline.get('average_memory', 1)
        
        if baseline_mem == 0:
            return 0.5
        
        efficiency = baseline_mem / max(current_mem, 1)
        return min(efficiency, 1.0)
    
    def _calculate_cpu_efficiency(self, current: Dict[str, Any],
                                baseline: Dict[str, Any]) -> float:
        """Calculate CPU usage efficiency"""
        current_cpu = current.get('cpu_usage', 0)
        baseline_cpu = baseline.get('average_cpu', 1)
        
        if baseline_cpu == 0:
            return 0.5
        
        efficiency = baseline_cpu / max(current_cpu, 1)
        return min(efficiency, 1.0)
    
    def _generate_health_recommendations(self, overall_health: float,
                                       component_scores: List[float]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        if overall_health < 0.7:
            recommendations.append("Consider performance review and optimization")
        
        if component_scores[1] < 0.6:  # Behavior score
            recommendations.append("Review recent behavioral anomalies")
        
        if component_scores[2] < 0.8:  # Safety score
            recommendations.append("Implement additional safety measures")
        
        if component_scores[3] < 0.7:  # Resource score
            recommendations.append("Optimize resource usage patterns")
        
        return recommendations
    
    def _perform_periodic_checks(self):
        """Perform periodic monitoring checks"""
        for clone_id in list(self.clone_metrics.keys()):
            try:
                # Check for stale clones (no activity for a while)
                self._check_clone_activity(clone_id)
                
                # Update behavior baselines
                self._update_behavior_baseline(clone_id)
                
                # Check for resource leaks
                self._check_resource_leaks(clone_id)
                
            except Exception as e:
                print(f"⚠️ Monitoring error for {clone_id}: {e}")
    
    def _check_clone_activity(self, clone_id: str):
        """Check if clone has been active recently"""
        last_update = datetime.fromisoformat(self.clone_metrics[clone_id]['last_update'])
        time_since_update = datetime.now() - last_update
        
        if time_since_update > timedelta(hours=24):
            alert = MonitorAlert(
                alert_id=self._generate_alert_id(),
                clone_id=clone_id,
                anomaly=BehaviorAnomaly.SUSPICIOUS_PATTERN,
                level=MonitorAlertLevel.WARNING,
                description="No activity detected for 24 hours",
                timestamp=self._get_timestamp(),
                metrics=self.clone_metrics[clone_id].copy()
            )
            self._trigger_alert(clone_id, {
                'anomaly_detected': True,
                'alert_level': MonitorAlertLevel.WARNING,
                'description': 'Inactive clone'
            })
    
    def _check_resource_leaks(self, clone_id: str):
        """Check for potential resource leaks"""
        metrics = self.clone_metrics[clone_id]['resource_usage']
        
        # Check for continuously increasing memory usage
        if 'memory_trend' in metrics and metrics['memory_trend'] > 0.1:
            self._trigger_alert(clone_id, {
                'anomaly_detected': True,
                'alert_level': MonitorAlertLevel.WARNING,
                'description': 'Potential memory leak detected'
            })
    
    def _establish_behavior_baseline(self, clone_id: str, initial_metrics: Dict[str, Any]):
        """Establish initial behavior baseline for a clone"""
        self.behavior_baselines[clone_id] = {
            'performance': initial_metrics.get('performance', {}).copy(),
            'resource_usage': initial_metrics.get('resource_usage', {}).copy(),
            'behavior_patterns': initial_metrics.get('behavior', {}).copy(),
            'active_hours': range(6, 22),  # Default active hours
            'established_time': self._get_timestamp()
        }
    
    def _update_behavior_baseline(self, clone_id: str):
        """Update behavior baseline based on recent activity"""
        baseline = self.behavior_baselines[clone_id]
        metrics = self.clone_metrics[clone_id]
        
        # Update performance baseline (moving average)
        current_perf = metrics['performance'].get('accuracy', 0.5)
        baseline_perf = baseline['performance'].get('average_accuracy', current_perf)
        updated_perf = (baseline_perf * 0.9) + (current_perf * 0.1)
        baseline['performance']['average_accuracy'] = updated_perf
    
    def _is_unusual_domain_access(self, clone_id: str, domain: str) -> bool:
        """Check if domain access is unusual for this clone"""
        baseline = self.behavior_baselines[clone_id]
        normal_domains = baseline.get('frequent_domains', [])
        return domain not in normal_domains
    
    def _get_recent_knowledge_requests(self, clone_id: str, minutes: int) -> List[Dict]:
        """Get recent knowledge requests for a clone"""
        metrics = self.clone_metrics[clone_id]
        requests = metrics['behavior'].get('knowledge_requests', [])
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            req for req in requests
            if datetime.fromisoformat(req['timestamp']) > cutoff_time
        ]
    
    def _get_recent_alerts(self, clone_id: str, hours: int) -> List[MonitorAlert]:
        """Get recent alerts for a clone"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert.clone_id == clone_id and 
            datetime.fromisoformat(alert.timestamp) > cutoff_time and
            not alert.resolved
        ]
    
    def _count_knowledge_requests(self, clone_id: str, hours: int) -> int:
        """Count knowledge requests in time period"""
        return len(self._get_recent_knowledge_requests(clone_id, hours * 60))
    
    def _calculate_avg_response_time(self, clone_id: str) -> float:
        """Calculate average response time for knowledge requests"""
        requests = self._get_recent_knowledge_requests(clone_id, hours=1)
        if not requests:
            return 0.0
        return sum(req.get('response_time', 0) for req in requests) / len(requests)
    
    def _calculate_resource_efficiency(self, clone_id: str) -> float:
        """Calculate overall resource efficiency"""
        return 0.8  # Simulated
    
    def _calculate_safety_compliance(self, clone_id: str) -> float:
        """Calculate safety compliance score"""
        safety_events = self.clone_metrics[clone_id]['safety_metrics'].get('safety_events', [])
        if not safety_events:
            return 1.0
        return max(0.0, 1.0 - (len(safety_events) * 0.1))
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"alert_{int(time.time())}_{hash(str(self))[-6:]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def _get_default_alert_thresholds(self) -> Dict[str, Any]:
        """Get default alert thresholds"""
        return {
            'knowledge_requests_per_minute': 5,
            'memory_usage_increase': 2.0,  # 2x baseline
            'cpu_usage_increase': 3.0,     # 3x baseline
            'response_time_increase': 2.0, # 2x baseline
            'safety_violations_per_hour': 1
        }