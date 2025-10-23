"""
Energy Monitor - Real-time energy consumption tracking and optimization
"""

import time
import psutil
import GPUtil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import threading
import json

class EnergySource(Enum):
    GRID = "grid"
    SOLAR = "solar" 
    BATTERY = "battery"
    HYBRID = "hybrid"

class PowerState(Enum):
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    ULTRA_SAVER = "ultra_saver"

@dataclass
class EnergyConfig:
    target_efficiency: float  # GFLOPS per watt
    max_power_watts: float
    preferred_energy_source: EnergySource
    power_state: PowerState
    co2_limit: float = 100.0  # gCO2 per hour

class EnergyMonitor:
    def __init__(self):
        self.energy_stats = {}
        self.power_limits = {}
        self.optimization_callbacks = []
        self.monitoring_active = False
        self.energy_history = []
        
    def start_energy_monitoring(self, config: EnergyConfig) -> Dict[str, Any]:
        """Start real-time energy monitoring"""
        
        print(f"⚡ Starting energy monitoring...")
        print(f"   Target efficiency: {config.target_efficiency} GFLOPS/W")
        print(f"   Max power: {config.max_power_watts}W")
        print(f"   Power state: {config.power_state.value}")
        
        self.config = config
        self.monitoring_active = True
        
        # Initialize statistics
        self.energy_stats = {
            'total_energy_consumed_wh': 0.0,
            'average_power_watts': 0.0,
            'peak_power_watts': 0.0,
            'current_efficiency': 0.0,
            'co2_emissions_g': 0.0,
            'energy_cost_usd': 0.0,
            'monitoring_start_time': datetime.now(),
            'device_breakdown': {}
        }
        
        # Start background monitoring thread
        self._start_monitoring_thread()
        
        return {
            'status': 'monitoring_active',
            'sampling_interval': 2,  # seconds
            'metrics_tracked': self._get_tracked_metrics(),
            'optimization_enabled': True
        }
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self._sample_energy_metrics()
                    self._check_power_limits()
                    self._run_optimization_callbacks()
                    time.sleep(2)  # Sample every 2 seconds
                except Exception as e:
                    print(f"⚠️ Energy monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
        print("   Background energy monitoring started")
    
    def _sample_energy_metrics(self):
        """Sample current energy and power metrics"""
        
        current_time = datetime.now()
        
        # Get CPU power consumption (estimation)
        cpu_power = self._estimate_cpu_power()
        
        # Get GPU power consumption
        gpu_power = self._get_gpu_power()
        
        # Get memory power
        memory_power = self._estimate_memory_power()
        
        # Get system power (if available)
        system_power = self._get_system_power()
        
        total_power = cpu_power + gpu_power + memory_power + system_power
        
        # Calculate energy consumed since last sample
        time_delta = 2 / 3600  # 2 seconds in hours
        energy_wh = total_power * time_delta
        
        # Update statistics
        self.energy_stats['total_energy_consumed_wh'] += energy_wh
        self.energy_stats['average_power_watts'] = (
            self.energy_stats['average_power_watts'] + total_power
        ) / 2
        self.energy_stats['peak_power_watts'] = max(
            self.energy_stats['peak_power_watts'], total_power
        )
        
        # Update device breakdown
        self.energy_stats['device_breakdown'] = {
            'cpu_watts': cpu_power,
            'gpu_watts': gpu_power,
            'memory_watts': memory_power,
            'system_watts': system_power,
            'total_watts': total_power
        }
        
        # Calculate efficiency (if compute metrics available)
        self._update_efficiency_metrics()
        
        # Calculate CO2 emissions
        self._update_environmental_metrics(energy_wh)
        
        # Store historical data
        self.energy_history.append({
            'timestamp': current_time,
            'total_power_watts': total_power,
            'energy_wh': energy_wh,
            'efficiency_gflops_w': self.energy_stats['current_efficiency'],
            'co2_emissions_g': self.energy_stats['co2_emissions_g']
        })
        
        # Keep only last hour of history
        one_hour_ago = current_time - timedelta(hours=1)
        self.energy_history = [
            point for point in self.energy_history 
            if point['timestamp'] > one_hour_ago
        ]
    
    def _estimate_cpu_power(self) -> float:
        """Estimate CPU power consumption"""
        
        try:
            # Get CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            current_freq = cpu_freq.current if cpu_freq else 2500  # MHz
            
            # Simplified power estimation model
            # Base power + utilization-based power
            base_power = 10.0  # Watts at idle
            dynamic_power = (cpu_percent / 100) * (current_freq / 1000) * 15.0
            
            return base_power + dynamic_power
            
        except Exception:
            return 45.0  # Fallback estimation
    
    def _get_gpu_power(self) -> float:
        """Get GPU power consumption"""
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                total_gpu_power = sum(gpu.load * gpu.powerDraw for gpu in gpus)
                return total_gpu_power
            else:
                return 0.0
        except Exception:
            return 0.0  # No GPU or error
    
    def _estimate_memory_power(self) -> float:
        """Estimate memory power consumption"""
        
        try:
            # Simple estimation based on memory usage
            memory = psutil.virtual_memory()
            memory_usage_gb = memory.used / (1024**3)
            
            # ~3W per GB of active memory (simplified)
            return memory_usage_gb * 3.0
            
        except Exception:
            return 10.0  # Fallback
    
    def _get_system_power(self) -> float:
        """Get system power consumption (motherboard, disks, etc.)"""
        
        # Simplified system power estimation
        return 20.0  # Watts
    
    def _update_efficiency_metrics(self):
        """Update computational efficiency metrics"""
        
        # This would use actual FLOPs measurements
        # For simulation, calculate based on power state
        base_efficiency = 5.0  # GFLOPS/W
        
        # Adjust based on power state
        power_state_multipliers = {
            PowerState.HIGH_PERFORMANCE: 1.2,
            PowerState.BALANCED: 1.0,
            PowerState.POWER_SAVER: 0.8,
            PowerState.ULTRA_SAVER: 0.6
        }
        
        multiplier = power_state_multipliers.get(self.config.power_state, 1.0)
        self.energy_stats['current_efficiency'] = base_efficiency * multiplier
    
    def _update_environmental_metrics(self, energy_wh: float):
        """Update CO2 emissions and cost metrics"""
        
        # CO2 emissions factor (gCO2/Wh) - varies by energy source
        co2_factors = {
            EnergySource.GRID: 0.5,      # gCO2/Wh - grid average
            EnergySource.SOLAR: 0.05,    # gCO2/Wh - solar with manufacturing
            EnergySource.BATTERY: 0.3,   # gCO2/Wh - battery with charging losses
            EnergySource.HYBRID: 0.2     # gCO2/Wh - hybrid average
        }
        
        co2_factor = co2_factors.get(self.config.preferred_energy_source, 0.5)
        co2_emissions = energy_wh * co2_factor
        
        self.energy_stats['co2_emissions_g'] += co2_emissions
        
        # Energy cost (simplified)
        cost_per_kwh = 0.12  # USD per kWh
        self.energy_stats['energy_cost_usd'] += (energy_wh / 1000) * cost_per_kwh
    
    def _check_power_limits(self):
        """Check if power consumption exceeds limits"""
        
        current_power = self.energy_stats['device_breakdown'].get('total_watts', 0)
        
        if current_power > self.config.max_power_watts:
            print(f"🚨 POWER LIMIT EXCEEDED: {current_power:.1f}W > {self.config.max_power_watts}W")
            self._trigger_power_reduction()
    
    def _trigger_power_reduction(self):
        """Trigger power reduction measures"""
        
        print("   Activating power reduction measures...")
        
        # Reduce model precision
        self._reduce_computation_precision()
        
        # Limit batch size
        self._reduce_batch_size()
        
        # Switch to more efficient power state
        if self.config.power_state != PowerState.ULTRA_SAVER:
            self.config.power_state = PowerState.ULTRA_SAVER
            print("   Switching to ultra power saver mode")
    
    def _reduce_computation_precision(self):
        """Reduce computation precision to save power"""
        
        print("   Reducing computation precision...")
        # This would actually modify model precision
        # For now, just log the action
    
    def _reduce_batch_size(self):
        """Reduce batch size to lower memory and compute requirements"""
        
        print("   Reducing batch size...")
        # This would adjust training/inference batch sizes
    
    def _run_optimization_callbacks(self):
        """Run registered optimization callbacks"""
        
        current_stats = self.get_current_stats()
        
        for callback in self.optimization_callbacks:
            try:
                callback(current_stats)
            except Exception as e:
                print(f"⚠️ Optimization callback error: {e}")
    
    def register_optimization_callback(self, callback: Callable):
        """Register callback for automatic optimization"""
        
        self.optimization_callbacks.append(callback)
        print(f"✅ Registered energy optimization callback")
    
    def dynamic_power_management(self, performance_requirement: float):
        """Dynamically manage power based on performance requirements"""
        
        def power_manager(current_stats: Dict[str, Any]):
            current_efficiency = current_stats['current_efficiency']
            target_efficiency = self.config.target_efficiency
            
            efficiency_ratio = current_efficiency / target_efficiency
            
            if efficiency_ratio < 0.8:
                # Efficiency too low, switch to power saver
                if self.config.power_state != PowerState.POWER_SAVER:
                    self.config.power_state = PowerState.POWER_SAVER
                    print("   🔄 Switching to power saver mode")
            
            elif efficiency_ratio > 1.2 and performance_requirement > 0.8:
                # High efficiency and high performance requirement
                if self.config.power_state != PowerState.HIGH_PERFORMANCE:
                    self.config.power_state = PowerState.HIGH_PERFORMANCE
                    print("   🔄 Switching to high performance mode")
        
        return power_manager
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current energy statistics"""
        
        stats = self.energy_stats.copy()
        
        # Add real-time calculations
        monitoring_duration = datetime.now() - stats['monitoring_start_time']
        hours_monitored = monitoring_duration.total_seconds() / 3600
        
        if hours_monitored > 0:
            stats['average_power_over_time'] = (
                stats['total_energy_consumed_wh'] / hours_monitored
            )
        else:
            stats['average_power_over_time'] = 0.0
        
        # Add trend analysis
        stats['power_trend'] = self._calculate_power_trend()
        stats['efficiency_trend'] = self._calculate_efficiency_trend()
        
        return stats
    
    def get_energy_report(self, duration_hours: float = 1.0) -> Dict[str, Any]:
        """Generate comprehensive energy report"""
        
        # Filter history for requested duration
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        relevant_history = [
            point for point in self.energy_history 
            if point['timestamp'] > cutoff_time
        ]
        
        if not relevant_history:
            return {'error': 'No data available for specified duration'}
        
        # Calculate statistics
        power_values = [point['total_power_watts'] for point in relevant_history]
        efficiency_values = [point['efficiency_gflops_w'] for point in relevant_history]
        co2_values = [point['co2_emissions_g'] for point in relevant_history]
        
        total_energy = sum(point['energy_wh'] for point in relevant_history)
        total_co2 = sum(co2_values)
        
        report = {
            'report_duration_hours': duration_hours,
            'total_energy_consumed_wh': total_energy,
            'average_power_watts': np.mean(power_values),
            'power_std_dev': np.std(power_values),
            'peak_power_watts': max(power_values),
            'average_efficiency_gflops_w': np.mean(efficiency_values),
            'total_co2_emissions_g': total_co2,
            'co2_intensity_g_wh': total_co2 / total_energy if total_energy > 0 else 0,
            'energy_cost_usd': total_energy * 0.12 / 1000,  # $0.12 per kWh
            'device_breakdown': self.energy_stats['device_breakdown'],
            'power_state': self.config.power_state.value,
            'efficiency_vs_target': (
                np.mean(efficiency_values) / self.config.target_efficiency
            ),
            'recommendations': self._generate_energy_recommendations(relevant_history)
        }
        
        return report
    
    def _calculate_power_trend(self) -> str:
        """Calculate power consumption trend"""
        
        if len(self.energy_history) < 10:
            return "insufficient_data"
        
        recent_power = [point['total_power_watts'] for point in self.energy_history[-10:]]
        
        if len(recent_power) >= 2:
            trend = np.polyfit(range(len(recent_power)), recent_power, 1)[0]
            if trend > 1.0:
                return "increasing"
            elif trend < -1.0:
                return "decreasing"
            else:
                return "stable"
        
        return "unknown"
    
    def _calculate_efficiency_trend(self) -> str:
        """Calculate efficiency trend"""
        
        if len(self.energy_history) < 10:
            return "insufficient_data"
        
        recent_efficiency = [point['efficiency_gflops_w'] for point in self.energy_history[-10:]]
        
        if len(recent_efficiency) >= 2:
            trend = np.polyfit(range(len(recent_efficiency)), recent_efficiency, 1)[0]
            if trend > 0.1:
                return "improving"
            elif trend < -0.1:
                return "declining"
            else:
                return "stable"
        
        return "unknown"
    
    def _generate_energy_recommendations(self, history: List[Dict]) -> List[str]:
        """Generate energy optimization recommendations"""
        
        recommendations = []
        
        # Analyze power consumption patterns
        avg_power = np.mean([point['total_power_watts'] for point in history])
        avg_efficiency = np.mean([point['efficiency_gflops_w'] for point in history])
        
        if avg_power > self.config.max_power_watts * 0.8:
            recommendations.append("Consider reducing model complexity or batch size")
        
        if avg_efficiency < self.config.target_efficiency * 0.9:
            recommendations.append("Optimize model architecture for better energy efficiency")
        
        if self.config.power_state == PowerState.HIGH_PERFORMANCE:
            recommendations.append("Switch to balanced power state for better efficiency")
        
        # Check for power spikes
        power_std = np.std([point['total_power_watts'] for point in history])
        if power_std > avg_power * 0.3:
            recommendations.append("High power variability detected - optimize workload distribution")
        
        if not recommendations:
            recommendations.append("Energy consumption is within optimal ranges")
        
        return recommendations
    
    def set_power_limit(self, component: str, max_watts: float):
        """Set power limit for specific component"""
        
        self.power_limits[component] = max_watts
        print(f"🔧 Set power limit for {component}: {max_watts}W")
    
    def optimize_for_green_energy(self, solar_availability: float):
        """Optimize energy usage based on green energy availability"""
        
        def green_optimizer(current_stats: Dict[str, Any]):
            # Adjust operations based on solar availability
            if solar_availability > 0.7:
                # Plenty of solar power - can use high performance
                if self.config.power_state != PowerState.HIGH_PERFORMANCE:
                    self.config.power_state = PowerState.HIGH_PERFORMANCE
                    print("   🌞 High solar availability - enabling high performance mode")
            
            elif solar_availability < 0.3:
                # Low solar - use power saver
                if self.config.power_state != PowerState.ULTRA_SAVER:
                    self.config.power_state = PowerState.ULTRA_SAVER
                    print("   🌙 Low solar availability - enabling ultra power saver mode")
        
        return green_optimizer
    
    def estimate_training_energy(self, model_size: int, dataset_size: int, 
                               epochs: int) -> Dict[str, Any]:
        """Estimate energy consumption for training job"""
        
        # Simplified estimation model
        base_energy_per_epoch_wh = model_size * 0.001  # Wh per million parameters
        data_energy_wh = dataset_size * 0.0001  # Wh per sample
        
        total_energy_wh = (base_energy_per_epoch_wh + data_energy_wh) * epochs
        
        # Adjust for power state
        power_state_factors = {
            PowerState.HIGH_PERFORMANCE: 1.2,
            PowerState.BALANCED: 1.0,
            PowerState.POWER_SAVER: 0.7,
            PowerState.ULTRA_SAVER: 0.5
        }
        
        factor = power_state_factors.get(self.config.power_state, 1.0)
        adjusted_energy_wh = total_energy_wh * factor
        
        # Calculate CO2 and cost
        co2_emissions_g = adjusted_energy_wh * 0.5  # gCO2/Wh grid average
        cost_usd = adjusted_energy_wh * 0.12 / 1000  # $0.12 per kWh
        
        return {
            'estimated_energy_wh': adjusted_energy_wh,
            'estimated_co2_emissions_g': co2_emissions_g,
            'estimated_cost_usd': cost_usd,
            'equivalent_car_km': co2_emissions_g / 120,  # Average car: 120g CO2/km
            'equivalent_trees': co2_emissions_g / 21000,  # Tree absorbs ~21kg CO2/year
            'recommended_power_state': self._recommend_power_state_for_training(epochs)
        }
    
    def _recommend_power_state_for_training(self, epochs: int) -> str:
        """Recommend optimal power state for training"""
        
        if epochs > 100:
            return PowerState.POWER_SAVER.value
        elif epochs > 50:
            return PowerState.BALANCED.value
        else:
            return PowerState.HIGH_PERFORMANCE.value
    
    def stop_energy_monitoring(self) -> Dict[str, Any]:
        """Stop energy monitoring and generate final report"""
        
        self.monitoring_active = False
        
        final_report = self.get_energy_report(
            duration_hours=(
                (datetime.now() - self.energy_stats['monitoring_start_time']).total_seconds() / 3600
            )
        )
        
        print("🔌 Energy monitoring stopped")
        print(f"   Total energy consumed: {final_report['total_energy_consumed_wh']:.1f} Wh")
        print(f"   Total CO2 emissions: {final_report['total_co2_emissions_g']:.1f} g")
        print(f"   Average efficiency: {final_report['average_efficiency_gflops_w']:.1f} GFLOPS/W")
        
        return final_report
    
    def _get_tracked_metrics(self) -> List[str]:
        """Get list of tracked energy metrics"""
        
        return [
            'total_energy_consumed_wh',
            'average_power_watts', 
            'peak_power_watts',
            'current_efficiency',
            'co2_emissions_g',
            'energy_cost_usd',
            'device_breakdown',
            'power_trend',
            'efficiency_trend'
        ]
