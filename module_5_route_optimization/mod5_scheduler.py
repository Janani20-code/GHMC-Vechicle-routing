"""
MODULE 5.4: TRIP SCHEDULER
Creates optimized schedules for vehicles including timing and sequencing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import os

class TripScheduler:
    def __init__(self, router, constraints: Dict = None):
        self.router = router
        self.routes = router.routes
        self.schedules = {}
        
        # Default constraints
        self.constraints = {
            'working_hours_start': '06:00',
            'working_hours_end': '14:00',
            'working_hours': 8,  # hours
            'avg_speed_kmh': 30,
            'service_time_per_stop_min': 10,
            'loading_time_depot_min': 15,
            'unloading_time_depot_min': 30,
            'break_time_min': 30,
            'max_routes_per_vehicle': 3
        }
        
        if constraints:
            self.constraints.update(constraints)
    
    def create_schedules(self) -> Dict:
        """
        Create schedules for all vehicles
        Returns: Dictionary of schedules by cluster
        """
        print("\n📅 Creating vehicle schedules...")
        
        self.schedules = {}
        
        for cluster_name, cluster_routes in self.routes.items():
            print(f"\n⏰ Scheduling for {cluster_name}...")
            
            # Group routes by vehicle
            vehicle_routes = {}
            for route in cluster_routes:
                vehicle_id = route.get('assigned_vehicle')
                if vehicle_id:
                    if vehicle_id not in vehicle_routes:
                        vehicle_routes[vehicle_id] = []
                    vehicle_routes[vehicle_id].append(route)
            
            # Schedule each vehicle
            cluster_schedules = []
            for vehicle_id, routes in vehicle_routes.items():
                vehicle_schedule = self._schedule_vehicle(vehicle_id, routes)
                if vehicle_schedule:
                    cluster_schedules.append(vehicle_schedule)
            
            self.schedules[cluster_name] = cluster_schedules
            
            # Print schedule summary
            total_trips = sum(len(s['trips']) for s in cluster_schedules)
            print(f"   • Vehicles scheduled: {len(cluster_schedules)}")
            print(f"   • Total trips scheduled: {total_trips}")
        
        return self.schedules
    
    def _schedule_vehicle(self, vehicle_id: str, routes: List[Dict]) -> Dict:
        """
        Create schedule for a single vehicle
        """
        # Sort routes by waste volume (descending) - larger waste first
        routes_sorted = sorted(routes, key=lambda x: x['total_waste_tonnes'], reverse=True)
        
        # Calculate time for each route
        trips = []
        current_time = datetime.strptime(self.constraints['working_hours_start'], '%H:%M')
        end_time = datetime.strptime(self.constraints['working_hours_end'], '%H:%M')
        
        for route in routes_sorted[:self.constraints['max_routes_per_vehicle']]:
            # Check if we have time for another trip
            if current_time >= end_time:
                break
            
            # Calculate trip time
            travel_time_hours = route['total_distance_km'] / self.constraints['avg_speed_kmh']
            service_time_hours = (route['total_stops'] * self.constraints['service_time_per_stop_min']) / 60
            depot_time_hours = (self.constraints['loading_time_depot_min'] + 
                              self.constraints['unloading_time_depot_min']) / 60
            
            total_trip_time = travel_time_hours + service_time_hours + depot_time_hours
            
            # Create trip schedule
            start_time = current_time
            end_trip_time = current_time + timedelta(hours=total_trip_time)
            
            # Check if trip fits in working hours
            if end_trip_time > end_time:
                # Try to fit by removing some stops (simplified approach)
                # In reality, we would re-optimize the route
                continue
            
            trip = {
                'route_id': route['route_id'],
                'start_time': start_time.strftime('%H:%M'),
                'end_time': end_trip_time.strftime('%H:%M'),
                'duration_hours': round(total_trip_time, 2),
                'travel_time_hours': round(travel_time_hours, 2),
                'service_time_hours': round(service_time_hours, 2),
                'depot_time_hours': round(depot_time_hours, 2),
                'total_stops': route['total_stops'],
                'total_waste_tonnes': route['total_waste_tonnes'],
                'total_distance_km': route['total_distance_km']
            }
            
            trips.append(trip)
            
            # Update current time (add break between trips)
            current_time = end_trip_time + timedelta(minutes=self.constraints['break_time_min'])
        
        if trips:
            # Calculate vehicle utilization
            total_working_hours = (end_time - datetime.strptime(self.constraints['working_hours_start'], '%H:%M')).seconds / 3600
            scheduled_hours = sum(t['duration_hours'] for t in trips)
            utilization = (scheduled_hours / total_working_hours) * 100
            
            vehicle_schedule = {
                'vehicle_id': vehicle_id,
                'vehicle_type': routes[0].get('vehicle_type', 'Unknown'),
                'trips': trips,
                'total_trips': len(trips),
                'total_scheduled_hours': round(scheduled_hours, 2),
                'total_waste_collected': sum(t['total_waste_tonnes'] for t in trips),
                'total_distance_traveled': sum(t['total_distance_km'] for t in trips),
                'utilization_percentage': round(utilization, 1)
            }
            
            return vehicle_schedule
        
        return None
    
    def calculate_efficiency_metrics(self) -> Dict:
        """
        Calculate efficiency metrics for the schedule
        """
        print("\n📊 Calculating efficiency metrics...")
        
        metrics = {
            'overall': {
                'total_vehicles_scheduled': 0,
                'total_trips_scheduled': 0,
                'total_waste_collected': 0,
                'total_distance_traveled': 0,
                'total_scheduled_hours': 0,
                'avg_vehicle_utilization': 0
            },
            'by_cluster': {}
        }
        
        total_utilization = []
        
        for cluster_name, schedules in self.schedules.items():
            cluster_metrics = {
                'vehicles_scheduled': len(schedules),
                'trips_scheduled': sum(len(s['trips']) for s in schedules),
                'waste_collected': sum(s['total_waste_collected'] for s in schedules),
                'distance_traveled': sum(s['total_distance_traveled'] for s in schedules),
                'scheduled_hours': sum(s['total_scheduled_hours'] for s in schedules),
                'avg_utilization': np.mean([s['utilization_percentage'] for s in schedules]) if schedules else 0
            }
            
            metrics['by_cluster'][cluster_name] = cluster_metrics
            
            # Update overall metrics
            metrics['overall']['total_vehicles_scheduled'] += cluster_metrics['vehicles_scheduled']
            metrics['overall']['total_trips_scheduled'] += cluster_metrics['trips_scheduled']
            metrics['overall']['total_waste_collected'] += cluster_metrics['waste_collected']
            metrics['overall']['total_distance_traveled'] += cluster_metrics['distance_traveled']
            metrics['overall']['total_scheduled_hours'] += cluster_metrics['scheduled_hours']
            
            total_utilization.extend([s['utilization_percentage'] for s in schedules])
        
        if total_utilization:
            metrics['overall']['avg_vehicle_utilization'] = np.mean(total_utilization)
        
        return metrics
    
    def save_schedules(self, output_path: str):
        """
        Save all schedules to files
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save detailed schedules
        all_schedules = []
        for cluster_name, schedules in self.schedules.items():
            for schedule in schedules:
                for trip in schedule['trips']:
                    all_schedules.append({
                        'cluster': cluster_name,
                        'vehicle_id': schedule['vehicle_id'],
                        'vehicle_type': schedule['vehicle_type'],
                        'route_id': trip['route_id'],
                        'start_time': trip['start_time'],
                        'end_time': trip['end_time'],
                        'duration_hours': trip['duration_hours'],
                        'stops': trip['total_stops'],
                        'waste_tonnes': trip['total_waste_tonnes'],
                        'distance_km': trip['total_distance_km']
                    })
        
        schedules_df = pd.DataFrame(all_schedules)
        schedules_file = os.path.join(output_path, "vehicle_schedules.xlsx")
        schedules_df.to_excel(schedules_file, index=False)
        
        # Save vehicle summary
        vehicle_summary = []
        for cluster_name, schedules in self.schedules.items():
            for schedule in schedules:
                vehicle_summary.append({
                    'cluster': cluster_name,
                    'vehicle_id': schedule['vehicle_id'],
                    'vehicle_type': schedule['vehicle_type'],
                    'total_trips': schedule['total_trips'],
                    'total_hours': schedule['total_scheduled_hours'],
                    'total_waste': schedule['total_waste_collected'],
                    'total_distance': schedule['total_distance_traveled'],
                    'utilization_percent': schedule['utilization_percentage']
                })
        
        summary_df = pd.DataFrame(vehicle_summary)
        summary_file = os.path.join(output_path, "vehicle_summary.xlsx")
        summary_df.to_excel(summary_file, index=False)
        
        # Calculate and save efficiency metrics
        metrics = self.calculate_efficiency_metrics()
        
        metrics_file = os.path.join(output_path, "efficiency_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCHEDULING EFFICIENCY METRICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 60 + "\n")
            f.write(f"• Vehicles scheduled: {metrics['overall']['total_vehicles_scheduled']}\n")
            f.write(f"• Trips scheduled: {metrics['overall']['total_trips_scheduled']}\n")
            f.write(f"• Waste to collect: {metrics['overall']['total_waste_collected']:.2f} tonnes\n")
            f.write(f"• Total distance: {metrics['overall']['total_distance_traveled']:.2f} km\n")
            f.write(f"• Total scheduled hours: {metrics['overall']['total_scheduled_hours']:.2f}\n")
            f.write(f"• Average vehicle utilization: {metrics['overall']['avg_vehicle_utilization']:.1f}%\n\n")
            
            f.write("CLUSTER-WISE PERFORMANCE:\n")
            f.write("-" * 60 + "\n")
            for cluster_name, cluster_metrics in metrics['by_cluster'].items():
                f.write(f"\n{cluster_name}:\n")
                f.write(f"  • Vehicles: {cluster_metrics['vehicles_scheduled']}\n")
                f.write(f"  • Trips: {cluster_metrics['trips_scheduled']}\n")
                f.write(f"  • Waste: {cluster_metrics['waste_collected']:.2f} tonnes\n")
                f.write(f"  • Distance: {cluster_metrics['distance_traveled']:.2f} km\n")
                f.write(f"  • Hours: {cluster_metrics['scheduled_hours']:.2f}\n")
                f.write(f"  • Utilization: {cluster_metrics['avg_utilization']:.1f}%\n")
        
        print(f"\n✅ Scheduling results saved to: {output_path}")
        print(f"   • Vehicle schedules: vehicle_schedules.xlsx")
        print(f"   • Vehicle summary: vehicle_summary.xlsx")
        print(f"   • Efficiency metrics: efficiency_metrics.txt")
        
        return schedules_file, summary_file, metrics_file