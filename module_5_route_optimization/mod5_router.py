"""
MODULE 5.3: ROUTE OPTIMIZER
Optimizes routes within each cluster using nearest neighbor algorithm
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import Dict, List, Tuple
import os
import itertools

class RouteOptimizer:
    def __init__(self, data_loader, allocator):
        self.data_loader = data_loader
        self.allocator = allocator
        self.routes = {}
        self.distance_matrix_cache = {}
        
    def optimize_routes_for_cluster(self, cluster_name: str, 
                                   vehicle_capacity: float) -> List[Dict]:
        """
        Optimize routes for a specific cluster using vehicle capacity
        Returns: List of optimized routes
        """
        print(f"  Optimizing routes for {cluster_name} (capacity: {vehicle_capacity} tonnes)...")
        
        # Get GVPs for this cluster
        gvps_df = self.data_loader.get_cluster_gvps(cluster_name)
        if gvps_df.empty:
            return []
        
        # Get SCTP location for this cluster
        sctp_df = self.data_loader.data['clusters']
        sctp_data = sctp_df[sctp_df['SCTP'] == cluster_name].iloc[0]
        sctp_location = (sctp_data['SCTP_Lat'], sctp_data['SCTP_Lon'])
        
        # Sort GVPs by waste volume (descending)
        gvps_sorted = gvps_df.sort_values('waste_tonnes', ascending=False)
        
        # Use Nearest Neighbor algorithm with capacity constraints
        routes = self._nearest_neighbor_with_capacity(
            gvps_sorted, sctp_location, vehicle_capacity
        )
        
        return routes
    
    def _nearest_neighbor_with_capacity(self, gvps_df: pd.DataFrame, 
                                       depot: Tuple, capacity: float) -> List[Dict]:
        """
        Nearest neighbor algorithm with capacity constraints
        """
        routes = []
        unvisited = gvps_df.copy()
        
        route_id = 1
        while not unvisited.empty:
            # Start new route at depot
            current_location = depot
            current_load = 0
            route_stops = []
            route_distance = 0
            
            # Find nearest GVP to start
            nearest_gvp = self._find_nearest_gvp(current_location, unvisited)
            
            while nearest_gvp is not None:
                gvp_waste = nearest_gvp['waste_tonnes']
                
                # Check capacity constraint
                if current_load + gvp_waste <= capacity:
                    # Calculate distance to this GVP
                    gvp_location = (nearest_gvp['latitude'], nearest_gvp['longitude'])
                    distance = self._calculate_distance(current_location, gvp_location)
                    
                    # Add to route
                    route_stops.append({
                        'gvp_id': int(nearest_gvp['s_no']),
                        'location': nearest_gvp['location'],
                        'waste_tonnes': gvp_waste,
                        'latitude': nearest_gvp['latitude'],
                        'longitude': nearest_gvp['longitude'],
                        'distance_from_prev_km': distance
                    })
                    
                    current_load += gvp_waste
                    route_distance += distance
                    current_location = gvp_location
                    
                    # Remove from unvisited
                    unvisited = unvisited[unvisited['s_no'] != nearest_gvp['s_no']]
                    
                    # Find next nearest GVP
                    nearest_gvp = self._find_nearest_gvp(current_location, unvisited)
                else:
                    # Capacity reached, return to depot
                    break
            
            # Return to depot
            if current_location != depot:
                distance_to_depot = self._calculate_distance(current_location, depot)
                route_distance += distance_to_depot
            
            # Create route summary
            if route_stops:
                route_summary = {
                    'route_id': f"R{route_id}",
                    'total_stops': len(route_stops),
                    'total_waste_tonnes': current_load,
                    'total_distance_km': round(route_distance, 2),
                    'capacity_utilization': (current_load / capacity) * 100,
                    'stops': route_stops,
                    'depot_return_distance': distance_to_depot if 'distance_to_depot' in locals() else 0
                }
                routes.append(route_summary)
                route_id += 1
            else:
                break
        
        return routes
    
    def _find_nearest_gvp(self, current_location: Tuple, 
                         gvps_df: pd.DataFrame) -> pd.Series:
        """
        Find the nearest GVP to current location
        """
        if gvps_df.empty:
            return None
        
        min_distance = float('inf')
        nearest_gvp = None
        
        for _, gvp in gvps_df.iterrows():
            gvp_location = (gvp['latitude'], gvp['longitude'])
            distance = self._calculate_distance(current_location, gvp_location)
            
            if distance < min_distance:
                min_distance = distance
                nearest_gvp = gvp
        
        return nearest_gvp
    
    def _calculate_distance(self, location1: Tuple, location2: Tuple) -> float:
        """
        Calculate geodesic distance between two points
        Uses caching for performance
        """
        cache_key = f"{location1}-{location2}"
        if cache_key in self.distance_matrix_cache:
            return self.distance_matrix_cache[cache_key]
        
        try:
            distance = geodesic(location1, location2).km
            self.distance_matrix_cache[cache_key] = distance
            return distance
        except:
            return 0.0
    
    def optimize_all_clusters(self, output_path: str):
        """
        Optimize routes for all clusters
        """
        print("\n🔄 Optimizing routes for all clusters...")
        
        all_routes = {}
        route_details = []
        
        for cluster_name in self.allocator.allocation.keys():
            print(f"\n📦 Processing cluster: {cluster_name}")
            
            # Get vehicles allocated to this cluster
            allocated_vehicles = self.allocator.allocation[cluster_name]
            
            # For each vehicle type, create routes
            cluster_routes = []
            
            # Group vehicles by type for efficiency
            vehicle_types = {}
            for vehicle in allocated_vehicles:
                v_type = vehicle['vehicle_type']
                if v_type not in vehicle_types:
                    vehicle_types[v_type] = []
                vehicle_types[v_type].append(vehicle)
            
            # Create routes for each vehicle type
            for v_type, vehicles in vehicle_types.items():
                capacity = vehicles[0]['capacity_tonnes']
                
                # Get routes for this vehicle type
                routes = self.optimize_routes_for_cluster(cluster_name, capacity)
                
                # Assign routes to vehicles
                for i, (vehicle, route) in enumerate(zip(vehicles, routes)):
                    if i < len(routes):
                        route['assigned_vehicle'] = vehicle['vehicle_id']
                        route['vehicle_type'] = v_type
                        cluster_routes.append(route)
                        
                        # Save route details
                        for stop in route['stops']:
                            route_details.append({
                                'cluster': cluster_name,
                                'route_id': route['route_id'],
                                'vehicle_id': vehicle['vehicle_id'],
                                'vehicle_type': v_type,
                                'gvp_id': stop['gvp_id'],
                                'location': stop['location'],
                                'waste_tonnes': stop['waste_tonnes'],
                                'stop_sequence': route['stops'].index(stop) + 1,
                                'distance_from_prev_km': stop['distance_from_prev_km']
                            })
            
            all_routes[cluster_name] = cluster_routes
            
            # Print cluster summary
            total_stops = sum(len(r['stops']) for r in cluster_routes)
            total_distance = sum(r['total_distance_km'] for r in cluster_routes)
            total_waste = sum(r['total_waste_tonnes'] for r in cluster_routes)
            
            print(f"   • Routes created: {len(cluster_routes)}")
            print(f"   • Total stops: {total_stops}")
            print(f"   • Total distance: {total_distance:.2f} km")
            print(f"   • Total waste: {total_waste:.2f} tonnes")
        
        self.routes = all_routes
        
        # Save results
        self._save_route_results(output_path, route_details)
        
        return all_routes
    
    def _save_route_results(self, output_path: str, route_details: List):
        """
        Save route optimization results
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save detailed route information
        route_details_df = pd.DataFrame(route_details)
        details_file = os.path.join(output_path, "route_details.xlsx")
        route_details_df.to_excel(details_file, index=False)
        
        # Save route summaries
        summary_records = []
        for cluster_name, cluster_routes in self.routes.items():
            for route in cluster_routes:
                summary_records.append({
                    'cluster': cluster_name,
                    'route_id': route['route_id'],
                    'vehicle_id': route.get('assigned_vehicle', 'N/A'),
                    'vehicle_type': route.get('vehicle_type', 'N/A'),
                    'total_stops': route['total_stops'],
                    'total_waste_tonnes': route['total_waste_tonnes'],
                    'total_distance_km': route['total_distance_km'],
                    'capacity_utilization': route['capacity_utilization'],
                    'depot_return_distance': route.get('depot_return_distance', 0)
                })
        
        summary_df = pd.DataFrame(summary_records)
        summary_file = os.path.join(output_path, "route_summaries.xlsx")
        summary_df.to_excel(summary_file, index=False)
        
        # Save optimization report
        report_file = os.path.join(output_path, "route_optimization_report.txt")
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ROUTE OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            total_routes = sum(len(routes) for routes in self.routes.values())
            total_stops = sum(len(r['stops']) for routes in self.routes.values() for r in routes)
            total_distance = sum(r['total_distance_km'] for routes in self.routes.values() for r in routes)
            total_waste = sum(r['total_waste_tonnes'] for routes in self.routes.values() for r in routes)
            
            f.write("OVERALL SUMMARY:\n")
            f.write(f"• Total routes optimized: {total_routes}\n")
            f.write(f"• Total GVP stops covered: {total_stops}\n")
            f.write(f"• Total route distance: {total_distance:.2f} km\n")
            f.write(f"• Total waste allocated: {total_waste:.2f} tonnes\n\n")
            
            f.write("CLUSTER-WISE ANALYSIS:\n")
            f.write("-" * 60 + "\n")
            for cluster_name, cluster_routes in self.routes.items():
                cluster_stops = sum(len(r['stops']) for r in cluster_routes)
                cluster_distance = sum(r['total_distance_km'] for r in cluster_routes)
                cluster_waste = sum(r['total_waste_tonnes'] for r in cluster_routes)
                
                f.write(f"\n{cluster_name}:\n")
                f.write(f"  • Routes: {len(cluster_routes)}\n")
                f.write(f"  • Stops: {cluster_stops}\n")
                f.write(f"  • Distance: {cluster_distance:.2f} km\n")
                f.write(f"  • Waste: {cluster_waste:.2f} tonnes\n")
                
                # Efficiency metrics
                if cluster_routes:
                    avg_utilization = np.mean([r['capacity_utilization'] for r in cluster_routes])
                    avg_distance = np.mean([r['total_distance_km'] for r in cluster_routes])
                    f.write(f"  • Avg utilization: {avg_utilization:.1f}%\n")
                    f.write(f"  • Avg route distance: {avg_distance:.2f} km\n")
        
        print(f"\n✅ Route optimization results saved to: {output_path}")
        print(f"   • Detailed routes: route_details.xlsx")
        print(f"   • Route summaries: route_summaries.xlsx")
        print(f"   • Optimization report: route_optimization_report.txt")