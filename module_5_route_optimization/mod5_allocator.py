"""
MODULE 5.2: VEHICLE ALLOCATOR
Allocates vehicles to clusters based on waste volume and constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class VehicleAllocator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.allocation = {}
        self.vehicle_pool = []
        self.constraints = {
            'max_vehicles_per_cluster': 10,
            'min_vehicle_utilization': 0.6,  # 60% minimum utilization
            'preferred_capacity_margin': 0.1,  # 10% margin
            'working_hours_per_day': 8,
            'avg_speed_kmh': 30,
            'service_time_per_stop_min': 10
        }
        
    def create_vehicle_pool(self):
        """
        Create a pool of available vehicles from fleet data
        Each vehicle is treated as an individual unit
        """
        print("\n🚚 Creating vehicle pool...")
        
        self.vehicle_pool = []
        fleet_df = self.data_loader.data['fleet']
        
        for _, row in fleet_df.iterrows():
            vehicle_type = row['vehicle_type']
            capacity = row['capacity_tonnes']
            count = int(row['num_vehicles'])
            
            # Create individual vehicle entries
            for i in range(count):
                self.vehicle_pool.append({
                    'vehicle_id': f"{vehicle_type}_{i+1}",
                    'vehicle_type': vehicle_type,
                    'capacity_tonnes': capacity,
                    'available': True,
                    'assigned_cluster': None,
                    'assigned_trips': []
                })
        
        print(f"✅ Created pool of {len(self.vehicle_pool)} vehicles")
        return self.vehicle_pool
    
    def allocate_vehicles_to_clusters(self) -> Dict:
        """
        Allocate vehicles to clusters based on waste volume
        Returns: Allocation dictionary
        """
        print("\n📍 Allocating vehicles to clusters...")
        
        clusters = self.data_loader.data['clusters']
        self.allocation = {}
        
        # Sort clusters by waste volume (descending)
        sorted_clusters = clusters.sort_values('Total_Waste_Tonnes', ascending=False)
        
        # Create a copy of vehicle pool for allocation
        available_vehicles = self.vehicle_pool.copy()
        
        for _, cluster in sorted_clusters.iterrows():
            cluster_name = cluster['SCTP']
            cluster_waste = cluster['Total_Waste_Tonnes']
            
            print(f"\n📊 Processing cluster: {cluster_name}")
            print(f"   • Waste to collect: {cluster_waste:.2f} tonnes")
            
            # Calculate required vehicles for this cluster
            allocated = self._allocate_for_cluster(
                cluster_name, cluster_waste, available_vehicles
            )
            
            self.allocation[cluster_name] = allocated
            print(f"   • Vehicles allocated: {len(allocated)}")
            
            if allocated:
                total_capacity = sum(v['capacity_tonnes'] for v in allocated)
                utilization = (cluster_waste / total_capacity) * 100
                print(f"   • Capacity utilization: {utilization:.1f}%")
        
        # Check for unallocated vehicles
        unallocated = [v for v in available_vehicles if v['available']]
        if unallocated:
            print(f"\n⚠️  {len(unallocated)} vehicles remain unallocated")
            # Distribute remaining vehicles to clusters with highest waste
            self._distribute_remaining_vehicles(unallocated)
        
        return self.allocation
    
    def _allocate_for_cluster(self, cluster_name: str, waste_tonnes: float, 
                            available_vehicles: List) -> List:
        """
        Allocate vehicles for a specific cluster
        """
        allocated = []
        remaining_waste = waste_tonnes
        
        # Sort vehicles by capacity (descending)
        available_vehicles.sort(key=lambda x: x['capacity_tonnes'], reverse=True)
        
        for vehicle in available_vehicles:
            if not vehicle['available']:
                continue
                
            if remaining_waste <= 0:
                break
            
            # Check if this vehicle would be underutilized
            utilization = remaining_waste / vehicle['capacity_tonnes']
            
            # Only allocate if utilization is reasonable
            if utilization >= self.constraints['min_vehicle_utilization']:
                vehicle['available'] = False
                vehicle['assigned_cluster'] = cluster_name
                
                allocated.append({
                    'vehicle_id': vehicle['vehicle_id'],
                    'vehicle_type': vehicle['vehicle_type'],
                    'capacity_tonnes': vehicle['capacity_tonnes'],
                    'allocated_waste': min(vehicle['capacity_tonnes'], remaining_waste)
                })
                
                remaining_waste -= vehicle['capacity_tonnes']
        
        # If still waste remains, try with smaller vehicles
        if remaining_waste > 0:
            for vehicle in available_vehicles:
                if not vehicle['available']:
                    continue
                    
                if remaining_waste <= 0:
                    break
                
                # Allocate even partially filled vehicles
                vehicle['available'] = False
                vehicle['assigned_cluster'] = cluster_name
                
                allocated.append({
                    'vehicle_id': vehicle['vehicle_id'],
                    'vehicle_type': vehicle['vehicle_type'],
                    'capacity_tonnes': vehicle['capacity_tonnes'],
                    'allocated_waste': min(vehicle['capacity_tonnes'], remaining_waste)
                })
                
                remaining_waste -= vehicle['capacity_tonnes']
        
        return allocated
    
    def _distribute_remaining_vehicles(self, unallocated_vehicles: List):
        """
        Distribute remaining vehicles to clusters as backup
        """
        # Sort clusters by waste per vehicle ratio
        cluster_ratios = []
        for cluster_name, vehicles in self.allocation.items():
            cluster_data = self.data_loader.data['clusters']
            cluster_waste = cluster_data[cluster_data['SCTP'] == cluster_name]['Total_Waste_Tonnes'].iloc[0]
            ratio = cluster_waste / len(vehicles) if vehicles else cluster_waste
            cluster_ratios.append((cluster_name, ratio))
        
        cluster_ratios.sort(key=lambda x: x[1], reverse=True)
        
        for vehicle in unallocated_vehicles:
            if cluster_ratios:
                cluster_name = cluster_ratios[0][0]
                vehicle['available'] = False
                vehicle['assigned_cluster'] = cluster_name
                
                self.allocation[cluster_name].append({
                    'vehicle_id': vehicle['vehicle_id'],
                    'vehicle_type': vehicle['vehicle_type'],
                    'capacity_tonnes': vehicle['capacity_tonnes'],
                    'allocated_waste': 0  # Backup capacity
                })
                
                # Update ratio for next iteration
                cluster_ratios[0] = (cluster_name, cluster_ratios[0][1] / 2)
                cluster_ratios.sort(key=lambda x: x[1], reverse=True)
    
    def calculate_trips_needed(self) -> Dict:
        """
        Calculate number of trips needed for each vehicle
        Returns: Trip requirements per cluster
        """
        print("\n📈 Calculating trip requirements...")
        
        trip_requirements = {}
        clusters = self.data_loader.data['clusters']
        
        for _, cluster in clusters.iterrows():
            cluster_name = cluster['SCTP']
            cluster_waste = cluster['Total_Waste_Tonnes']
            
            if cluster_name not in self.allocation:
                continue
            
            allocated_vehicles = self.allocation[cluster_name]
            total_capacity = sum(v['capacity_tonnes'] for v in allocated_vehicles)
            
            # Calculate trips needed
            if total_capacity > 0:
                trips_needed = max(1, np.ceil(cluster_waste / total_capacity))
            else:
                trips_needed = 0
            
            trip_requirements[cluster_name] = {
                'total_waste_tonnes': cluster_waste,
                'allocated_vehicles': len(allocated_vehicles),
                'total_capacity': total_capacity,
                'trips_needed': int(trips_needed),
                'capacity_utilization': (cluster_waste / total_capacity) * 100 if total_capacity > 0 else 0
            }
        
        return trip_requirements
    
    def save_allocation_results(self, output_path: str):
        """
        Save vehicle allocation results
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save allocation details
        allocation_records = []
        for cluster_name, vehicles in self.allocation.items():
            for vehicle in vehicles:
                allocation_records.append({
                    'cluster': cluster_name,
                    'vehicle_id': vehicle['vehicle_id'],
                    'vehicle_type': vehicle['vehicle_type'],
                    'capacity_tonnes': vehicle['capacity_tonnes'],
                    'allocated_waste': vehicle['allocated_waste']
                })
        
        allocation_df = pd.DataFrame(allocation_records)
        allocation_file = os.path.join(output_path, "vehicle_allocation.xlsx")
        allocation_df.to_excel(allocation_file, index=False)
        
        # Save trip requirements
        trip_reqs = self.calculate_trips_needed()
        trip_df = pd.DataFrame([
            {
                'cluster': cluster,
                **details
            }
            for cluster, details in trip_reqs.items()
        ])
        trip_file = os.path.join(output_path, "trip_requirements.xlsx")
        trip_df.to_excel(trip_file, index=False)
        
        # Save summary report
        summary_file = os.path.join(output_path, "allocation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("VEHICLE ALLOCATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            total_vehicles = sum(len(v) for v in self.allocation.values())
            f.write(f"Total vehicles allocated: {total_vehicles}\n")
            f.write(f"Total clusters: {len(self.allocation)}\n\n")
            
            f.write("ALLOCATION BY CLUSTER:\n")
            f.write("-" * 60 + "\n")
            for cluster_name, vehicles in self.allocation.items():
                cluster_waste = self.data_loader.data['clusters'][
                    self.data_loader.data['clusters']['SCTP'] == cluster_name
                ]['Total_Waste_Tonnes'].iloc[0]
                
                f.write(f"\n{cluster_name}:\n")
                f.write(f"  • GVPs: {len(self.data_loader.get_cluster_gvps(cluster_name))}\n")
                f.write(f"  • Waste: {cluster_waste:.1f} tonnes\n")
                f.write(f"  • Vehicles allocated: {len(vehicles)}\n")
                
                # Vehicle type breakdown
                type_counts = {}
                for v in vehicles:
                    type_counts[v['vehicle_type']] = type_counts.get(v['vehicle_type'], 0) + 1
                
                for v_type, count in type_counts.items():
                    f.write(f"    - {v_type}: {count}\n")
        
        print(f"✅ Allocation results saved to: {output_path}")
        return allocation_file, trip_file, summary_file