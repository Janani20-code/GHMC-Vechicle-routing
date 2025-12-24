"""
MODULE 5.5: VISUALIZER
Creates visualizations for routes and schedules
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class RouteVisualizer:
    def __init__(self, data_loader, router, scheduler):
        self.data_loader = data_loader
        self.router = router
        self.scheduler = scheduler
        self.output_path = ""
        
    def set_output_path(self, path: str):
        """Set output directory for visualizations"""
        self.output_path = path
        os.makedirs(path, exist_ok=True)
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        self._create_dashboard()
        
        # Create cluster-wise visualizations
        for cluster_name in self.router.routes.keys():
            self._create_cluster_visualization(cluster_name)
        
        # Create schedule visualization
        self._create_schedule_visualization()
        
        print(f"✅ All visualizations saved to: {self.output_path}")
    
    def _create_dashboard(self):
        """Create comprehensive dashboard visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Waste Distribution by Cluster
        ax1 = plt.subplot(2, 3, 1)
        self._plot_waste_distribution(ax1)
        
        # 2. Vehicle Utilization
        ax2 = plt.subplot(2, 3, 2)
        self._plot_vehicle_utilization(ax2)
        
        # 3. Route Distance Distribution
        ax3 = plt.subplot(2, 3, 3)
        self._plot_route_distances(ax3)
        
        # 4. Schedule Timeline
        ax4 = plt.subplot(2, 3, 4)
        self._plot_schedule_timeline(ax4)
        
        # 5. Vehicle Type Efficiency
        ax5 = plt.subplot(2, 3, 5)
        self._plot_vehicle_efficiency(ax5)
        
        # 6. Cost Analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_cost_analysis(ax6)
        
        plt.suptitle('Waste Collection Optimization Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_path, "optimization_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_waste_distribution(self, ax):
        """Plot waste distribution by cluster"""
        clusters = self.data_loader.data['clusters']
        
        bars = ax.bar(clusters['SCTP'], clusters['Total_Waste_Tonnes'], 
                     color=plt.cm.tab20.colors)
        ax.set_xlabel('Clusters (SCTPs)')
        ax.set_ylabel('Waste (tonnes)')
        ax.set_title('Waste Distribution by Cluster')
        ax.set_xticklabels(clusters['SCTP'], rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
    
    def _plot_vehicle_utilization(self, ax):
        """Plot vehicle utilization percentages"""
        if not self.scheduler.schedules:
            ax.text(0.5, 0.5, 'No schedule data', ha='center', va='center')
            ax.set_title('Vehicle Utilization')
            return
        
        utilizations = []
        vehicle_ids = []
        
        for cluster_schedules in self.scheduler.schedules.values():
            for schedule in cluster_schedules:
                utilizations.append(schedule['utilization_percentage'])
                vehicle_ids.append(schedule['vehicle_id'])
        
        # Plot top 20 vehicles
        if len(utilizations) > 20:
            indices = np.argsort(utilizations)[-20:]
            utilizations = [utilizations[i] for i in indices]
            vehicle_ids = [vehicle_ids[i] for i in indices]
        
        bars = ax.bar(range(len(utilizations)), utilizations, color=plt.cm.viridis(np.linspace(0, 1, len(utilizations))))
        ax.set_xlabel('Vehicles')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Vehicle Utilization Rates')
        ax.set_xticks(range(len(vehicle_ids)))
        ax.set_xticklabels(vehicle_ids, rotation=45, ha='right')
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.legend()
    
    def _plot_route_distances(self, ax):
        """Plot distribution of route distances"""
        distances = []
        for cluster_routes in self.router.routes.values():
            for route in cluster_routes:
                distances.append(route['total_distance_km'])
        
        ax.hist(distances, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(distances), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(distances):.1f} km')
        ax.set_xlabel('Route Distance (km)')
        ax.set_ylabel('Frequency')
        ax.set_title('Route Distance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_schedule_timeline(self, ax):
        """Plot schedule timeline"""
        if not self.scheduler.schedules:
            ax.text(0.5, 0.5, 'No schedule data', ha='center', va='center')
            ax.set_title('Schedule Timeline')
            return
        
        # Prepare data for Gantt chart
        y_labels = []
        y_positions = []
        start_times = []
        durations = []
        
        y_pos = 0
        for cluster_name, schedules in self.scheduler.schedules.items():
            for schedule in schedules:
                for trip in schedule['trips']:
                    y_labels.append(f"{schedule['vehicle_id']}\n{trip['route_id']}")
                    
                    # Convert time string to numeric
                    start_hour = int(trip['start_time'].split(':')[0])
                    start_minute = int(trip['start_time'].split(':')[1])
                    start_decimal = start_hour + start_minute/60
                    
                    start_times.append(start_decimal)
                    durations.append(trip['duration_hours'])
                    y_positions.append(y_pos)
                    y_pos += 1
        
        # Create Gantt bars
        ax.barh(y_positions, durations, left=start_times, height=0.8, 
               color=plt.cm.tab20.colors, edgecolor='black')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel('Time of Day (Hours)')
        ax.set_title('Vehicle Schedule Timeline')
        ax.grid(True, alpha=0.3)
    
    def _plot_vehicle_efficiency(self, ax):
        """Plot efficiency by vehicle type"""
        if not self.scheduler.schedules:
            ax.text(0.5, 0.5, 'No schedule data', ha='center', va='center')
            ax.set_title('Vehicle Efficiency')
            return
        
        # Group by vehicle type
        type_data = {}
        for cluster_schedules in self.scheduler.schedules.values():
            for schedule in cluster_schedules:
                v_type = schedule['vehicle_type']
                if v_type not in type_data:
                    type_data[v_type] = {'waste': [], 'distance': [], 'utilization': []}
                
                type_data[v_type]['waste'].append(schedule['total_waste_collected'])
                type_data[v_type]['distance'].append(schedule['total_distance_traveled'])
                type_data[v_type]['utilization'].append(schedule['utilization_percentage'])
        
        # Calculate averages
        types = list(type_data.keys())
        avg_waste = [np.mean(type_data[t]['waste']) for t in types]
        avg_distance = [np.mean(type_data[t]['distance']) for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        
        ax.bar(x - width/2, avg_waste, width, label='Avg Waste (tonnes)', alpha=0.7)
        ax.bar(x + width/2, avg_distance, width, label='Avg Distance (km)', alpha=0.7)
        
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Value')
        ax.set_title('Efficiency by Vehicle Type')
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_cost_analysis(self, ax):
        """Plot cost analysis"""
        # Calculate costs (simplified)
        fuel_cost_per_km = 100  # ₹100 per km
        driver_cost_per_hour = 300  # ₹300 per hour
        
        total_distance = 0
        total_hours = 0
        
        for cluster_schedules in self.scheduler.schedules.values():
            for schedule in cluster_schedules:
                total_distance += schedule['total_distance_traveled']
                total_hours += schedule['total_scheduled_hours']
        
        fuel_cost = total_distance * fuel_cost_per_km
        driver_cost = total_hours * driver_cost_per_hour
        total_cost = fuel_cost + driver_cost
        
        # Create pie chart
        costs = [fuel_cost, driver_cost]
        labels = ['Fuel Cost', 'Driver Cost']
        colors = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax.pie(costs, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Cost Analysis\nTotal: ₹{total_cost:,.0f}')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _create_cluster_visualization(self, cluster_name: str):
        """Create visualization for a specific cluster"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get cluster data
        cluster_gvps = self.data_loader.get_cluster_gvps(cluster_name)
        cluster_routes = self.router.routes.get(cluster_name, [])
        
        # 1. Map visualization
        ax1.scatter(cluster_gvps['longitude'], cluster_gvps['latitude'],
                   s=cluster_gvps['waste_tonnes'] * 50,  # Size by waste
                   alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot routes
        colors = plt.cm.tab20.colors
        for i, route in enumerate(cluster_routes):
            color = colors[i % len(colors)]
            
            # Get route stops coordinates
            stops_lon = [s['longitude'] for s in route['stops']]
            stops_lat = [s['latitude'] for s in route['stops']]
            
            # Plot route line
            ax1.plot(stops_lon, stops_lat, color=color, linewidth=2, alpha=0.7, 
                    marker='o', markersize=4)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Route Map - {cluster_name}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Route efficiency
        if cluster_routes:
            route_ids = [r['route_id'] for r in cluster_routes]
            utilizations = [r['capacity_utilization'] for r in cluster_routes]
            distances = [r['total_distance_km'] for r in cluster_routes]
            
            x = np.arange(len(route_ids))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, utilizations, width, label='Utilization (%)', color='skyblue')
            bars2 = ax2.bar(x + width/2, distances, width, label='Distance (km)', color='lightcoral')
            
            ax2.set_xlabel('Route ID')
            ax2.set_ylabel('Value')
            ax2.set_title(f'Route Efficiency - {cluster_name}')
            ax2.set_xticks(x)
            ax2.set_xticklabels(route_ids, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save cluster visualization
        cluster_viz_path = os.path.join(self.output_path, f"cluster_{cluster_name}_routes.png")
        plt.savefig(cluster_viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_schedule_visualization(self):
        """Create detailed schedule visualization"""
        if not self.scheduler.schedules:
            return
        
        # Prepare data
        schedule_data = []
        for cluster_name, schedules in self.scheduler.schedules.items():
            for schedule in schedules:
                for trip in schedule['trips']:
                    schedule_data.append({
                        'cluster': cluster_name,
                        'vehicle': schedule['vehicle_id'],
                        'route': trip['route_id'],
                        'start': trip['start_time'],
                        'end': trip['end_time'],
                        'duration': trip['duration_hours'],
                        'waste': trip['total_waste_tonnes']
                    })
        
        df = pd.DataFrame(schedule_data)
        
        # Create heatmap of schedule density
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert time to numerical
        df['start_hour'] = df['start'].apply(lambda x: int(x.split(':')[0]))
        
        # Create pivot table
        pivot = df.pivot_table(index='vehicle', columns='start_hour', 
                              values='waste', aggfunc='count', fill_value=0)
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        ax.set_xlabel('Start Hour')
        ax.set_ylabel('Vehicle')
        ax.set_title('Schedule Density Heatmap')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        plt.colorbar(im, ax=ax, label='Number of Trips')
        plt.tight_layout()
        
        # Save schedule visualization
        schedule_viz_path = os.path.join(self.output_path, "schedule_heatmap.png")
        plt.savefig(schedule_viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)