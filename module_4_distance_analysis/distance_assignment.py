"""
MODULE 4: DISTANCE ANALYSIS & CLUSTERING
This module builds on the transformed data from Module 3 to:
1. Calculate distances between all GVPs and SCTPs
2. Assign each GVP to the nearest SCTP
3. Create optimized clusters for vehicle routing
4. Generate visualizations and statistics
"""

import pandas as pd
import numpy as np
import math
import os
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
BASE_DIR = r"E:\vehicle_routing_system\data"
INPUT_FILE = os.path.join(BASE_DIR, "transformed_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "module_4_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# DISTANCE ANALYZER CLASS
# ---------------------------------------------------
class DistanceAnalyzer:
    def __init__(self):
        self.gvp_df = None
        self.sctp_df = None
        self.fleet_df = None
        self.distance_matrix = None
        self.clusters = {}
        self.statistics = {}
        self.results = {}
        
        # Constants
        self.avg_speed_kmh = 30  # Average vehicle speed
        self.service_time_min = 10  # Time per stop
        self.working_hours = 8  # Hours per day
        self.fuel_rate_lpkm = 0.3  # Fuel consumption liter/km
        self.fuel_cost_pl = 100  # Fuel cost per liter
        
    def load_data(self):
        """Load transformed data from Module 3"""
        print("📂 Loading transformed data...")
        
        try:
            # Load all sheets
            self.gvp_df = pd.read_excel(INPUT_FILE, sheet_name='GVPs')
            self.sctp_df = pd.read_excel(INPUT_FILE, sheet_name='SCTPs')
            self.fleet_df = pd.read_excel(INPUT_FILE, sheet_name='Fleet')
            
            # Validate data
            self._validate_data()
            
            print(f"✅ Data loaded successfully")
            print(f"   GVPs: {len(self.gvp_df)} locations")
            print(f"   SCTPs: {len(self.sctp_df)} transfer stations")
            print(f"   Fleet: {self.fleet_df['num_vehicles'].sum()} vehicles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _validate_data(self):
        """Validate the loaded data"""
        # Check for required columns
        gvp_required = ['s_no', 'location', 'longitude', 'latitude', 'waste_generation']
        sctp_required = ['s_no', 'transfer_station', 'longitude', 'latitude']
        fleet_required = ['vehicle_type', 'capacity_tonnes', 'num_vehicles']
        
        for col in gvp_required:
            if col not in self.gvp_df.columns:
                raise ValueError(f"Missing column in GVPs: {col}")
        
        for col in sctp_required:
            if col not in self.sctp_df.columns:
                raise ValueError(f"Missing column in SCTPs: {col}")
        
        for col in fleet_required:
            if col not in self.fleet_df.columns:
                raise ValueError(f"Missing column in Fleet: {col}")
        
        # Check for valid coordinates
        if self.gvp_df[['longitude', 'latitude']].isnull().any().any():
            print("⚠️  Warning: Some GVPs have missing coordinates")
        
        if self.sctp_df[['longitude', 'latitude']].isnull().any().any():
            print("⚠️  Warning: Some SCTPs have missing coordinates")
    
    def calculate_distance_matrix(self):
        """Calculate distance matrix between all GVPs and SCTPs"""
        print("\n📏 Calculating distance matrix...")
        
        n_gvp = len(self.gvp_df)
        n_sctp = len(self.sctp_df)
        
        # Initialize matrix
        self.distance_matrix = np.zeros((n_gvp, n_sctp))
        
        # Progress tracking
        total_calculations = n_gvp * n_sctp
        progress_interval = max(1, total_calculations // 20)  # Update 20 times
        
        for i, gvp in enumerate(self.gvp_df.itertuples()):
            gvp_coords = (gvp.latitude, gvp.longitude)
            
            for j, sctp in enumerate(self.sctp_df.itertuples()):
                sctp_coords = (sctp.latitude, sctp.longitude)
                
                # Calculate geodesic distance
                try:
                    distance_km = geodesic(gvp_coords, sctp_coords).km
                    self.distance_matrix[i, j] = distance_km
                except:
                    self.distance_matrix[i, j] = np.nan
                
                # Progress update
                if (i * n_sctp + j) % progress_interval == 0:
                    progress = (i * n_sctp + j) / total_calculations * 100
                    print(f"   Progress: {progress:.1f}%", end='\r')
        
        print(f"✅ Distance matrix created: {n_gvp} GVPs × {n_sctp} SCTPs")
        
        # Save distance matrix
        dist_df = pd.DataFrame(
            self.distance_matrix,
            columns=[f"SCTP_{i+1}" for i in range(n_sctp)]
        )
        dist_df.insert(0, 'GVP_ID', self.gvp_df['s_no'].values)
        dist_path = os.path.join(OUTPUT_DIR, 'distance_matrix.xlsx')
        dist_df.to_excel(dist_path, index=False)
        print(f"💾 Distance matrix saved to: {dist_path}")
        
        return self.distance_matrix
    
    def assign_to_nearest_sctp(self):
        """Assign each GVP to its nearest SCTP"""
        print("\n📍 Assigning GVPs to nearest SCTPs...")
        
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated")
        
        # Find nearest SCTP for each GVP
        nearest_indices = np.nanargmin(self.distance_matrix, axis=1)
        nearest_distances = np.take_along_axis(
            self.distance_matrix, 
            nearest_indices[:, None], 
            axis=1
        ).flatten()
        
        # Add assignment information to GVP dataframe
        self.gvp_df['nearest_sctp_index'] = nearest_indices
        self.gvp_df['nearest_sctp_distance_km'] = nearest_distances
        
        # Get SCTP names
        sctp_names = self.sctp_df['transfer_station'].values
        self.gvp_df['assigned_sctp'] = [sctp_names[i] for i in nearest_indices]
        
        # Also get SCTP coordinates for reference
        self.gvp_df['assigned_sctp_lat'] = [self.sctp_df.iloc[i]['latitude'] for i in nearest_indices]
        self.gvp_df['assigned_sctp_lon'] = [self.sctp_df.iloc[i]['longitude'] for i in nearest_indices]
        
        print(f"✅ Assignment complete")
        print(f"   Average distance: {nearest_distances.mean():.2f} km")
        print(f"   Maximum distance: {nearest_distances.max():.2f} km")
        print(f"   Minimum distance: {nearest_distances.min():.2f} km")
        
        return self.gvp_df
    
    def create_clusters(self):
        """Create clusters of GVPs for each SCTP"""
        print("\n👥 Creating clusters...")
        
        self.clusters = {}
        cluster_stats = []
        
        for sctp_name in self.sctp_df['transfer_station']:
            # Get GVPs assigned to this SCTP
            cluster_gvps = self.gvp_df[self.gvp_df['assigned_sctp'] == sctp_name].copy()
            
            if len(cluster_gvps) > 0:
                # Calculate cluster statistics
                total_waste = cluster_gvps['waste_generation'].sum()
                avg_distance = cluster_gvps['nearest_sctp_distance_km'].mean()
                max_distance = cluster_gvps['nearest_sctp_distance_km'].max()
                
                # Calculate cluster centroid
                centroid_lat = cluster_gvps['latitude'].mean()
                centroid_lon = cluster_gvps['longitude'].mean()
                
                # Get the SCTP coordinates
                sctp_row = self.sctp_df[self.sctp_df['transfer_station'] == sctp_name].iloc[0]
                
                # Store cluster information
                self.clusters[sctp_name] = {
                    'gvps': cluster_gvps,
                    'count': len(cluster_gvps),
                    'total_waste': total_waste,
                    'avg_distance': avg_distance,
                    'max_distance': max_distance,
                    'centroid_lat': centroid_lat,
                    'centroid_lon': centroid_lon,
                    'sctp_lat': sctp_row['latitude'],
                    'sctp_lon': sctp_row['longitude']
                }
                
                cluster_stats.append({
                    'SCTP': sctp_name,
                    'GVPs_Count': len(cluster_gvps),
                    'Total_Waste': total_waste,
                    'Avg_Distance_km': avg_distance,
                    'Max_Distance_km': max_distance,
                    'Centroid_Lat': centroid_lat,
                    'Centroid_Lon': centroid_lon
                })
        
        # Create clusters dataframe
        self.clusters_df = pd.DataFrame(cluster_stats)
        
        print(f"✅ Created {len(self.clusters)} clusters")
        
        # Display cluster summary
        print("\n📊 CLUSTER SUMMARY:")
        print("-" * 60)
        for idx, row in self.clusters_df.iterrows():
            print(f"  {row['SCTP']}:")
            print(f"    • GVPs: {row['GVPs_Count']}")
            print(f"    • Waste: {row['Total_Waste']:.1f} units")
            print(f"    • Avg distance: {row['Avg_Distance_km']:.2f} km")
        
        # Save clusters data
        clusters_path = os.path.join(OUTPUT_DIR, 'clusters_summary.xlsx')
        self.clusters_df.to_excel(clusters_path, index=False)
        print(f"\n💾 Cluster summary saved to: {clusters_path}")
        
        return self.clusters
    
    def calculate_system_statistics(self):
        """Calculate comprehensive system statistics"""
        print("\n📈 Calculating system statistics...")
        
        # Basic statistics
        total_gvps = len(self.gvp_df)
        total_sctps = len(self.sctp_df)
        total_waste = self.gvp_df['waste_generation'].sum()
        
        # Fleet capacity calculations
        total_vehicles = self.fleet_df['num_vehicles'].sum()
        total_capacity = (self.fleet_df['capacity_tonnes'] * self.fleet_df['num_vehicles']).sum()
        
        # Distance statistics
        avg_distance = self.gvp_df['nearest_sctp_distance_km'].mean()
        total_distance_all = self.gvp_df['nearest_sctp_distance_km'].sum()
        
        # Time calculations
        avg_travel_time_hours = avg_distance / self.avg_speed_kmh
        total_service_time_hours = (total_gvps * self.service_time_min) / 60
        total_working_hours_needed = (total_distance_all / self.avg_speed_kmh) + total_service_time_hours
        
        # Vehicle requirements (simplified)
        vehicles_needed = math.ceil(total_working_hours_needed / self.working_hours)
        
        # Cost calculations
        total_fuel_liters = total_distance_all * self.fuel_rate_lpkm
        total_fuel_cost = total_fuel_liters * self.fuel_cost_pl
        
        # Efficiency metrics
        if total_capacity > 0:
            capacity_utilization = (total_waste / total_capacity) * 100
        else:
            capacity_utilization = 0
        
        # Store statistics
        self.statistics = {
            'Basic': {
                'Total_GVPs': total_gvps,
                'Total_SCTPs': total_sctps,
                'Total_Waste_units_per_day': total_waste,
                'Total_Vehicles': total_vehicles,
                'Total_Capacity_tonnes': total_capacity
            },
            'Distance': {
                'Average_Distance_km': avg_distance,
                'Total_Distance_All_GVPs_km': total_distance_all,
                'Maximum_Distance_km': self.gvp_df['nearest_sctp_distance_km'].max(),
                'Minimum_Distance_km': self.gvp_df['nearest_sctp_distance_km'].min()
            },
            'Time': {
                'Average_Travel_Time_hours': avg_travel_time_hours,
                'Total_Service_Time_hours': total_service_time_hours,
                'Total_Working_Hours_Needed': total_working_hours_needed,
                'Vehicles_Needed': vehicles_needed
            },
            'Cost': {
                'Total_Fuel_Liters': total_fuel_liters,
                'Total_Fuel_Cost': total_fuel_cost,
                'Fuel_Cost_per_GVP': total_fuel_cost / total_gvps if total_gvps > 0 else 0
            },
            'Efficiency': {
                'Capacity_Utilization_%': capacity_utilization,
                'GVPs_per_SCTP': total_gvps / total_sctps if total_sctps > 0 else 0,
                'Waste_per_GVP': total_waste / total_gvps if total_gvps > 0 else 0
            }
        }
        
        # Print statistics
        print("\n📊 SYSTEM STATISTICS:")
        print("=" * 60)
        
        for category, metrics in self.statistics.items():
            print(f"\n{category}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        return self.statistics
    
    def visualize_clusters(self):
        """Create comprehensive visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cluster Map (Main visualization)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_cluster_map(ax1)
        
        # 2. Waste Distribution by Cluster
        ax2 = plt.subplot(2, 3, 2)
        self._plot_waste_distribution(ax2)
        
        # 3. Distance Distribution
        ax3 = plt.subplot(2, 3, 3)
        self._plot_distance_distribution(ax3)
        
        # 4. Cluster Size Distribution
        ax4 = plt.subplot(2, 3, 4)
        self._plot_cluster_sizes(ax4)
        
        # 5. Fleet Capacity Analysis
        ax5 = plt.subplot(2, 3, 5)
        self._plot_fleet_analysis(ax5)
        
        # 6. Efficiency Metrics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_efficiency_metrics(ax6)
        
        # Adjust layout and save
        plt.suptitle('Waste Management System Analysis - Distance & Clustering', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the main visualization
        viz_path = os.path.join(OUTPUT_DIR, 'comprehensive_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualization saved to: {viz_path}")
        
        # Create separate detailed map
        self._create_detailed_map()
        
        # Create interactive HTML map (optional)
        self._create_interactive_map()
        
        plt.show()
    
    def _plot_cluster_map(self, ax):
        """Plot cluster map"""
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.clusters)))
        
        for idx, (sctp_name, cluster_info) in enumerate(self.clusters.items()):
            color = colors[idx]
            cluster_gvps = cluster_info['gvps']
            
            # Plot GVPs
            ax.scatter(cluster_gvps['longitude'], cluster_gvps['latitude'],
                      c=[color], s=cluster_gvps['waste_generation'] * 5,
                      alpha=0.7, label=sctp_name,
                      edgecolors='black', linewidth=0.5)
            
            # Plot SCTP
            ax.scatter(cluster_info['sctp_lon'], cluster_info['sctp_lat'],
                      c=[color], marker='*', s=500,
                      edgecolors='black', linewidth=2)
            
            # Add SCTP label
            ax.annotate(sctp_name, 
                       (cluster_info['sctp_lon'], cluster_info['sctp_lat']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Waste Collection Clusters')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_waste_distribution(self, ax):
        """Plot waste distribution by cluster"""
        cluster_names = list(self.clusters.keys())
        waste_amounts = [cluster['total_waste'] for cluster in self.clusters.values()]
        
        bars = ax.bar(range(len(cluster_names)), waste_amounts, color=plt.cm.tab20.colors)
        ax.set_xlabel('SCTP Clusters')
        ax.set_ylabel('Total Waste (units)')
        ax.set_title('Waste Distribution by Cluster')
        ax.set_xticks(range(len(cluster_names)))
        ax.set_xticklabels(cluster_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_distance_distribution(self, ax):
        """Plot distance distribution"""
        distances = self.gvp_df['nearest_sctp_distance_km'].dropna()
        
        ax.hist(distances, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(distances.mean(), color='red', linestyle='--', 
                  label=f'Mean: {distances.mean():.2f} km')
        
        ax.set_xlabel('Distance to SCTP (km)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance Distribution of GVPs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cluster_sizes(self, ax):
        """Plot cluster sizes"""
        cluster_names = list(self.clusters.keys())
        cluster_sizes = [cluster['count'] for cluster in self.clusters.values()]
        
        wedges, texts, autotexts = ax.pie(cluster_sizes, labels=cluster_names, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 8})
        ax.set_title('Distribution of GVPs across Clusters')
        
        # Make autotexts white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_fleet_analysis(self, ax):
        """Plot fleet capacity analysis"""
        vehicle_types = self.fleet_df['vehicle_type']
        capacities = self.fleet_df['capacity_tonnes']
        counts = self.fleet_df['num_vehicles']
        
        x = range(len(vehicle_types))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], capacities, width, label='Capacity (tonnes)', alpha=0.7)
        ax.bar([i + width/2 for i in x], counts, width, label='Number of Vehicles', alpha=0.7)
        
        ax.set_xlabel('Vehicle Type')
        ax.set_ylabel('Value')
        ax.set_title('Fleet Capacity Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_metrics(self, ax):
        """Plot efficiency metrics"""
        metrics = ['Capacity Utilization', 'GVPs per SCTP', 'Avg Distance']
        values = [
            self.statistics['Efficiency']['Capacity_Utilization_%'],
            self.statistics['Efficiency']['GVPs_per_SCTP'],
            self.statistics['Distance']['Average_Distance_km']
        ]
        
        bars = ax.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_ylabel('Value')
        ax.set_title('System Efficiency Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
    
    def _create_detailed_map(self):
        """Create a detailed map visualization"""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot Hyderabad boundary (approximate)
        hyderabad_lat = [17.3, 17.6, 17.6, 17.3, 17.3]
        hyderabad_lon = [78.3, 78.3, 78.6, 78.6, 78.3]
        ax.fill(hyderabad_lon, hyderabad_lat, alpha=0.1, color='blue', label='Hyderabad Area')
        
        # Plot clusters
        for sctp_name, cluster_info in self.clusters.items():
            color = np.random.rand(3,)
            cluster_gvps = cluster_info['gvps']
            
            # Plot GVP connections to SCTP
            for _, gvp in cluster_gvps.iterrows():
                ax.plot([gvp['longitude'], cluster_info['sctp_lon']],
                       [gvp['latitude'], cluster_info['sctp_lat']],
                       color=color, alpha=0.2, linewidth=0.5)
            
            # Plot GVPs
            ax.scatter(cluster_gvps['longitude'], cluster_gvps['latitude'],
                      c=[color], s=cluster_gvps['waste_generation'] * 3,
                      alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Plot SCTP
            ax.scatter(cluster_info['sctp_lon'], cluster_info['sctp_lat'],
                      c=[color], marker='*', s=400,
                      edgecolors='black', linewidth=2, label=sctp_name)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Detailed Waste Collection Network\n(Lines show GVP to SCTP connections)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save detailed map
        detailed_map_path = os.path.join(OUTPUT_DIR, 'detailed_network_map.png')
        plt.tight_layout()
        plt.savefig(detailed_map_path, dpi=300, bbox_inches='tight')
        print(f"✅ Detailed network map saved to: {detailed_map_path}")
        
        plt.close(fig)
    
    def _create_interactive_map(self):
        """Create interactive HTML map using folium (optional)"""
        try:
            import folium
            
            # Create base map centered on Hyderabad
            hyderabad_center = [17.3850, 78.4867]
            m = folium.Map(location=hyderabad_center, zoom_start=12)
            
            # Add SCTP markers
            for _, sctp in self.sctp_df.iterrows():
                folium.Marker(
                    location=[sctp['latitude'], sctp['longitude']],
                    popup=f"<b>{sctp['transfer_station']}</b><br>"
                          f"Coordinates: {sctp['latitude']:.6f}, {sctp['longitude']:.6f}",
                    icon=folium.Icon(color='red', icon='industry', prefix='fa')
                ).add_to(m)
            
            # Add GVP markers by cluster
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
            
            for idx, (sctp_name, cluster_info) in enumerate(self.clusters.items()):
                color = colors[idx % len(colors)]
                
                for _, gvp in cluster_info['gvps'].iterrows():
                    folium.CircleMarker(
                        location=[gvp['latitude'], gvp['longitude']],
                        radius=5 + (gvp['waste_generation'] / 10),
                        popup=f"<b>GVP {gvp['s_no']}</b><br>"
                              f"Location: {gvp['location']}<br>"
                              f"Waste: {gvp['waste_generation']} units<br>"
                              f"Distance to SCTP: {gvp['nearest_sctp_distance_km']:.2f} km",
                        color=color,
                        fill=True,
                        fill_color=color
                    ).add_to(m)
            
            # Save interactive map
            html_path = os.path.join(OUTPUT_DIR, 'interactive_map.html')
            m.save(html_path)
            print(f"✅ Interactive map saved to: {html_path}")
            print(f"   Open this file in a web browser to view the interactive map")
            
        except ImportError:
            print("⚠️  Folium not installed. Install with: pip install folium")
            print("   Skipping interactive map generation...")
    
    def save_all_results(self):
        """Save all analysis results to files"""
        print("\n💾 Saving all results...")
        
        # 1. Save clustered GVPs
        gvp_output = os.path.join(OUTPUT_DIR, 'clustered_gvps.xlsx')
        self.gvp_df.to_excel(gvp_output, index=False)
        print(f"✅ Clustered GVPs saved to: {gvp_output}")
        
        # 2. Save cluster statistics
        cluster_stats_output = os.path.join(OUTPUT_DIR, 'cluster_statistics.xlsx')
        self.clusters_df.to_excel(cluster_stats_output, index=False)
        
        # 3. Save system statistics
        stats_output = os.path.join(OUTPUT_DIR, 'system_statistics.xlsx')
        
        # Convert statistics dictionary to DataFrame
        stats_rows = []
        for category, metrics in self.statistics.items():
            for metric, value in metrics.items():
                stats_rows.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': value
                })
        
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_excel(stats_output, index=False)
        print(f"✅ System statistics saved to: {stats_output}")
        
        # 4. Save analysis report
        self._save_analysis_report()
        
        print(f"\n📁 All outputs saved in: {OUTPUT_DIR}")
    
    def _save_analysis_report(self):
        """Save a comprehensive analysis report"""
        report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WASTE MANAGEMENT SYSTEM - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"• Total GVPs analyzed: {len(self.gvp_df)}\n")
            f.write(f"• Total waste per day: {self.gvp_df['waste_generation'].sum():.1f} units\n")
            f.write(f"• Number of SCTPs: {len(self.sctp_df)}\n")
            f.write(f"• Total vehicles available: {self.fleet_df['num_vehicles'].sum()}\n")
            f.write(f"• Average distance to SCTP: {self.gvp_df['nearest_sctp_distance_km'].mean():.2f} km\n\n")
            
            f.write("CLUSTER ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for sctp_name, cluster_info in self.clusters.items():
                f.write(f"\n{sctp_name}:\n")
                f.write(f"  • GVPs assigned: {cluster_info['count']}\n")
                f.write(f"  • Total waste: {cluster_info['total_waste']:.1f} units\n")
                f.write(f"  • Average distance: {cluster_info['avg_distance']:.2f} km\n")
                f.write(f"  • Maximum distance: {cluster_info['max_distance']:.2f} km\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS FOR ROUTE OPTIMIZATION\n")
            f.write("=" * 80 + "\n")
            
            recommendations = [
                "1. Prioritize clusters with high waste volume for morning collection",
                "2. Consider vehicle capacity when assigning routes to clusters",
                "3. Optimize routes within each cluster using VRP algorithms",
                "4. Balance workload across vehicles for efficient utilization",
                "5. Consider time windows for commercial vs residential areas",
                "6. Implement dynamic routing based on real-time waste levels",
                "7. Use smaller vehicles for clusters with scattered GVPs",
                "8. Consider establishing additional SCTPs in remote clusters"
            ]
            
            for rec in recommendations:
                f.write(f"• {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("NEXT STEPS FOR MODULE 5\n")
            f.write("=" * 80 + "\n")
            f.write("1. Use the clusters defined in this analysis\n")
            f.write("2. Apply Vehicle Routing Problem (VRP) algorithms\n")
            f.write("3. Consider vehicle capacity constraints\n")
            f.write("4. Optimize routes within each cluster\n")
            f.write("5. Create daily schedules for each vehicle\n")
            f.write("6. Calculate cost savings from optimized routes\n")
        
        print(f"✅ Analysis report saved to: {report_path}")

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
def main():
    """Main execution function"""
    print("=" * 80)
    print("MODULE 4: DISTANCE ANALYSIS & CLUSTERING")
    print("=" * 80)
    print("This module will:")
    print("1. Calculate distances between GVPs and SCTPs")
    print("2. Assign each GVP to the nearest SCTP")
    print("3. Create optimized clusters for vehicle routing")
    print("4. Generate comprehensive visualizations")
    print("5. Save all results for Module 5")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = DistanceAnalyzer()
    
    try:
        # Step 1: Load data
        print("\n" + "=" * 80)
        print("STEP 1: LOADING TRANSFORMED DATA")
        print("=" * 80)
        if not analyzer.load_data():
            return
        
        # Step 2: Calculate distance matrix
        print("\n" + "=" * 80)
        print("STEP 2: CALCULATING DISTANCE MATRIX")
        print("=" * 80)
        analyzer.calculate_distance_matrix()
        
        # Step 3: Assign GVPs to SCTPs
        print("\n" + "=" * 80)
        print("STEP 3: ASSIGNING GVPS TO NEAREST SCTPS")
        print("=" * 80)
        analyzer.assign_to_nearest_sctp()
        
        # Step 4: Create clusters
        print("\n" + "=" * 80)
        print("STEP 4: CREATING OPTIMIZED CLUSTERS")
        print("=" * 80)
        analyzer.create_clusters()
        
        # Step 5: Calculate statistics
        print("\n" + "=" * 80)
        print("STEP 5: CALCULATING SYSTEM STATISTICS")
        print("=" * 80)
        analyzer.calculate_system_statistics()
        
        # Step 6: Create visualizations
        print("\n" + "=" * 80)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("=" * 80)
        analyzer.visualize_clusters()
        
        # Step 7: Save results
        print("\n" + "=" * 80)
        print("STEP 7: SAVING ALL RESULTS")
        print("=" * 80)
        analyzer.save_all_results()
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 MODULE 4 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📁 OUTPUTS GENERATED:")
        print("   • clustered_gvps.xlsx - GVPs with assigned SCTPs")
        print("   • distance_matrix.xlsx - All GVP-SCTP distances")
        print("   • clusters_summary.xlsx - Cluster statistics")
        print("   • cluster_statistics.xlsx - Detailed cluster data")
        print("   • system_statistics.xlsx - Overall system metrics")
        print("   • comprehensive_analysis.png - Visualizations")
        print("   • detailed_network_map.png - Network connections")
        print("   • analysis_report.txt - Comprehensive report")
        print("   • interactive_map.html (if folium installed)")
        
        print("\n➡️  NEXT STEP: MODULE 5 - VEHICLE ROUTING OPTIMIZATION")
        print("   Use the clustered data to optimize routes within each cluster!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------------------------
# INSTALLATION CHECK
# ---------------------------------------------------
def check_installations():
    """Check and install required packages"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'geopy': 'geopy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    print("🔍 Checking required packages...")
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            install = input(f"Install {package}? (y/n): ")
            if install.lower() == 'y':
                import subprocess
                subprocess.check_call(['pip', 'install', package])
                print(f"✅ {package} installed successfully")
    
    print("\n" + "=" * 80)
    print("📦 All required packages are ready!")
    print("=" * 80)

# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    # Check installations first
    check_installations()
    
    # Run main analysis
    main()