"""
MODULE 5 MAIN: VEHICLE ALLOCATION & TRIP SCHEDULING
Orchestrates all sub-modules for complete vehicle allocation and scheduling
"""

import os
import sys
from datetime import datetime

# Add current directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mod5_loader import DataLoader
from mod5_allocator import VehicleAllocator
from mod5_router import RouteOptimizer
from mod5_scheduler import TripScheduler
from mod5_visualizer import RouteVisualizer

class WasteRoutingSystem:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.output_base = os.path.join(base_path, "module_5_outputs")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_base, self.timestamp)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "step_1_loaded_data"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "step_2_allocation"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "step_3_routing"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "step_4_scheduling"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "step_5_visualizations"), exist_ok=True)
        
        self.data_loader = None
        self.allocator = None
        self.router = None
        self.scheduler = None
        self.visualizer = None
        
    def run_pipeline(self):
        """Execute complete vehicle allocation and scheduling pipeline"""
        print("=" * 80)
        print("MODULE 5: VEHICLE ALLOCATION & TRIP SCHEDULING")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load Data
        print("\n" + "=" * 80)
        print("STEP 1: LOADING CLEANED DATA FROM MODULE 4.5")
        print("=" * 80)
        self.data_loader = DataLoader(self.base_path)
        data = self.data_loader.load_cleaned_data()
        
        if not data or not self.data_loader.validate_data():
            print("❌ Failed to load or validate data. Exiting.")
            return False
        
        # Save loaded data summary
        self.data_loader.save_loaded_data(
            os.path.join(self.output_dir, "step_1_loaded_data")
        )
        
        # Step 2: Vehicle Allocation
        print("\n" + "=" * 80)
        print("STEP 2: VEHICLE ALLOCATION TO CLUSTERS")
        print("=" * 80)
        self.allocator = VehicleAllocator(self.data_loader)
        self.allocator.create_vehicle_pool()
        allocation = self.allocator.allocate_vehicles_to_clusters()
        
        # Save allocation results
        self.allocator.save_allocation_results(
            os.path.join(self.output_dir, "step_2_allocation")
        )
        
        # Step 3: Route Optimization
        print("\n" + "=" * 80)
        print("STEP 3: ROUTE OPTIMIZATION WITHIN CLUSTERS")
        print("=" * 80)
        self.router = RouteOptimizer(self.data_loader, self.allocator)
        routes = self.router.optimize_all_clusters(
            os.path.join(self.output_dir, "step_3_routing")
        )
        
        # Step 4: Trip Scheduling
        print("\n" + "=" * 80)
        print("STEP 4: TRIP SCHEDULING & TIMING")
        print("=" * 80)
        self.scheduler = TripScheduler(self.router)
        schedules = self.scheduler.create_schedules()
        
        # Save scheduling results
        self.scheduler.save_schedules(
            os.path.join(self.output_dir, "step_4_scheduling")
        )
        
        # Step 5: Visualization
        print("\n" + "=" * 80)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("=" * 80)
        self.visualizer = RouteVisualizer(self.data_loader, self.router, self.scheduler)
        self.visualizer.set_output_path(
            os.path.join(self.output_dir, "step_5_visualizations")
        )
        self.visualizer.create_all_visualizations()
        
        # Final Report
        self._create_final_report()
        
        return True
    
    def _create_final_report(self):
        """Create final summary report"""
        report_path = os.path.join(self.output_dir, "FINAL_REPORT.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WASTE COLLECTION OPTIMIZATION SYSTEM - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROJECT OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write("• Challenge: nVisionX'26 Design Challenge by IIT Hyderabad\n")
            f.write("• Problem: Optimize waste collection routing for GHMC\n")
            f.write("• Submission: Vehicle Allocation & Trip Scheduling System\n")
            f.write(f"• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SYSTEM PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            # Get metrics from scheduler
            if self.scheduler:
                metrics = self.scheduler.calculate_efficiency_metrics()
                overall = metrics['overall']
                
                f.write(f"• Total vehicles scheduled: {overall['total_vehicles_scheduled']}\n")
                f.write(f"• Total trips scheduled: {overall['total_trips_scheduled']}\n")
                f.write(f"• Total waste to collect: {overall['total_waste_collected']:.2f} tonnes\n")
                f.write(f"• Total distance to travel: {overall['total_distance_traveled']:.2f} km\n")
                f.write(f"• Total scheduled hours: {overall['total_scheduled_hours']:.2f}\n")
                f.write(f"• Average vehicle utilization: {overall['avg_vehicle_utilization']:.1f}%\n\n")
            
            f.write("KEY ACHIEVEMENTS\n")
            f.write("-" * 80 + "\n")
            f.write("1. Smart Vehicle Allocation: Vehicles assigned based on cluster waste volume\n")
            f.write("2. Efficient Routing: Nearest-neighbor algorithm with capacity constraints\n")
            f.write("3. Optimized Scheduling: Time-based scheduling with working hour constraints\n")
            f.write("4. Cost Efficiency: Minimized travel distance and maximized vehicle utilization\n")
            f.write("5. Scalable Design: Modular architecture for easy maintenance\n\n")
            
            f.write("TECHNICAL APPROACH\n")
            f.write("-" * 80 + "\n")
            f.write("1. Data Cleaning (Module 4.5):\n")
            f.write("   • Removed outliers (>15km distance)\n")
            f.write("   • Standardized waste units (1 unit = 0.5 tonnes)\n")
            f.write("   • Validated coordinate ranges for Hyderabad\n\n")
            
            f.write("2. Vehicle Allocation:\n")
            f.write("   • Capacity-based allocation algorithm\n")
            f.write("   • Minimum 60% vehicle utilization constraint\n")
            f.write("   • Backup vehicle distribution for peak loads\n\n")
            
            f.write("3. Route Optimization:\n")
            f.write("   • Nearest-neighbor algorithm with capacity constraints\n")
            f.write("   • Cluster-wise route optimization\n")
            f.write("   • Distance minimization with waste volume consideration\n\n")
            
            f.write("4. Trip Scheduling:\n")
            f.write("   • Time-based scheduling within working hours\n")
            f.write("   • Service time and travel time calculations\n")
            f.write("   • Break time consideration between trips\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write("Step 1 (Loaded Data):\n")
            f.write("   • loaded_data_summary.txt\n\n")
            
            f.write("Step 2 (Allocation):\n")
            f.write("   • vehicle_allocation.xlsx\n")
            f.write("   • trip_requirements.xlsx\n")
            f.write("   • allocation_summary.txt\n\n")
            
            f.write("Step 3 (Routing):\n")
            f.write("   • route_details.xlsx\n")
            f.write("   • route_summaries.xlsx\n")
            f.write("   • route_optimization_report.txt\n\n")
            
            f.write("Step 4 (Scheduling):\n")
            f.write("   • vehicle_schedules.xlsx\n")
            f.write("   • vehicle_summary.xlsx\n")
            f.write("   • efficiency_metrics.txt\n\n")
            
            f.write("Step 5 (Visualizations):\n")
            f.write("   • optimization_dashboard.png\n")
            f.write("   • cluster_*_routes.png (for each cluster)\n")
            f.write("   • schedule_heatmap.png\n\n")
            
            f.write("RECOMMENDATIONS FOR IMPLEMENTATION\n")
            f.write("-" * 80 + "\n")
            f.write("1. Start with high-waste clusters (Ziyaguda, IBT) in morning hours\n")
            f.write("2. Monitor actual waste generation and adjust routes dynamically\n")
            f.write("3. Use GPS tracking for real-time route optimization\n")
            f.write("4. Implement feedback mechanism for route efficiency\n")
            f.write("5. Consider seasonal variations in waste generation\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✅ Final report saved to: {report_path}")
        
        # Create quick summary
        summary_path = os.path.join(self.output_dir, "QUICK_SUMMARY.txt")
        with open(summary_path, 'w') as f:
            f.write("QUICK SUMMARY - Key Metrics:\n")
            f.write("=" * 40 + "\n")
            if self.scheduler:
                metrics = self.scheduler.calculate_efficiency_metrics()
                overall = metrics['overall']
                
                f.write(f"Vehicles: {overall['total_vehicles_scheduled']}\n")
                f.write(f"Trips: {overall['total_trips_scheduled']}\n")
                f.write(f"Waste: {overall['total_waste_collected']:.1f} tonnes\n")
                f.write(f"Distance: {overall['total_distance_traveled']:.1f} km\n")
                f.write(f"Hours: {overall['total_scheduled_hours']:.1f}\n")
                f.write(f"Utilization: {overall['avg_vehicle_utilization']:.1f}%\n")
        
        print(f"✅ Quick summary saved to: {summary_path}")

def main():
    """Main execution function"""
    # Define base path (same as your previous modules)
    BASE_PATH = r"E:\vehicle_routing_system\data"
    
    # Create and run the system
    system = WasteRoutingSystem(BASE_PATH)
    
    try:
        success = system.run_pipeline()
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 MODULE 5 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📁 ALL OUTPUTS SAVED IN: {system.output_dir}")
            print("\n📋 KEY FILES GENERATED:")
            print("   1. vehicle_allocation.xlsx - Which vehicles go to which clusters")
            print("   2. route_details.xlsx - Detailed routes for each vehicle")
            print("   3. vehicle_schedules.xlsx - Complete daily schedule")
            print("   4. optimization_dashboard.png - Visual summary")
            print("   5. FINAL_REPORT.txt - Comprehensive analysis")
            
            print("\n➡️  NEXT STEPS FOR COMPETITION:")
            print("   • Review the FINAL_REPORT.txt for key insights")
            print("   • Use visualizations in your presentation")
            print("   • Highlight efficiency improvements in your submission")
            print("   • Prepare to explain your algorithm choices to judges")
        else:
            print("\n❌ Module 5 failed to complete. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()