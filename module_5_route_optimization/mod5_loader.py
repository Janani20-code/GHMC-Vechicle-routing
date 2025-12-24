"""
MODULE 5.1: DATA LOADER
Loads cleaned data from Module 4.5 outputs
"""

import pandas as pd
import os
from typing import Dict, Tuple

class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.data = {}
        self.metadata = {}
        
    def load_cleaned_data(self) -> Dict:
        """
        Load all necessary data files from Module 4.5 outputs
        Returns: Dictionary containing all loaded data
        """
        print("📂 Loading Module 4.5 cleaned data...")
        
        # Define file paths
        cleaned_gvp_path = os.path.join(self.base_path, "module_4_corrected", "cleaned_gvps_for_routing.xlsx")
        cluster_summary_path = os.path.join(self.base_path, "module_4_corrected", "updated_clusters_summary.xlsx")
        original_fleet_path = os.path.join(self.base_path, "transformed_data.xlsx")
        
        try:
            # Load cleaned GVPs
            self.data['gvps'] = pd.read_excel(cleaned_gvp_path)
            print(f"✅ Loaded {len(self.data['gvps'])} cleaned GVPs")
            
            # Load cluster summary
            self.data['clusters'] = pd.read_excel(cluster_summary_path)
            print(f"✅ Loaded {len(self.data['clusters'])} cluster summaries")
            
            # Load fleet data
            self.data['fleet'] = pd.read_excel(original_fleet_path, sheet_name='Fleet')
            print(f"✅ Loaded {len(self.data['fleet'])} vehicle types")
            
            # Store metadata
            self.metadata['total_waste_tonnes'] = self.data['gvps']['waste_tonnes'].sum()
            self.metadata['total_gvps'] = len(self.data['gvps'])
            self.metadata['total_clusters'] = len(self.data['clusters'])
            self.metadata['total_fleet_capacity'] = (
                self.data['fleet']['capacity_tonnes'] * self.data['fleet']['num_vehicles']
            ).sum()
            
            print("\n📊 DATA SUMMARY:")
            print(f"   • Total GVPs: {self.metadata['total_gvps']}")
            print(f"   • Total waste: {self.metadata['total_waste_tonnes']:.2f} tonnes")
            print(f"   • Fleet capacity: {self.metadata['total_fleet_capacity']:.2f} tonnes")
            
            return self.data
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def validate_data(self) -> bool:
        """
        Validate loaded data for vehicle allocation
        Returns: True if data is valid, False otherwise
        """
        if not self.data:
            print("❌ No data loaded")
            return False
        
        required_columns = {
            'gvps': ['s_no', 'location', 'latitude', 'longitude', 'waste_tonnes', 'assigned_sctp'],
            'clusters': ['SCTP', 'GVPs_Count', 'Total_Waste_Tonnes'],
            'fleet': ['vehicle_type', 'capacity_tonnes', 'num_vehicles']
        }
        
        for dataset, columns in required_columns.items():
            if dataset in self.data:
                missing = [col for col in columns if col not in self.data[dataset].columns]
                if missing:
                    print(f"❌ Missing columns in {dataset}: {missing}")
                    return False
        
        # Check for negative values
        if (self.data['gvps']['waste_tonnes'] < 0).any():
            print("❌ Negative waste values found in GVPs")
            return False
            
        if (self.data['fleet']['capacity_tonnes'] < 0).any():
            print("❌ Negative capacity values found in fleet")
            return False
        
        print("✅ Data validation passed")
        return True
    
    def get_cluster_gvps(self, cluster_name: str) -> pd.DataFrame:
        """
        Get GVPs belonging to a specific cluster
        """
        return self.data['gvps'][self.data['gvps']['assigned_sctp'] == cluster_name].copy()
    
    def save_loaded_data(self, output_path: str):
        """
        Save loaded data for reference
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save loaded data summary
        summary_file = os.path.join(output_path, "loaded_data_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODULE 5 - LOADED DATA SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. GVP DATA\n")
            f.write(f"   • Total GVPs: {len(self.data['gvps'])}\n")
            f.write(f"   • Total waste: {self.data['gvps']['waste_tonnes'].sum():.2f} tonnes\n")
            f.write(f"   • Clusters represented: {self.data['gvps']['assigned_sctp'].nunique()}\n\n")
            
            f.write("2. CLUSTER DATA\n")
            for idx, row in self.data['clusters'].iterrows():
                f.write(f"   • {row['SCTP']}: {row['GVPs_Count']} GVPs, {row['Total_Waste_Tonnes']:.2f} tonnes\n")
            
            f.write(f"\n3. FLEET DATA\n")
            f.write(f"   • Vehicle types: {len(self.data['fleet'])}\n")
            f.write(f"   • Total vehicles: {self.data['fleet']['num_vehicles'].sum()}\n")
            f.write(f"   • Total capacity: {self.metadata['total_fleet_capacity']:.2f} tonnes\n")
        
        print(f"✅ Loaded data summary saved to: {summary_file}")
        return summary_file