import pandas as pd
import re
import os
import numpy as np
from datetime import datetime

# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------
BASE_DIR = r"E:\vehicle_routing_system\data"
INPUT_FILE = os.path.join(BASE_DIR, "cleaned_data.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "transformed_data.xlsx")
PROFILE_FILE = os.path.join(BASE_DIR, "data_profile_report.txt")

# ---------------------------------------------------
# DATA PROFILING FUNCTIONS
# ---------------------------------------------------
def create_data_profile(df, sheet_name):
    """Create detailed data profile for a DataFrame"""
    profile = []
    profile.append(f"\n{'='*70}")
    profile.append(f"SHEET: {sheet_name}")
    profile.append(f"{'='*70}")
    
    # Basic information
    profile.append(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Column details
    profile.append("\n📋 COLUMN DETAILS:")
    profile.append("-" * 70)
    
    for col in df.columns:
        profile.append(f"\n  🔹 {col}:")
        profile.append(f"     Type: {df[col].dtype}")
        
        # Missing values
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        profile.append(f"     Missing: {missing} ({missing_pct:.1f}%)")
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                profile.append(f"     Min: {df[col].min():.6f}")
                profile.append(f"     Max: {df[col].max():.6f}")
                profile.append(f"     Mean: {df[col].mean():.6f}")
                profile.append(f"     Std: {df[col].std():.6f}")
        
        # For object/string columns
        elif df[col].dtype == 'object':
            unique_count = df[col].nunique()
            profile.append(f"     Unique values: {unique_count}")
            if unique_count <= 10:
                unique_values = df[col].dropna().unique()[:10]
                profile.append(f"     Sample values: {list(unique_values)}")
            else:
                top_values = df[col].value_counts().head(5).to_dict()
                profile.append(f"     Top 5 values: {top_values}")
        
        # Sample values
        non_null_sample = df[col].dropna().head(3).tolist()
        if non_null_sample:
            profile.append(f"     Sample: {non_null_sample}")
    
    return "\n".join(profile)

def analyze_coordinate_quality(df, lat_col, lon_col):
    """Analyze coordinate data quality"""
    analysis = []
    
    # Check if columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        return "Coordinate columns not found"
    
    # Convert both columns to numeric, forcing errors to NaN
    lat_series = pd.to_numeric(df[lat_col], errors='coerce')
    lon_series = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Create a DataFrame with the numeric coordinates
    coords = pd.DataFrame({lat_col: lat_series, lon_col: lon_series})
    
    # Count valid coordinates (non-NaN in both)
    valid_coords = coords.dropna()
    valid_count = len(valid_coords)
    total_count = len(df)
    
    analysis.append(f"\n📍 COORDINATE ANALYSIS:")
    analysis.append(f"   Total rows: {total_count}")
    analysis.append(f"   Valid coordinates: {valid_count} ({valid_count/total_count*100:.1f}%)")
    analysis.append(f"   Missing coordinates: {total_count - valid_count}")
    
    if valid_count > 0:
        # Range analysis
        lat_min, lat_max = valid_coords[lat_col].min(), valid_coords[lat_col].max()
        lon_min, lon_max = valid_coords[lon_col].min(), valid_coords[lon_col].max()
        
        analysis.append(f"\n   Coordinate Ranges:")
        analysis.append(f"     Latitude:  {lat_min:.6f} to {lat_max:.6f}")
        analysis.append(f"     Longitude: {lon_min:.6f} to {lon_max:.6f}")
        
        # Check for Hyderabad range
        HYDERABAD_LAT = (17.0, 18.0)
        HYDERABAD_LON = (78.0, 79.0)
        
        in_hyderabad = (
            (valid_coords[lat_col] >= HYDERABAD_LAT[0]) & 
            (valid_coords[lat_col] <= HYDERABAD_LAT[1]) &
            (valid_coords[lon_col] >= HYDERABAD_LON[0]) & 
            (valid_coords[lon_col] <= HYDERABAD_LON[1])
        )
        
        hyderabad_count = in_hyderabad.sum()
        analysis.append(f"\n   Hyderabad Range Check:")
        analysis.append(f"     Within Hyderabad: {hyderabad_count} ({hyderabad_count/valid_count*100:.1f}%)")
        analysis.append(f"     Outside Hyderabad: {valid_count - hyderabad_count}")
        
        # Check for outliers
        lat_outliers = valid_coords[(valid_coords[lat_col] < -90) | (valid_coords[lat_col] > 90)]
        lon_outliers = valid_coords[(valid_coords[lon_col] < -180) | (valid_coords[lon_col] > 180)]
        
        if len(lat_outliers) > 0:
            analysis.append(f"\n   ⚠️  Latitude outliers (outside [-90, 90]): {len(lat_outliers)}")
        if len(lon_outliers) > 0:
            analysis.append(f"   ⚠️  Longitude outliers (outside [-180, 180]): {len(lon_outliers)}")
    
    return "\n".join(analysis)

def analyze_waste_data(df, waste_col):
    """Analyze waste generation data"""
    analysis = []
    
    if waste_col not in df.columns:
        return "Waste column not found"
    
    valid_waste = df[waste_col].dropna()
    analysis.append(f"\n🗑️  WASTE GENERATION ANALYSIS:")
    analysis.append(f"   Total rows with waste data: {len(valid_waste)}")
    
    if len(valid_waste) > 0:
        analysis.append(f"   Total waste: {valid_waste.sum():.2f} units/day")
        analysis.append(f"   Average waste per location: {valid_waste.mean():.2f} units")
        analysis.append(f"   Min waste: {valid_waste.min():.2f}")
        analysis.append(f"   Max waste: {valid_waste.max():.2f}")
        analysis.append(f"   Std deviation: {valid_waste.std():.2f}")
        
        # Distribution analysis
        percentiles = [0, 25, 50, 75, 100]
        perc_values = np.percentile(valid_waste, percentiles)
        analysis.append(f"\n   Percentile Distribution:")
        for p, v in zip(percentiles, perc_values):
            analysis.append(f"     {p}th percentile: {v:.2f}")
    
    return "\n".join(analysis)

def analyze_vehicle_data(df):
    """Analyze vehicle fleet data"""
    analysis = []
    
    analysis.append(f"\n🚚 FLEET ANALYSIS:")
    
    # Check required columns
    required_cols = ['capacity_tonnes', 'num_vehicles']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return f"Missing columns: {missing_cols}"
    
    # Total calculations
    total_vehicles = df['num_vehicles'].sum()
    total_capacity = (df['capacity_tonnes'] * df['num_vehicles']).sum()
    
    analysis.append(f"   Total vehicles: {total_vehicles}")
    analysis.append(f"   Total capacity: {total_capacity:.2f} tonnes")
    analysis.append(f"   Vehicle types: {len(df)}")
    
    # Show each vehicle type details
    analysis.append(f"\n   Vehicle Type Details:")
    for idx, row in df.iterrows():
        vehicle_capacity = row['capacity_tonnes'] * row['num_vehicles']
        analysis.append(f"     {row['vehicle_type']}:")
        analysis.append(f"       Count: {row['num_vehicles']}")
        analysis.append(f"       Capacity each: {row['capacity_tonnes']} tonnes")
        analysis.append(f"       Total capacity: {vehicle_capacity:.1f} tonnes")
    
    return "\n".join(analysis)

# ---------------------------------------------------
# MAIN TRANSFORMATION FUNCTIONS (Your working code)
# ---------------------------------------------------
def transform_gvp_data(df):
    print("\n📍 Transforming GVP Data...")
    initial_count = len(df)
    
    # Show BEFORE profile
    print("\n📋 BEFORE TRANSFORMATION:")
    print(create_data_profile(df, "GVP Raw Data"))
    print(analyze_coordinate_quality(df, 'Latitude', 'Longitude'))
    print(analyze_waste_data(df, 'Estimated Waste Generation'))
    
    # Your transformation logic here (keep it as is)
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Estimated Waste Generation'] = pd.to_numeric(df['Estimated Waste Generation'], errors='coerce')
    
    # Smart swap logic
    swapped_mask = (df['Longitude'] < 40) & (df['Latitude'] > 40)
    if swapped_mask.any():
        df.loc[swapped_mask, ['Latitude', 'Longitude']] = df.loc[swapped_mask, ['Longitude', 'Latitude']].values
    
    # Outlier fix
    df.loc[df['Longitude'] > 180, 'Longitude'] = df.loc[df['Longitude'] > 180, 'Longitude'] / 10**7
    
    # Validation filter
    initial_len = len(df)
    df = df[(df['Latitude'] >= 17.0) & (df['Latitude'] <= 18.0) & 
            (df['Longitude'] >= 78.0) & (df['Longitude'] <= 79.0)]
    
    # Standardize columns
    df = df.rename(columns={
        'S No.': 's_no',
        'Location of the GVPs': 'location',
        'Longitude': 'longitude',
        'Latitude': 'latitude',
        'Estimated Waste Generation': 'waste_generation'
    })
    
    print(f"\n✅ GVP Transformation Complete:")
    print(f"   Initial rows: {initial_count}")
    print(f"   Final rows: {len(df)}")
    print(f"   Removed: {initial_count - len(df)} rows ({((initial_count - len(df))/initial_count*100):.1f}%)")
    
    # Show AFTER profile
    print("\n📋 AFTER TRANSFORMATION:")
    print(create_data_profile(df, "GVP Transformed Data"))
    print(analyze_coordinate_quality(df, 'latitude', 'longitude'))
    print(analyze_waste_data(df, 'waste_generation'))
    
    return df

def transform_vehicle_data(df):
    print("\n🚚 Transforming Vehicle Data...")
    
    # Show BEFORE profile
    print("\n📋 BEFORE TRANSFORMATION:")
    print(create_data_profile(df, "Vehicle Raw Data"))
    
    # Your transformation logic
    df = df.rename(columns={
        'S No.': 's_no',
        'Vehicle Particulars': 'vehicle_type',
        'GVW (Gross Vehicle Weight)': 'gvw',
        'Payload Capacity (in Tonnes)': 'capacity_tonnes',
        'No. of Vehicles Available': 'num_vehicles'
    })
    
    df['gvw'] = df['gvw'].astype(str).str.replace(' T', '', regex=False)
    df['gvw'] = pd.to_numeric(df['gvw'], errors='coerce')
    df['capacity_tonnes'] = pd.to_numeric(df['capacity_tonnes'], errors='coerce')
    df['num_vehicles'] = pd.to_numeric(df['num_vehicles'], errors='coerce')
    
    # Show AFTER profile
    print("\n📋 AFTER TRANSFORMATION:")
    print(create_data_profile(df, "Vehicle Transformed Data"))
    print(analyze_vehicle_data(df))
    
    return df

def transform_sctp_data(df):
    print("\n🏭 Transforming SCTP Data...")
    
    # Show BEFORE profile
    print("\n📋 BEFORE TRANSFORMATION:")
    print(create_data_profile(df, "SCTP Raw Data"))
    
    # Your transformation logic
    def dms_to_decimal(dms_string):
        if pd.isna(dms_string): return None
        dms_string = str(dms_string).strip()
        try: return float(dms_string)
        except ValueError: pass
        
        pattern = r'(\d+)°(\d+)\'([\d\.]+)"([NSEW])'
        match = re.search(pattern, dms_string)
        if match:
            deg, mins, secs, direction = match.groups()
            decimal = float(deg) + float(mins)/60 + float(secs)/3600
            if direction in ['S', 'W']: decimal = -decimal
            return round(decimal, 6)
        return None
    
    def split_and_parse(coord_str):
        if pd.isna(coord_str): return None, None
        parts = str(coord_str).split(',')
        if len(parts) != 2: return None, None
        return dms_to_decimal(parts[0]), dms_to_decimal(parts[1])
    
    coords = df['Coordinates'].apply(split_and_parse)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    df = df.rename(columns={
        'S.No': 's_no', 
        'Transferstation': 'transfer_station'
    })
    
    # Show AFTER profile
    print("\n📋 AFTER TRANSFORMATION:")
    print(create_data_profile(df, "SCTP Transformed Data"))
    print(analyze_coordinate_quality(df, 'latitude', 'longitude'))
    
    return df

def save_comprehensive_report(transformed_data):
    """Save a comprehensive data profiling report"""
    print("\n📝 Generating comprehensive data profile report...")
    
    with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA PROFILE REPORT - Waste Management System\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # System Overview
        f.write("📊 SYSTEM OVERVIEW\n")
        f.write("=" * 80 + "\n")
        
        gvp_df = transformed_data['GVPs']
        fleet_df = transformed_data['Fleet']
        sctp_df = transformed_data['SCTPs']
        
        f.write(f"\n1. GVP Data (Waste Collection Points):\n")
        f.write(f"   - Total locations: {len(gvp_df)}\n")
        f.write(f"   - Total waste per day: {gvp_df['waste_generation'].sum():.2f} units\n")
        
        f.write(f"\n2. Fleet Data:\n")
        total_vehicles = fleet_df['num_vehicles'].sum()
        total_capacity = (fleet_df['capacity_tonnes'] * fleet_df['num_vehicles']).sum()
        f.write(f"   - Total vehicles: {total_vehicles}\n")
        f.write(f"   - Total capacity: {total_capacity:.2f} tonnes\n")
        
        f.write(f"\n3. SCTP Data (Transfer Stations):\n")
        f.write(f"   - Total stations: {len(sctp_df)}\n")
        
        # Efficiency Analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("📈 EFFICIENCY ANALYSIS\n")
        f.write("=" * 80 + "\n")
        
        trips_needed = gvp_df['waste_generation'].sum() / total_capacity
        f.write(f"\nWaste Collection Requirements:\n")
        f.write(f"  - Waste to collect daily: {gvp_df['waste_generation'].sum():.2f} units\n")
        f.write(f"  - Fleet daily capacity: {total_capacity:.2f} tonnes\n")
        f.write(f"  - Required trips per day: {trips_needed:.2f}\n")
        f.write(f"  - Rounded up: {np.ceil(trips_needed)} trips needed\n")
        
        # Detailed Sheets Analysis
        for sheet_name, df in transformed_data.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DETAILED ANALYSIS: {sheet_name}\n")
            f.write("=" * 80 + "\n")
            
            f.write(create_data_profile(df, sheet_name))
            
            if sheet_name == 'GVPs':
                f.write(analyze_coordinate_quality(df, 'latitude', 'longitude'))
                f.write(analyze_waste_data(df, 'waste_generation'))
            
            elif sheet_name == 'Fleet':
                f.write(analyze_vehicle_data(df))
            
            elif sheet_name == 'SCTPs':
                f.write(analyze_coordinate_quality(df, 'latitude', 'longitude'))
        
        # Data Quality Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("✅ DATA QUALITY SUMMARY\n")
        f.write("=" * 80 + "\n")
        
        # Calculate data quality metrics
        quality_metrics = []
        
        # GVP Quality
        gvp_valid_coords = gvp_df[['latitude', 'longitude']].dropna()
        gvp_coord_quality = len(gvp_valid_coords) / len(gvp_df) * 100
        
        gvp_valid_waste = gvp_df['waste_generation'].dropna()
        gvp_waste_quality = len(gvp_valid_waste) / len(gvp_df) * 100
        
        quality_metrics.append(f"GVP Coordinate Completeness: {gvp_coord_quality:.1f}%")
        quality_metrics.append(f"GVP Waste Data Completeness: {gvp_waste_quality:.1f}%")
        
        # Fleet Quality
        fleet_valid = fleet_df[['capacity_tonnes', 'num_vehicles']].dropna()
        fleet_quality = len(fleet_valid) / len(fleet_df) * 100
        quality_metrics.append(f"Fleet Data Completeness: {fleet_quality:.1f}%")
        
        # SCTP Quality
        sctp_valid_coords = sctp_df[['latitude', 'longitude']].dropna()
        sctp_coord_quality = len(sctp_valid_coords) / len(sctp_df) * 100
        quality_metrics.append(f"SCTP Coordinate Completeness: {sctp_coord_quality:.1f}%")
        
        for metric in quality_metrics:
            f.write(f"  • {metric}\n")
        
        # Recommendations
        f.write("\n" + "=" * 80 + "\n")
        f.write("💡 RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        
        recommendations = [
            "1. Consider balancing GVPs across SCTPs for efficient routing",
            "2. Deploy appropriate vehicle types based on cluster waste volume",
            "3. Optimize routes within each cluster to minimize travel distance",
            "4. Monitor daily waste generation for dynamic routing adjustments",
            "5. Consider adding more SCTPs in high-density waste areas"
        ]
        
        for rec in recommendations:
            f.write(f"  {rec}\n")
    
    print(f"✅ Comprehensive report saved to: {PROFILE_FILE}")
    print(f"   Report includes: Data types, missing values, coordinate analysis, waste analysis, and recommendations")

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
def main():
    print("=" * 80)
    print("MODULE 3: DATA TRANSFORMATION WITH DETAILED PROFILING")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data from Module 2...")
    data = pd.read_excel(INPUT_FILE, sheet_name=None)
    print(f"✅ Loaded {len(data)} sheets: {list(data.keys())}")
    
    # Transform each sheet with detailed profiling
    transformed_data = {}
    
    # GVP Data
    if 'Sample Data' in data:
        print("\n" + "=" * 80)
        print("🔄 TRANSFORMING GVP DATA")
        print("=" * 80)
        transformed_data['GVPs'] = transform_gvp_data(data['Sample Data'])
    else:
        print("❌ 'Sample Data' sheet not found!")
        return
    
    # Fleet Data
    if 'Fleet Details' in data:
        print("\n" + "=" * 80)
        print("🔄 TRANSFORMING FLEET DATA")
        print("=" * 80)
        transformed_data['Fleet'] = transform_vehicle_data(data['Fleet Details'])
    else:
        print("❌ 'Fleet Details' sheet not found!")
        return
    
    # SCTP Data
    if 'SCTP' in data:
        print("\n" + "=" * 80)
        print("🔄 TRANSFORMING SCTP DATA")
        print("=" * 80)
        transformed_data['SCTPs'] = transform_sctp_data(data['SCTP'])
    else:
        print("❌ 'SCTP' sheet not found!")
        return
    
    # Save transformed data
    print("\n" + "=" * 80)
    print("💾 SAVING TRANSFORMED DATA")
    print("=" * 80)
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for name, df in transformed_data.items():
            df.to_excel(writer, sheet_name=name, index=False)
    
    print(f"✅ Transformed data saved to: {OUTPUT_FILE}")
    
    # Save comprehensive report
    save_comprehensive_report(transformed_data)
    
    # Quick summary
    print("\n" + "=" * 80)
    print("📊 QUICK SUMMARY")
    print("=" * 80)
    
    gvp_df = transformed_data['GVPs']
    fleet_df = transformed_data['Fleet']
    sctp_df = transformed_data['SCTPs']
    
    total_waste = gvp_df['waste_generation'].sum()
    total_capacity = (fleet_df['capacity_tonnes'] * fleet_df['num_vehicles']).sum()
    
    print(f"✅ GVP Data: {len(gvp_df)} locations, {total_waste:.2f} units waste/day")
    print(f"✅ Fleet Data: {fleet_df['num_vehicles'].sum()} vehicles, {total_capacity:.2f} tonnes capacity")
    print(f"✅ SCTP Data: {len(sctp_df)} transfer stations")
    print(f"📈 Efficiency: {total_waste/total_capacity:.2f} trips needed daily")
    
    print("\n" + "=" * 80)
    print("🎉 MODULE 3 COMPLETED WITH DETAILED PROFILING!")
    print("=" * 80)
    print(f"📁 Outputs:")
    print(f"   • transformed_data.xlsx → Cleaned data for Module 4")
    print(f"   • data_profile_report.txt → Detailed analysis report")
    print(f"   • Check the report for complete data quality insights!")

if __name__ == "__main__":
    main()