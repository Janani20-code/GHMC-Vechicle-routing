import pandas as pd
import os

# ---------------------------------------------------
# DATA CLEANING PIPELINE
# ---------------------------------------------------

def clean_excel_data(input_file, output_file):

    try:
        # STEP 1: UNDERSTAND
        if not os.path.exists(input_file):
            raise FileNotFoundError("Raw data file not found.")

        excel = pd.ExcelFile(input_file)
        cleaned_sheets = {}

        print("\n🧹 Starting Data Cleaning...\n")

        for sheet in excel.sheet_names:
            print(f"📄 Cleaning Sheet: {sheet}")

            # STEP 2: LOAD & FIX HEADERS
            df = pd.read_excel(input_file, sheet_name=sheet, header=2)
             # SHOW FIRST 5 ROWS
            # ------------------------------
            print("📝 First 5 rows of the sheet:")
            print(df.head(), "\n")

            # STEP 3: DATA TYPE CHECK
            print("🔍 Data Types Before:")
            print(df.dtypes)

            # STEP 4: MISSING VALUES
            missing = df.isnull().sum()
            print("⚠ Missing Values:")
            print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

            # STEP 5: DUPLICATES
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                print(f"❌ Duplicates Found: {duplicates}")
                df = df.drop_duplicates()
            else:
                print("✅ No duplicates found")

            # STEP 6: CLEAN COLUMN NAMES (DO NOT RENAME)
            # Only remove extra spaces, keep original column names intact
            df.columns = df.columns.str.strip()

            # STEP 7: VALIDATE
            if df.empty:
                raise ValueError(f"{sheet} became empty after cleaning.")

            cleaned_sheets[sheet] = df
            print("✅ Sheet cleaned successfully\n")

        # STEP 8: SAVE CLEANED DATA
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for name, clean_df in cleaned_sheets.items():
                clean_df.to_excel(writer, sheet_name=name, index=False)

        print("🎉 CLEANING COMPLETED SUCCESSFULLY!")
        print("📁 Cleaned file saved as:", output_file)

    except Exception as error:
        print("❌ Error during cleaning:", error)


# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------

if __name__ == "__main__":
    input_file = r"E:\vehicle_routing_system\data\loaded_data.xlsx"
    output_file = r"E:\vehicle_routing_system\data\cleaned_data.xlsx"


    clean_excel_data(input_file, output_file)
