# data_loader.py - Your first data science script!

import pandas as pd
import numpy as np

def load_crime_data(filename):
    """
    Load and inspect crime data from CSV file
    """
    print("🔄 Loading crime data...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(filename)
        print(f"✅ Successfully loaded {len(df)} crime records!")
        return df
    except FileNotFoundError:
        print(f"❌ Could not find file: {filename}")
        print("Make sure the file is in your project folder!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def inspect_data(df):
    """
    Look at the structure and quality of our data
    """
    print("\n📊 DATA INSPECTION REPORT")
    print("=" * 50)
    
    # Basic info
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Column names
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes)
    
    # Missing values
    print(f"\nMissing values:")
    missing_counts = df.isnull().sum()
    for col in missing_counts.index:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} missing")
    
    # First few rows
    print(f"\nFirst 5 rows of data:")
    print(df.head())
    
    return df

# Main execution
if __name__ == "__main__":
    # Replace 'your_crime_data.csv' with your actual filename
    filename = "Crime_Data_from_2020_to_Present.csv"  # Update this!
    
    # Load the data
    crime_df = load_crime_data(filename)
    
    if crime_df is not None:
        # Inspect the data
        inspect_data(crime_df)
        
        # Save a summary
        print(f"\n💾 Saving data summary...")
        crime_df.describe().to_csv("data_summary.csv")
        print(f"✅ Summary saved to 'data_summary.csv'")
