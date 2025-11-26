<<<<<<< HEAD
# data_preprocessing.py - Clean and prepare data for machine learning

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filename):
    """
    Load and clean the Los Angeles crime data
    """
    print("🔄 Loading and cleaning crime data...")
    
    # Load data
    df = pd.read_csv(filename)
    print(f"✅ Loaded {len(df)} records")
    
    # 1. Clean and rename columns to match our needs
    df_clean = df.copy()
    
    # Rename columns to be more user-friendly
    column_mapping = {
        'DATE OCC': 'date',
        'TIME OCC': 'time_raw',
        'Crm Cd Desc': 'crime_type',
        'LAT': 'latitude',
        'LON': 'longitude',
        'AREA NAME': 'district',
        'AREA': 'area_code',
        'Premis Desc': 'location_description'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # 2. Handle missing values in key columns
    print("🧹 Cleaning missing values...")
    
    # Remove records with missing location data (essential for mapping)
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
    removed_no_location = initial_count - len(df_clean)
    print(f"   Removed {removed_no_location} records with missing location data")
    
    # Remove records with missing dates
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['date'])
    removed_no_date = initial_count - len(df_clean)
    print(f"   Removed {removed_no_date} records with missing dates")
    
    # 3. Fix date and time formatting
    print("📅 Processing dates and times...")
    
    # Convert date strings to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Remove invalid dates
    df_clean = df_clean.dropna(subset=['date'])
    
    # Convert TIME OCC (which is in HHMM format) to proper time
    df_clean['hour'] = df_clean['time_raw'] // 100
    df_clean['minute'] = df_clean['time_raw'] % 100
    
    # Handle invalid times (some datasets have times like 2400)
    df_clean['hour'] = df_clean['hour'].clip(0, 23)
    df_clean['minute'] = df_clean['minute'].clip(0, 59)
    
    # Create proper datetime column
    df_clean['datetime'] = pd.to_datetime(
        df_clean['date'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_clean['hour'].astype(str).str.zfill(2) + ':' + 
        df_clean['minute'].astype(str).str.zfill(2)
    )
    
    # 4. Feature engineering - create useful time-based features
    print("⚙️ Creating time-based features...")
    
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek  # 0=Monday
    df_clean['day_name'] = df_clean['date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])  # Saturday, Sunday
    
    # Create time periods
    df_clean['time_period'] = pd.cut(df_clean['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
    
    # Create season
    df_clean['season'] = df_clean['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # 5. Clean crime types
    print("🏷️ Processing crime types...")
    
    # Fill missing crime types
    df_clean['crime_type'] = df_clean['crime_type'].fillna('UNKNOWN')
    
    # Group similar crime types to reduce categories
    def simplify_crime_type(crime_type):
        crime_type = str(crime_type).upper()
        if 'THEFT' in crime_type or 'STEAL' in crime_type or 'LARCENY' in crime_type:
            return 'THEFT'
        elif 'ASSAULT' in crime_type or 'BATTERY' in crime_type:
            return 'ASSAULT'
        elif 'BURGLARY' in crime_type or 'BREAK' in crime_type:
            return 'BURGLARY'
        elif 'ROBBERY' in crime_type or 'ROB' in crime_type:
            return 'ROBBERY'
        elif 'VANDAL' in crime_type or 'DAMAGE' in crime_type:
            return 'VANDALISM'
        elif 'DRUG' in crime_type or 'NARCOTIC' in crime_type:
            return 'DRUG_OFFENSE'
        elif 'VEHICLE' in crime_type or 'AUTO' in crime_type:
            return 'VEHICLE_CRIME'
        else:
            return 'OTHER'
    
    df_clean['crime_category'] = df_clean['crime_type'].apply(simplify_crime_type)
    
    # 6. Filter out invalid coordinates (outside LA area)
    print("🗺️ Filtering location data...")
    
    # LA County approximate bounds
    LA_LAT_MIN, LA_LAT_MAX = 33.7, 34.8
    LA_LON_MIN, LA_LON_MAX = -119.0, -117.6
    
    initial_count = len(df_clean)
    df_clean = df_clean[
        (df_clean['latitude'] >= LA_LAT_MIN) & (df_clean['latitude'] <= LA_LAT_MAX) &
        (df_clean['longitude'] >= LA_LON_MIN) & (df_clean['longitude'] <= LA_LON_MAX)
    ]
    removed_outside_LA = initial_count - len(df_clean)
    print(f"   Removed {removed_outside_LA} records outside LA area")
    
    # 7. Final dataset info
    print(f"\n✅ CLEANING COMPLETE!")
    print(f"   Final dataset: {len(df_clean)} records")
    print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"   Unique crime categories: {df_clean['crime_category'].nunique()}")
    print(f"   Geographic areas: {df_clean['district'].nunique()}")
    
    return df_clean

def create_summary_visualizations(df):
    """
    Create basic visualizations to understand the data
    """
    print("📈 Creating summary visualizations...")
    
    plt.figure(figsize=(16, 12))
    
    # 1. Crime types distribution
    plt.subplot(2, 3, 1)
    crime_counts = df['crime_category'].value_counts().head(10)
    plt.bar(crime_counts.index, crime_counts.values)
    plt.title('Top 10 Crime Categories')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Crimes')
    
    # 2. Crimes by hour of day
    plt.subplot(2, 3, 2)
    hourly_crimes = df['hour'].value_counts().sort_index()
    plt.plot(hourly_crimes.index, hourly_crimes.values, marker='o')
    plt.title('Crimes by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Crimes')
    plt.grid(True)
    
    # 3. Crimes by day of week
    plt.subplot(2, 3, 3)
    daily_crimes = df['day_name'].value_counts()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_crimes = daily_crimes.reindex(days_order)
    plt.bar(daily_crimes.index, daily_crimes.values)
    plt.title('Crimes by Day of Week')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Crimes')
    
    # 4. Crimes by month
    plt.subplot(2, 3, 4)
    monthly_crimes = df['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.plot(monthly_crimes.index, monthly_crimes.values, marker='o')
    plt.title('Crimes by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.xticks(range(1, 13), month_names)
    plt.grid(True)
    
    # 5. Crimes by time period
    plt.subplot(2, 3, 5)
    time_period_crimes = df['time_period'].value_counts()
    plt.pie(time_period_crimes.values, labels=time_period_crimes.index, autopct='%1.1f%%')
    plt.title('Crimes by Time Period')
    
    # 6. Top districts by crime count
    plt.subplot(2, 3, 6)
    district_crimes = df['district'].value_counts().head(10)
    plt.barh(district_crimes.index, district_crimes.values)
    plt.title('Top 10 Districts by Crime Count')
    plt.xlabel('Number of Crimes')
    
    plt.tight_layout()
    plt.savefig('crime_data_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'crime_data_summary.png'")

def save_clean_data(df, filename='clean_crime_data.csv'):
    """
    Save the cleaned dataset
    """
    # Select the most important columns for machine learning
    important_columns = [
        'datetime', 'date', 'hour', 'minute', 
        'crime_type', 'crime_category',
        'latitude', 'longitude', 'district', 'area_code',
        'year', 'month', 'day', 'day_of_week', 'day_name',
        'is_weekend', 'time_period', 'season'
    ]
    
    # Include additional columns if they exist
    available_columns = [col for col in important_columns if col in df.columns]
    df_final = df[available_columns].copy()
    
    # Save to CSV
    df_final.to_csv(filename, index=False)
    print(f"✅ Clean dataset saved as '{filename}'")
    print(f"   Columns: {len(df_final.columns)}")
    print(f"   Rows: {len(df_final)}")
    
    return df_final

# Main execution
if __name__ == "__main__":
    # Replace with your actual filename
    raw_data_file = "Crime_Data_from_2020_to_Present.csv"
    
    # Process the data
    clean_df = load_and_clean_data(raw_data_file)
    
    if clean_df is not None and len(clean_df) > 0:
        # Create visualizations
        create_summary_visualizations(clean_df)
        
        # Save clean data
        final_df = save_clean_data(clean_df)
        
        print(f"\n🎉 DATA PREPROCESSING COMPLETE!")
        print(f"Your dataset is now ready for machine learning!")
        
    else:
        print("❌ Data preprocessing failed. Please check your input file.")
=======
# data_preprocessing.py - Clean and prepare data for machine learning

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filename):
    """
    Load and clean the Los Angeles crime data
    """
    print("🔄 Loading and cleaning crime data...")
    
    # Load data
    df = pd.read_csv(filename)
    print(f"✅ Loaded {len(df)} records")
    
    # 1. Clean and rename columns to match our needs
    df_clean = df.copy()
    
    # Rename columns to be more user-friendly
    column_mapping = {
        'DATE OCC': 'date',
        'TIME OCC': 'time_raw',
        'Crm Cd Desc': 'crime_type',
        'LAT': 'latitude',
        'LON': 'longitude',
        'AREA NAME': 'district',
        'AREA': 'area_code',
        'Premis Desc': 'location_description'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # 2. Handle missing values in key columns
    print("🧹 Cleaning missing values...")
    
    # Remove records with missing location data (essential for mapping)
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
    removed_no_location = initial_count - len(df_clean)
    print(f"   Removed {removed_no_location} records with missing location data")
    
    # Remove records with missing dates
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['date'])
    removed_no_date = initial_count - len(df_clean)
    print(f"   Removed {removed_no_date} records with missing dates")
    
    # 3. Fix date and time formatting
    print("📅 Processing dates and times...")
    
    # Convert date strings to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Remove invalid dates
    df_clean = df_clean.dropna(subset=['date'])
    
    # Convert TIME OCC (which is in HHMM format) to proper time
    df_clean['hour'] = df_clean['time_raw'] // 100
    df_clean['minute'] = df_clean['time_raw'] % 100
    
    # Handle invalid times (some datasets have times like 2400)
    df_clean['hour'] = df_clean['hour'].clip(0, 23)
    df_clean['minute'] = df_clean['minute'].clip(0, 59)
    
    # Create proper datetime column
    df_clean['datetime'] = pd.to_datetime(
        df_clean['date'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_clean['hour'].astype(str).str.zfill(2) + ':' + 
        df_clean['minute'].astype(str).str.zfill(2)
    )
    
    # 4. Feature engineering - create useful time-based features
    print("⚙️ Creating time-based features...")
    
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek  # 0=Monday
    df_clean['day_name'] = df_clean['date'].dt.day_name()
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])  # Saturday, Sunday
    
    # Create time periods
    df_clean['time_period'] = pd.cut(df_clean['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
    
    # Create season
    df_clean['season'] = df_clean['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # 5. Clean crime types
    print("🏷️ Processing crime types...")
    
    # Fill missing crime types
    df_clean['crime_type'] = df_clean['crime_type'].fillna('UNKNOWN')
    
    # Group similar crime types to reduce categories
    def simplify_crime_type(crime_type):
        crime_type = str(crime_type).upper()
        if 'THEFT' in crime_type or 'STEAL' in crime_type or 'LARCENY' in crime_type:
            return 'THEFT'
        elif 'ASSAULT' in crime_type or 'BATTERY' in crime_type:
            return 'ASSAULT'
        elif 'BURGLARY' in crime_type or 'BREAK' in crime_type:
            return 'BURGLARY'
        elif 'ROBBERY' in crime_type or 'ROB' in crime_type:
            return 'ROBBERY'
        elif 'VANDAL' in crime_type or 'DAMAGE' in crime_type:
            return 'VANDALISM'
        elif 'DRUG' in crime_type or 'NARCOTIC' in crime_type:
            return 'DRUG_OFFENSE'
        elif 'VEHICLE' in crime_type or 'AUTO' in crime_type:
            return 'VEHICLE_CRIME'
        else:
            return 'OTHER'
    
    df_clean['crime_category'] = df_clean['crime_type'].apply(simplify_crime_type)
    
    # 6. Filter out invalid coordinates (outside LA area)
    print("🗺️ Filtering location data...")
    
    # LA County approximate bounds
    LA_LAT_MIN, LA_LAT_MAX = 33.7, 34.8
    LA_LON_MIN, LA_LON_MAX = -119.0, -117.6
    
    initial_count = len(df_clean)
    df_clean = df_clean[
        (df_clean['latitude'] >= LA_LAT_MIN) & (df_clean['latitude'] <= LA_LAT_MAX) &
        (df_clean['longitude'] >= LA_LON_MIN) & (df_clean['longitude'] <= LA_LON_MAX)
    ]
    removed_outside_LA = initial_count - len(df_clean)
    print(f"   Removed {removed_outside_LA} records outside LA area")
    
    # 7. Final dataset info
    print(f"\n✅ CLEANING COMPLETE!")
    print(f"   Final dataset: {len(df_clean)} records")
    print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"   Unique crime categories: {df_clean['crime_category'].nunique()}")
    print(f"   Geographic areas: {df_clean['district'].nunique()}")
    
    return df_clean

def create_summary_visualizations(df):
    """
    Create basic visualizations to understand the data
    """
    print("📈 Creating summary visualizations...")
    
    plt.figure(figsize=(16, 12))
    
    # 1. Crime types distribution
    plt.subplot(2, 3, 1)
    crime_counts = df['crime_category'].value_counts().head(10)
    plt.bar(crime_counts.index, crime_counts.values)
    plt.title('Top 10 Crime Categories')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Crimes')
    
    # 2. Crimes by hour of day
    plt.subplot(2, 3, 2)
    hourly_crimes = df['hour'].value_counts().sort_index()
    plt.plot(hourly_crimes.index, hourly_crimes.values, marker='o')
    plt.title('Crimes by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Crimes')
    plt.grid(True)
    
    # 3. Crimes by day of week
    plt.subplot(2, 3, 3)
    daily_crimes = df['day_name'].value_counts()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_crimes = daily_crimes.reindex(days_order)
    plt.bar(daily_crimes.index, daily_crimes.values)
    plt.title('Crimes by Day of Week')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Crimes')
    
    # 4. Crimes by month
    plt.subplot(2, 3, 4)
    monthly_crimes = df['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.plot(monthly_crimes.index, monthly_crimes.values, marker='o')
    plt.title('Crimes by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.xticks(range(1, 13), month_names)
    plt.grid(True)
    
    # 5. Crimes by time period
    plt.subplot(2, 3, 5)
    time_period_crimes = df['time_period'].value_counts()
    plt.pie(time_period_crimes.values, labels=time_period_crimes.index, autopct='%1.1f%%')
    plt.title('Crimes by Time Period')
    
    # 6. Top districts by crime count
    plt.subplot(2, 3, 6)
    district_crimes = df['district'].value_counts().head(10)
    plt.barh(district_crimes.index, district_crimes.values)
    plt.title('Top 10 Districts by Crime Count')
    plt.xlabel('Number of Crimes')
    
    plt.tight_layout()
    plt.savefig('crime_data_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'crime_data_summary.png'")

def save_clean_data(df, filename='clean_crime_data.csv'):
    """
    Save the cleaned dataset
    """
    # Select the most important columns for machine learning
    important_columns = [
        'datetime', 'date', 'hour', 'minute', 
        'crime_type', 'crime_category',
        'latitude', 'longitude', 'district', 'area_code',
        'year', 'month', 'day', 'day_of_week', 'day_name',
        'is_weekend', 'time_period', 'season'
    ]
    
    # Include additional columns if they exist
    available_columns = [col for col in important_columns if col in df.columns]
    df_final = df[available_columns].copy()
    
    # Save to CSV
    df_final.to_csv(filename, index=False)
    print(f"✅ Clean dataset saved as '{filename}'")
    print(f"   Columns: {len(df_final.columns)}")
    print(f"   Rows: {len(df_final)}")
    
    return df_final

# Main execution
if __name__ == "__main__":
    # Replace with your actual filename
    raw_data_file = "Crime_Data_from_2020_to_Present.csv"
    
    # Process the data
    clean_df = load_and_clean_data(raw_data_file)
    
    if clean_df is not None and len(clean_df) > 0:
        # Create visualizations
        create_summary_visualizations(clean_df)
        
        # Save clean data
        final_df = save_clean_data(clean_df)
        
        print(f"\n🎉 DATA PREPROCESSING COMPLETE!")
        print(f"Your dataset is now ready for machine learning!")
        
    else:
        print("❌ Data preprocessing failed. Please check your input file.")
>>>>>>> origin/main
