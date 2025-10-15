#!/usr/bin/env python3
"""
Script to check for static data in CSV files.
Identifies files where velocity (GPS_Spd) and attitude (ATT_Roll, ATT_Pitch, ATT_Yaw) 
are predominantly 0.00, indicating static/stationary conditions.
"""

import csv
import glob
import os

def check_static_data(file_path, threshold=0.9):
    """
    Check if a CSV file contains mostly static data
    
    Args:
        file_path: Path to the CSV file
        threshold: Percentage of zeros required to consider as static (default: 90%)
    
    Returns:
        tuple: (is_static, velocity_zero_pct, attitude_zero_pct, total_records)
    """
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Define the columns to check for static data
            velocity_cols = ['GPS_Spd']  # GPS Speed
            attitude_cols = ['ATT_Roll', 'ATT_Pitch', 'ATT_Yaw']  # Attitude angles
            
            # Check which columns exist in the dataset
            fieldnames = reader.fieldnames or []
            available_velocity_cols = [col for col in velocity_cols if col in fieldnames]
            available_attitude_cols = [col for col in attitude_cols if col in fieldnames]
            
            if not available_velocity_cols and not available_attitude_cols:
                return False, 0, 0, 0
            
            rows = list(reader)
            total_records = len(rows)
            
            if total_records == 0:
                return False, 0, 0, 0
            
            # Check velocity data
            velocity_zero_count = 0
            if available_velocity_cols:
                for row in rows:
                    all_velocity_zero = True
                    for col in available_velocity_cols:
                        try:
                            value = float(row[col])
                            if abs(value) >= 0.001:
                                all_velocity_zero = False
                                break
                        except (ValueError, KeyError):
                            all_velocity_zero = False
                            break
                    if all_velocity_zero:
                        velocity_zero_count += 1
            
            velocity_zero_pct = velocity_zero_count / total_records if available_velocity_cols else 0
            
            # Check attitude data
            attitude_zero_count = 0
            if available_attitude_cols:
                for row in rows:
                    all_attitude_zero = True
                    for col in available_attitude_cols:
                        try:
                            value = float(row[col])
                            if abs(value) >= 0.001:
                                all_attitude_zero = False
                                break
                        except (ValueError, KeyError):
                            all_attitude_zero = False
                            break
                    if all_attitude_zero:
                        attitude_zero_count += 1
            
            attitude_zero_pct = attitude_zero_count / total_records if available_attitude_cols else 0
            
            # Consider static if either velocity or attitude data is mostly zero
            is_static = (velocity_zero_pct >= threshold) or (attitude_zero_pct >= threshold)
            
            return is_static, velocity_zero_pct, attitude_zero_pct, total_records
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0, 0, 0

def main():
    # Get current directory path
    current_dir = os.getcwd()
    
    # Find all CSV files in the current directory
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze for static data...")
    print("Threshold: Files with >90% zeros in velocity or attitude are considered static")
    print("-" * 80)
    print(f"{'Filename':<50} | {'Status':<8} | {'Vel Zero%':<9} | {'Att Zero%':<9} | {'Records':<7}")
    print("-" * 80)
    
    static_files = []
    dynamic_files = []
    total_static_records = 0
    total_dynamic_records = 0
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        is_static, vel_zero_pct, att_zero_pct, records = check_static_data(csv_file)
        
        if records > 0:
            if is_static:
                static_files.append(filename)
                total_static_records += records
                status = "STATIC"
            else:
                dynamic_files.append(filename)
                total_dynamic_records += records
                status = "DYNAMIC"
            
            print(f"{filename:<50} | {status:<8} | {vel_zero_pct*100:7.1f}% | {att_zero_pct*100:7.1f}% | {records:7d}")
        else:
            print(f"{filename:<50} | ERROR    | {'N/A':<7} | {'N/A':<7} | {records:7d}")
    
    print("-" * 80)
    print(f"SUMMARY:")
    print(f"Total files analyzed: {len(csv_files)}")
    print(f"Static files: {len(static_files)}")
    print(f"Dynamic files: {len(dynamic_files)}")
    print(f"Static data records: {total_static_records:,}")
    print(f"Dynamic data records: {total_dynamic_records:,}")
    
    if static_files:
        print(f"\nSTATIC FILES ({len(static_files)}):")
        for i, filename in enumerate(static_files, 1):
            print(f"  {i:2d}. {filename}")
    
    if len(static_files) > 20:
        print(f"\n... and {len(static_files) - 20} more static files")

if __name__ == "__main__":
    main()
