#!/usr/bin/env python3
"""
Script to calculate total Time of Flight (ToF) from multiple CSV files.
Each CSV file contains timestamped flight data where TimeUS is the timestamp in microseconds.
"""

import csv
import glob
import os

def calculate_tof_for_file(file_path):
    """Calculate time of flight for a single CSV file"""
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if TimeUS column exists
            if 'TimeUS' not in reader.fieldnames:
                print(f"Warning: TimeUS column not found in {file_path}")
                return 0, 0
            
            # Read all rows to get first and last timestamps
            rows = list(reader)
            if len(rows) == 0:
                return 0, 0
            
            # Get first and last timestamps (in microseconds)
            first_timestamp = float(rows[0]['TimeUS'])
            last_timestamp = float(rows[-1]['TimeUS'])
            
            # Calculate flight duration in microseconds, then convert to seconds
            flight_duration_us = last_timestamp - first_timestamp
            flight_duration_seconds = flight_duration_us / 1_000_000
            
            return flight_duration_seconds, len(rows)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def main():
    # Get current directory path
    current_dir = os.getcwd()
    
    # Find all CSV files in the current directory
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process...")
    print("-" * 60)
    
    total_tof_seconds = 0
    total_records = 0
    valid_files = 0
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        tof_seconds, records = calculate_tof_for_file(csv_file)
        
        if tof_seconds > 0:
            valid_files += 1
            total_tof_seconds += tof_seconds
            total_records += records
            
            print(f"{filename:50} | ToF: {tof_seconds:8.2f}s | Records: {records:6d}")
        else:
            print(f"{filename:50} | ToF: ERROR   | Records: {records:6d}")
    
    print("-" * 60)
    print(f"SUMMARY:")
    print(f"Valid files processed: {valid_files}")
    print(f"Total records: {total_records:,}")
    print(f"Total Time of Flight: {total_tof_seconds:.2f} seconds")
    print(f"Total Time of Flight: {total_tof_seconds/60:.2f} minutes")
    print(f"Total Time of Flight: {total_tof_seconds/3600:.2f} hours")

if __name__ == "__main__":
    main()
