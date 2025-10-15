#!/usr/bin/env python3
"""
Drone Log Data Cleaner for TCN GPS Error Reduction Model

This script extracts relevant sensor data from ArduPilot log files for training
a Temporal Convolutional Network (TCN) to reduce GPS positioning errors.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DroneLogCleaner:
    """
    A class to parse and clean ArduPilot drone log files for TCN training.
    
    Extracts relevant GPS, IMU, attitude, and EKF data while filtering out
    unnecessary parameters to prepare data for GPS error reduction model training.
    """
    
    def __init__(self):
        # Define the message types and their relevant fields for TCN training
        self.target_messages = {
            'GPS': ['TimeUS', 'I', 'Status', 'NSats', 'HDop', 'Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ', 'Yaw'],
            'IMU': ['TimeUS', 'I', 'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ', 'T'],
            'ACC': ['TimeUS', 'I', 'SampleUS', 'AccX', 'AccY', 'AccZ'],
            'GYR': ['TimeUS', 'I', 'SampleUS', 'GyrX', 'GyrY', 'GyrZ'],
            'ATT': ['TimeUS', 'DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw', 'AEKF'],
            'XKF1': ['TimeUS', 'C', 'Roll', 'Pitch', 'Yaw', 'VN', 'VE', 'VD', 'dPD', 'PN', 'PE', 'PD'],
            'XKF2': ['TimeUS', 'C', 'AX', 'AY', 'AZ', 'VWN', 'VWE'],
            'XKF3': ['TimeUS', 'C', 'IVN', 'IVE', 'IVD', 'IPN', 'IPE', 'IPD'],
            'XKF4': ['TimeUS', 'C', 'SV', 'SP', 'SH', 'SM', 'SVT', 'GPS'],
            'BARO': ['TimeUS', 'I', 'Alt', 'Press', 'Temp'],
            'MAG': ['TimeUS', 'I', 'MagX', 'MagY', 'MagZ'],
            'POS': ['TimeUS', 'Lat', 'Lng', 'Alt', 'RelHomeAlt'],
            'AHR2': ['TimeUS', 'Roll', 'Pitch', 'Yaw', 'Alt', 'Lat', 'Lng'],
        }
        
        self.format_definitions = {}
        self.parsed_data = {}
        
    def parse_format_line(self, line: str) -> Optional[Dict]:
        """Parse FMT line to understand message structure."""
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5 and parts[0].endswith('FMT'):
                msg_type = parts[3]
                if msg_type in self.target_messages:
                    field_names = parts[5].split(',') if len(parts) > 5 else []
                    return {
                        'type': msg_type,
                        'fields': field_names,
                        'format': parts[4] if len(parts) > 4 else ''
                    }
        except Exception as e:
            logger.debug(f"Error parsing format line: {e}")
        return None
    
    def parse_data_line(self, line: str, msg_type: str) -> Optional[Dict]:
        """Parse data line based on message type."""
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) > 1 and parts[0].split('|')[-1] == msg_type:
                data = {}
                if msg_type in self.format_definitions:
                    fields = self.format_definitions[msg_type]['fields']
                    target_fields = self.target_messages.get(msg_type, [])
                    
                    for i, field in enumerate(fields):
                        if field in target_fields and i + 1 < len(parts):
                            try:
                                # Try to convert to float, keep as string if it fails
                                value = parts[i + 1]
                                if value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                                    data[field] = float(value)
                                else:
                                    data[field] = value
                            except (ValueError, IndexError):
                                continue
                    
                    if data:  # Only return if we got some data
                        return data
        except Exception as e:
            logger.debug(f"Error parsing data line: {e}")
        return None
    
    def parse_single_log_file(self, filepath: str) -> Dict[str, List[Dict]]:
        """Parse a single log file and extract relevant data."""
        logger.info(f"Processing file: {filepath}")
        
        file_data = {msg_type: [] for msg_type in self.target_messages.keys()}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                # First pass: Parse format definitions
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    if 'FMT,' in line:
                        fmt_info = self.parse_format_line(line)
                        if fmt_info:
                            self.format_definitions[fmt_info['type']] = fmt_info
                
                # Second pass: Parse data
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    for msg_type in self.target_messages.keys():
                        if f'|{msg_type},' in line or f'{msg_type},' in line:
                            data = self.parse_data_line(line, msg_type)
                            if data:
                                file_data[msg_type].append(data)
                
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            return {}
        
        # Log statistics
        total_records = sum(len(records) for records in file_data.values())
        logger.info(f"Extracted {total_records} total records from {filepath}")
        for msg_type, records in file_data.items():
            if records:
                logger.info(f"  {msg_type}: {len(records)} records")
        
        return file_data
    
    def process_all_log_files(self, input_dir: str, output_dir: str = None) -> None:
        """Process all log files in the input directory."""
        if output_dir is None:
            output_dir = input_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all log files
        log_files = glob.glob(os.path.join(input_dir, "*.log"))
        logger.info(f"Found {len(log_files)} log files to process")
        
        if not log_files:
            logger.warning(f"No .log files found in {input_dir}")
            return
        
        all_data = {msg_type: [] for msg_type in self.target_messages.keys()}
        
        # Process each log file
        for log_file in log_files:
            file_data = self.parse_single_log_file(log_file)
            
            # Aggregate data from all files
            for msg_type, records in file_data.items():
                if records:
                    # Add filename to each record for traceability
                    filename = os.path.basename(log_file)
                    for record in records:
                        record['source_file'] = filename
                    all_data[msg_type].extend(records)
        
        # Save extracted data
        self.save_cleaned_data(all_data, output_dir)
        
        # Generate summary report
        self.generate_summary_report(all_data, output_dir)
    
    def save_cleaned_data(self, data: Dict[str, List[Dict]], output_dir: str) -> None:
        """Save cleaned data to CSV files and combined formats."""
        logger.info("Saving cleaned data...")
        
        # Save individual message types
        for msg_type, records in data.items():
            if records:
                df = pd.DataFrame(records)
                csv_path = os.path.join(output_dir, f"cleaned_{msg_type.lower()}_data.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(records)} {msg_type} records to {csv_path}")
        
        # Create a combined dataset with synchronized timestamps
        self.create_synchronized_dataset(data, output_dir)
    
    def create_synchronized_dataset(self, data: Dict[str, List[Dict]], output_dir: str) -> None:
        """Create a time-synchronized dataset suitable for TCN training."""
        logger.info("Creating synchronized dataset for TCN training...")
        
        # Convert to DataFrames and handle timestamps
        dfs = {}
        for msg_type, records in data.items():
            if records:
                df = pd.DataFrame(records)
                if 'TimeUS' in df.columns:
                    df['TimeUS'] = pd.to_numeric(df['TimeUS'], errors='coerce')
                    df = df.dropna(subset=['TimeUS'])
                    df = df.sort_values('TimeUS')
                    dfs[msg_type] = df
        
        if not dfs:
            logger.warning("No data with valid timestamps found")
            return
        
        # Find common time range
        min_time = max(df['TimeUS'].min() for df in dfs.values() if 'TimeUS' in df.columns)
        max_time = min(df['TimeUS'].max() for df in dfs.values() if 'TimeUS' in df.columns)
        
        logger.info(f"Common time range: {min_time} to {max_time} microseconds")
        
        # Create synchronized dataset (basic version - can be enhanced based on specific needs)
        if 'GPS' in dfs:
            base_df = dfs['GPS'][['TimeUS', 'source_file']].copy()
            base_df = base_df[(base_df['TimeUS'] >= min_time) & (base_df['TimeUS'] <= max_time)]
            
            # Add GPS features
            gps_features = ['Lat', 'Lng', 'Alt', 'Spd', 'NSats', 'HDop', 'VZ']
            for feature in gps_features:
                if feature in dfs['GPS'].columns:
                    base_df[f'GPS_{feature}'] = dfs['GPS'][feature].values[:len(base_df)]
            
            # Add IMU features (interpolated to GPS timestamps)
            if 'IMU' in dfs:
                imu_df = dfs['IMU']
                imu_features = ['GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ']
                
                for feature in imu_features:
                    if feature in imu_df.columns:
                        # Simple nearest-neighbor interpolation
                        base_df[f'IMU_{feature}'] = np.interp(
                            base_df['TimeUS'].values,
                            imu_df['TimeUS'].values,
                            imu_df[feature].values
                        )
            
            # Add attitude data
            if 'ATT' in dfs:
                att_df = dfs['ATT']
                att_features = ['Roll', 'Pitch', 'Yaw']
                
                for feature in att_features:
                    if feature in att_df.columns:
                        base_df[f'ATT_{feature}'] = np.interp(
                            base_df['TimeUS'].values,
                            att_df['TimeUS'].values,
                            att_df[feature].values
                        )
            
            # Save synchronized dataset
            sync_path = os.path.join(output_dir, "synchronized_tcn_dataset.csv")
            base_df.to_csv(sync_path, index=False)
            logger.info(f"Saved synchronized dataset with {len(base_df)} records to {sync_path}")
    
    def generate_summary_report(self, data: Dict[str, List[Dict]], output_dir: str) -> None:
        """Generate a summary report of the cleaned data."""
        report_path = os.path.join(output_dir, "data_cleaning_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("DRONE LOG DATA CLEANING SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            total_records = sum(len(records) for records in data.values())
            f.write(f"Total records extracted: {total_records}\n\n")
            
            f.write("Records by message type:\n")
            f.write("-" * 30 + "\n")
            
            for msg_type in sorted(data.keys()):
                records = data[msg_type]
                f.write(f"{msg_type:15}: {len(records):8} records\n")
                
                if records:
                    # Show sample of fields
                    sample_record = records[0]
                    fields = [k for k in sample_record.keys() if k != 'source_file']
                    f.write(f"{'':15}  Fields: {', '.join(fields[:10])}\n")
                    if len(fields) > 10:
                        f.write(f"{'':15}          ... and {len(fields) - 10} more\n")
                f.write("\n")
            
            # Data quality insights
            f.write("\nDATA QUALITY INSIGHTS:\n")
            f.write("-" * 30 + "\n")
            
            if 'GPS' in data and data['GPS']:
                gps_data = data['GPS']
                f.write(f"GPS records: {len(gps_data)}\n")
                
                # Check for coordinate validity
                valid_coords = sum(1 for r in gps_data 
                                 if 'Lat' in r and 'Lng' in r 
                                 and abs(r['Lat']) <= 90 and abs(r['Lng']) <= 180)
                f.write(f"Valid GPS coordinates: {valid_coords}/{len(gps_data)} ({valid_coords/len(gps_data)*100:.1f}%)\n")
            
            if 'IMU' in data and data['IMU']:
                f.write(f"IMU records: {len(data['IMU'])}\n")
            
            # Recommend next steps
            f.write("\nRECOMMENDED NEXT STEPS FOR TCN TRAINING:\n")
            f.write("-" * 45 + "\n")
            f.write("1. Review synchronized_tcn_dataset.csv for training\n")
            f.write("2. Implement additional preprocessing:\n")
            f.write("   - Normalize sensor data\n")
            f.write("   - Handle outliers and noise\n")
            f.write("   - Create sliding time windows for TCN input\n")
            f.write("   - Split data into train/validation/test sets\n")
            f.write("3. Define target variable (GPS error to predict/correct)\n")
            f.write("4. Consider feature engineering:\n")
            f.write("   - Velocity and acceleration derivatives\n")
            f.write("   - Statistical features over time windows\n")
            f.write("   - Cross-correlation between sensors\n")
        
        logger.info(f"Generated summary report: {report_path}")

def main():
    """Main function to run the data cleaning process."""
    parser = argparse.ArgumentParser(
        description="Clean drone log files for TCN GPS error reduction model training"
    )
    parser.add_argument(
        "input_dir", 
        help="Directory containing ArduPilot log files"
    )
    parser.add_argument(
        "--output_dir", 
        help="Output directory for cleaned data (default: same as input_dir)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create cleaner instance and process files
    cleaner = DroneLogCleaner()
    
    try:
        cleaner.process_all_log_files(args.input_dir, args.output_dir)
        logger.info("Data cleaning completed successfully!")
        print("\n" + "="*60)
        print("CLEANING COMPLETED!")
        print("="*60)
        print(f"Check the output directory for:")
        print(f"• Individual CSV files for each sensor type")
        print(f"• synchronized_tcn_dataset.csv - Main dataset for TCN training")  
        print(f"• data_cleaning_report.txt - Summary and recommendations")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
