#!/usr/bin/env python3
"""
Drone Log Data Cleaner for TCN GPS Error Reduction Model (Per-File Version)

This script extracts specific sensor data from ArduPilot log files for training
a Temporal Convolutional Network (TCN) to reduce GPS positioning errors.
Saves one CSV file per input log file with all required parameters.

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

class DroneLogCleanerPerFile:
    """
    A class to parse and clean ArduPilot drone log files for TCN training.
    
    Extracts specific parameters and saves one CSV file per input log file.
    """
    
    def __init__(self):
        # Define the message types and their relevant fields for TCN training
        # Based on user requirements: TimeUS, IMU data, BARO, ATT, AHR2, XKF1, XKF3, XKF4, XKF5, MAG, VIBE, GPS
        self.target_messages = {
            'GPS': ['TimeUS', 'I', 'Status', 'NSats', 'HDop', 'Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ', 'Yaw', 'U'],
            'IMU': ['TimeUS', 'I', 'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ', 'EG', 'EA', 'T', 'GH', 'AH'],
            'ACC': ['TimeUS', 'I', 'SampleUS', 'AccX', 'AccY', 'AccZ'],
            'GYR': ['TimeUS', 'I', 'SampleUS', 'GyrX', 'GyrY', 'GyrZ'],
            'ATT': ['TimeUS', 'DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw', 'AEKF'],
            'AHR2': ['TimeUS', 'Roll', 'Pitch', 'Yaw', 'Alt', 'Lat', 'Lng'],
            'XKF1': ['TimeUS', 'C', 'Roll', 'Pitch', 'Yaw', 'VN', 'VE', 'VD', 'dPD', 'PN', 'PE', 'PD', 'GX', 'GY', 'GZ', 'OH'],
            'XKF2': ['TimeUS', 'C', 'AX', 'AY', 'AZ', 'VWN', 'VWE', 'MN', 'ME', 'MD', 'MX', 'MY', 'MZ'],
            'XKF3': ['TimeUS', 'C', 'IVN', 'IVE', 'IVD', 'IPN', 'IPE', 'IPD', 'IMX', 'IMY', 'IMZ', 'IYAW', 'IVT', 'RErr', 'ErSc'],
            'XKF4': ['TimeUS', 'C', 'SV', 'SP', 'SH', 'SM', 'SVT', 'errRP', 'OFN', 'OFE', 'FS', 'TS', 'SS', 'GPS', 'PI'],
            'XKF5': ['TimeUS', 'C', 'NI', 'FIX', 'FIY', 'AFI', 'HAGL', 'offset', 'RI', 'rng', 'Herr', 'eAng', 'eVel', 'ePos'],
            'BARO': ['TimeUS', 'I', 'Alt', 'AltAMSL', 'Press', 'Temp', 'CRt', 'SMS', 'Offset', 'GndTemp', 'Health'],
            'MAG': ['TimeUS', 'I', 'MagX', 'MagY', 'MagZ', 'OfsX', 'OfsY', 'OfsZ', 'MOX', 'MOY', 'MOZ', 'Health', 'S'],
            'VIBE': ['TimeUS', 'IMU', 'VibeX', 'VibeY', 'VibeZ', 'Clip'],
            'POS': ['TimeUS', 'Lat', 'Lng', 'Alt', 'RelHomeAlt', 'RelOriginAlt'],
        }
        
        # Define the specific parameters requested by user
        self.required_params = {
            'TimeUS': 'timestamp',
            # IMU data
            'IMU_GyrX': 'gyro_x', 'IMU_GyrY': 'gyro_y', 'IMU_GyrZ': 'gyro_z',
            'IMU_AccX': 'accel_x', 'IMU_AccY': 'accel_y', 'IMU_AccZ': 'accel_z',
            # BARO data
            'BARO_Press': 'pressure', 'BARO_Temp': 'temperature',
            # ATT and AHR2 data
            'ATT_Roll': 'att_roll', 'ATT_Pitch': 'att_pitch', 'ATT_Yaw': 'att_yaw',
            'AHR2_Roll': 'ahr2_roll', 'AHR2_Pitch': 'ahr2_pitch', 'AHR2_Yaw': 'ahr2_yaw',
            # XKF1 data
            'XKF1_VN': 'velocity_north', 'XKF1_VE': 'velocity_east', 'XKF1_VD': 'velocity_down',
            'XKF1_PN': 'position_north', 'XKF1_PE': 'position_east', 'XKF1_PD': 'position_down',
            # MAG data
            'MAG_MagX': 'mag_x', 'MAG_MagY': 'mag_y', 'MAG_MagZ': 'mag_z',
            # VIBE data
            'VIBE_VibeX': 'vibe_x', 'VIBE_VibeY': 'vibe_y', 'VIBE_VibeZ': 'vibe_z',
            # XKF3 data
            'XKF3_IVN': 'innovation_vel_north', 'XKF3_IVE': 'innovation_vel_east', 'XKF3_IVD': 'innovation_vel_down',
            # GPS data
            'GPS_Lat': 'latitude', 'GPS_Lng': 'longitude', 'GPS_Alt': 'altitude', 'GPS_Spd': 'ground_speed',
        }
        
        self.format_definitions = {}
    
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
                                value = parts[i + 1]
                                # Convert to appropriate type
                                if value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                                    data[field] = float(value)
                                elif value.isdigit():
                                    data[field] = int(value)
                                else:
                                    data[field] = value
                            except (ValueError, IndexError):
                                continue
                    
                    if data:  # Only return if we got some data
                        return data
        except Exception as e:
            logger.debug(f"Error parsing data line: {e}")
        return None
    
    def process_single_log_file(self, filepath: str, output_dir: str) -> None:
        """Process a single log file and save its cleaned data as CSV."""
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        logger.info(f"Processing file: {filepath}")
        
        # Parse the log file
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
            return
        
        # Convert to DataFrames and create time-synchronized dataset
        dfs = {}
        for msg_type, records in file_data.items():
            if records:
                df = pd.DataFrame(records)
                if 'TimeUS' in df.columns:
                    df['TimeUS'] = pd.to_numeric(df['TimeUS'], errors='coerce')
                    df = df.dropna(subset=['TimeUS'])
                    df = df.sort_values('TimeUS')
                    dfs[msg_type] = df
        
        if not dfs:
            logger.warning(f"No valid data found in {filepath}")
            return
        
        # Create unified dataset with all required parameters
        # Use the message type with the most records as base (usually IMU)
        base_msg_type = max(dfs.keys(), key=lambda k: len(dfs[k]) if 'TimeUS' in dfs[k].columns else 0)
        
        if base_msg_type not in dfs:
            logger.warning(f"No suitable base message type found in {filepath}")
            return
        
        base_df = dfs[base_msg_type][['TimeUS']].copy()
        base_df = base_df.drop_duplicates(subset=['TimeUS']).sort_values('TimeUS')
        
        # Add data from all message types using interpolation
        for msg_type, df in dfs.items():
            if msg_type == base_msg_type:
                # Add all columns from base message type
                for col in df.columns:
                    if col != 'TimeUS':
                        base_df[f'{msg_type}_{col}'] = np.interp(
                            base_df['TimeUS'].values,
                            df['TimeUS'].values,
                            df[col].fillna(0).values
                        )
            else:
                # Interpolate data from other message types
                for col in df.columns:
                    if col != 'TimeUS':
                        try:
                            base_df[f'{msg_type}_{col}'] = np.interp(
                                base_df['TimeUS'].values,
                                df['TimeUS'].values,
                                df[col].fillna(0).values
                            )
                        except Exception as e:
                            logger.debug(f"Could not interpolate {msg_type}_{col}: {e}")
        
        # Filter to only include the required parameters
        final_columns = ['TimeUS']
        for param_key in self.required_params.keys():
            if param_key != 'TimeUS':
                if param_key in base_df.columns:
                    final_columns.append(param_key)
                else:
                    # Try to find the column with different naming
                    for col in base_df.columns:
                        if param_key.replace('_', '_').lower() in col.replace('_', '_').lower():
                            final_columns.append(col)
                            break
        
        # Select only available required columns
        available_columns = [col for col in final_columns if col in base_df.columns]
        result_df = base_df[available_columns].copy()
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{base_name}_cleaned.csv")
        result_df.to_csv(output_file, index=False)
        
        # Log statistics
        total_records = len(result_df)
        logger.info(f"Saved {total_records} records with {len(available_columns)} parameters to {output_file}")
        logger.info(f"  Available parameters: {', '.join(available_columns[:10])}{'...' if len(available_columns) > 10 else ''}")
    
    def process_all_log_files(self, input_dir: str, output_dir: str = None) -> None:
        """Process all log files in the input directory."""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "cleaned_dataset")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all log files
        log_files = glob.glob(os.path.join(input_dir, "*.log"))
        logger.info(f"Found {len(log_files)} log files to process")
        
        if not log_files:
            logger.warning(f"No .log files found in {input_dir}")
            return
        
        # Process each log file
        successful_files = 0
        for log_file in log_files:
            try:
                self.process_single_log_file(log_file, output_dir)
                successful_files += 1
            except Exception as e:
                logger.error(f"Failed to process {log_file}: {e}")
        
        logger.info(f"Successfully processed {successful_files}/{len(log_files)} files")
        
        # Generate summary report
        self.generate_summary_report(output_dir, log_files, successful_files)
    
    def generate_summary_report(self, output_dir: str, log_files: List[str], successful_files: int) -> None:
        """Generate a summary report of the cleaning process."""
        report_path = os.path.join(output_dir, "processing_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("DRONE LOG DATA CLEANING SUMMARY REPORT (Per-File Processing)\n")
            f.write("=" * 65 + "\n\n")
            
            f.write(f"Total log files found: {len(log_files)}\n")
            f.write(f"Successfully processed: {successful_files}\n")
            f.write(f"Processing success rate: {successful_files/len(log_files)*100:.1f}%\n\n")
            
            f.write("EXTRACTED PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            
            for param_key, param_desc in self.required_params.items():
                f.write(f"{param_key:25}: {param_desc}\n")
            
            f.write("\nOUTPUT FILES:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Location: {output_dir}\n")
            f.write(f"Format: [original_filename]_cleaned.csv\n")
            f.write(f"One CSV file per input log file\n\n")
            
            # List some sample output files
            csv_files = glob.glob(os.path.join(output_dir, "*_cleaned.csv"))
            f.write(f"Sample output files ({min(5, len(csv_files))} of {len(csv_files)}):\n")
            for i, csv_file in enumerate(csv_files[:5]):
                f.write(f"  {os.path.basename(csv_file)}\n")
            
            f.write("\nRECOMMENDED NEXT STEPS FOR TCN TRAINING:\n")
            f.write("-" * 45 + "\n")
            f.write("1. Review individual CSV files for data quality\n")
            f.write("2. Combine multiple files if needed for training\n")
            f.write("3. Implement additional preprocessing:\n")
            f.write("   - Normalize sensor data\n")
            f.write("   - Handle outliers and noise\n")
            f.write("   - Create sliding time windows for TCN input\n")
            f.write("   - Split data into train/validation/test sets\n")
            f.write("4. Define target variable (GPS error to predict/correct)\n")
            f.write("5. Consider feature engineering:\n")
            f.write("   - Velocity and acceleration derivatives\n")
            f.write("   - Statistical features over time windows\n")
            f.write("   - Cross-correlation between sensors\n")
        
        logger.info(f"Generated processing summary: {report_path}")

def main():
    """Main function to run the data cleaning process."""
    parser = argparse.ArgumentParser(
        description="Clean drone log files for TCN GPS error reduction model training (per-file processing)"
    )
    parser.add_argument(
        "input_dir", 
        help="Directory containing ArduPilot log files"
    )
    parser.add_argument(
        "--output_dir", 
        help="Output directory for cleaned data (default: input_dir/cleaned_dataset)"
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
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "cleaned_dataset")
    
    # Create cleaner instance and process files
    cleaner = DroneLogCleanerPerFile()
    
    try:
        cleaner.process_all_log_files(args.input_dir, args.output_dir)
        logger.info("Data cleaning completed successfully!")
        print("\n" + "="*70)
        print("CLEANING COMPLETED!")
        print("="*70)
        print(f"Check the output directory: {args.output_dir}")
        print(f"• One CSV file per input log file")
        print(f"• Each CSV contains all required TCN parameters")
        print(f"• processing_summary.txt - Summary and recommendations")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
