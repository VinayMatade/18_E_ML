#!/usr/bin/env python3
"""
Enhanced Drone Log Data Extractor for TCN Training

This script extracts specific sensor parameters from ArduPilot log files
and saves them as individual CSV files per log, with all required data properly parsed.

Parameters extracted:
- TimeUS: Timestamp
- IMU: GyrX, GyrY, GyrZ, AccX, AccY, AccZ  
- BARO: Press, Temp
- ATT: Roll, Pitch, Yaw
- AHR2: Roll, Pitch, Yaw
- XKF1: VN, VE, VD, PN, PE, PD
- XKF3: IVN, IVE, IVD  
- MAG: MagX, MagY, MagZ
- VIBE: VibeX, VibeY, VibeZ
- GPS: Lat, Lng, Alt, Spd

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TCNDataExtractor:
    """Extract specific parameters from ArduPilot logs for TCN training."""
    
    def __init__(self):
        self.format_definitions = {}
        
        # Define required message types and fields
        self.required_messages = {
            'GPS': ['TimeUS', 'Lat', 'Lng', 'Alt', 'Spd'],
            'IMU': ['TimeUS', 'GyrX', 'GyrY', 'GyrZ', 'AccX', 'AccY', 'AccZ'],
            'BARO': ['TimeUS', 'Press', 'Temp'],
            'ATT': ['TimeUS', 'Roll', 'Pitch', 'Yaw'],
            'AHR2': ['TimeUS', 'Roll', 'Pitch', 'Yaw'],
            'XKF1': ['TimeUS', 'VN', 'VE', 'VD', 'PN', 'PE', 'PD'],
            'XKF3': ['TimeUS', 'IVN', 'IVE', 'IVD'],
            'XKF4': ['TimeUS'],  # Include for timestamp alignment
            'XKF5': ['TimeUS'],  # Include for timestamp alignment  
            'MAG': ['TimeUS', 'MagX', 'MagY', 'MagZ'],
            'VIBE': ['TimeUS', 'VibeX', 'VibeY', 'VibeZ'],
        }
    
    def parse_format_line(self, line):
        """Parse format definition line."""
        try:
            parts = [p.strip() for p in line.split(',')]
            # Ensure it's an FMT line
            if len(parts) >= 6 and parts[0].strip() == 'FMT':
                msg_type = parts[3]
                if msg_type in self.required_messages:
                    # All field names are from index 5 onward
                    field_names = [f.strip() for f in parts[5:]]
                    if field_names:
                        self.format_definitions[msg_type] = field_names
                        logger.debug(f"Found format for {msg_type}: {field_names}")
        except Exception as e:
            logger.debug(f"Error parsing format line: {e}")
    
    def parse_data_line(self, line, msg_type):
        """Parse data line for specific message type."""
        try:
            lstrip = line.lstrip()
            # Strictly match data lines (avoid matching FMT lines)
            if not (lstrip.startswith(f'{msg_type},') or f'|{msg_type},' in lstrip):
                return None

            # Split into comma-separated tokens
            parts_raw = line.split(',')
            parts = [p.strip() for p in parts_raw]
            if len(parts) < 2:
                return None

            # Handle optional leading line-number prefix like "123|GPS"
            if '|' in parts[0]:
                # Extract token after '|', rebuild list so parts[0] is the message type
                msg_tok = parts[0].split('|')[-1].strip()
                parts = [msg_tok] + parts[1:]

            # Validate message type
            if parts[0] != msg_type:
                return None

            # Get field definitions for this message type
            field_names = self.format_definitions.get(msg_type)
            if not field_names:
                return None

            required_fields = self.required_messages[msg_type]

            # Create data dictionary by aligning fields with values (offset by 1)
            data = {}
            for i, field_name in enumerate(field_names):
                if field_name in required_fields and (i + 1) < len(parts):
                    value_str = parts[i + 1]
                    try:
                        # Try numeric conversion
                        if value_str.lower().replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('e', '', 1).isdigit():
                            # Float if contains '.' or exponent
                            if ('.' in value_str) or ('e' in value_str.lower()):
                                data[field_name] = float(value_str)
                            else:
                                data[field_name] = int(value_str)
                        else:
                            data[field_name] = float(value_str)
                    except Exception:
                        data[field_name] = value_str

            return data if data else None
        except Exception as e:
            logger.debug(f"Error parsing data line: {e}")
            return None
    
    def process_single_file(self, log_file_path, output_dir):
        """Process a single log file and extract required parameters."""
        filename = os.path.basename(log_file_path)
        base_name = os.path.splitext(filename)[0]
        
        logger.info(f"Processing: {filename}")
        
        # Reset format definitions for each file
        self.format_definitions = {}
        
        # Store data for each message type
        extracted_data = {msg_type: [] for msg_type in self.required_messages.keys()}
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                # First pass: extract format definitions
                for line in lines:
                    if 'FMT,' in line:
                        self.parse_format_line(line)
                
                # Second pass: extract data
                for line in lines:
                    for msg_type in self.required_messages.keys():
                        if f'{msg_type},' in line:
                            data = self.parse_data_line(line, msg_type)
                            if data:
                                extracted_data[msg_type].append(data)
                
        except Exception as e:
            logger.error(f"Error reading file {log_file_path}: {e}")
            return
        
        # Convert to DataFrames and create unified dataset
        dfs = {}
        for msg_type, records in extracted_data.items():
            if records:
                df = pd.DataFrame(records)
                if 'TimeUS' in df.columns:
                    df['TimeUS'] = pd.to_numeric(df['TimeUS'], errors='coerce')
                    df = df.dropna(subset=['TimeUS']).sort_values('TimeUS')
                    dfs[msg_type] = df
                    logger.debug(f"{msg_type}: {len(df)} records")
        
        if not dfs:
            logger.warning(f"No valid data extracted from {filename}")
            return
        
        # Create time-synchronized dataset using the most frequent message as base
        # Usually IMU has the highest frequency
        base_msg = max(dfs.keys(), key=lambda k: len(dfs[k]))
        base_df = dfs[base_msg][['TimeUS']].copy()
        
        # Add data from each message type with proper column naming
        final_columns = ['TimeUS']
        
        for msg_type, df in dfs.items():
            for col in df.columns:
                if col != 'TimeUS':
                    new_col_name = f"{msg_type}_{col}"
                    
                    if msg_type == base_msg:
                        # Direct assignment for base message
                        base_df[new_col_name] = df[col].values
                    else:
                        # Interpolate for other messages
                        try:
                            base_df[new_col_name] = np.interp(
                                base_df['TimeUS'].values,
                                df['TimeUS'].values,
                                df[col].fillna(0).values
                            )
                        except Exception as e:
                            logger.debug(f"Could not interpolate {new_col_name}: {e}")
                    
                    final_columns.append(new_col_name)
        
        # Save the results
        output_file = os.path.join(output_dir, f"{base_name}_tcn_data.csv")
        base_df.to_csv(output_file, index=False)
        
        # Print summary
        total_params = len([col for col in base_df.columns if col != 'TimeUS'])
        logger.info(f"✓ Saved {len(base_df)} records with {total_params} parameters")
        logger.info(f"  Output: {os.path.basename(output_file)}")
        
        # Show available parameters
        param_summary = []
        for msg_type, df in dfs.items():
            params = [col for col in df.columns if col != 'TimeUS']
            if params:
                param_summary.append(f"{msg_type}({len(params)})")
        
        logger.info(f"  Parameters: {', '.join(param_summary)}")
        
        return len(base_df), total_params
    
    def process_all_files(self, input_dir, output_dir=None):
        """Process all log files in the input directory."""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "cleaned_dataset")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all log files
        log_files = glob.glob(os.path.join(input_dir, "*.log"))
        logger.info(f"Found {len(log_files)} log files")
        
        if not log_files:
            logger.error(f"No .log files found in {input_dir}")
            return
        
        # Process each file
        total_records = 0
        successful_files = 0
        
        for log_file in sorted(log_files):
            try:
                result = self.process_single_file(log_file, output_dir)
                if result:
                    records, params = result
                    total_records += records
                    successful_files += 1
            except Exception as e:
                logger.error(f"Failed to process {os.path.basename(log_file)}: {e}")
        
        # Generate summary
        self.generate_summary(output_dir, len(log_files), successful_files, total_records)
        
        logger.info(f"\nProcessing complete!")
        logger.info(f"Successfully processed: {successful_files}/{len(log_files)} files")
        logger.info(f"Total records extracted: {total_records:,}")
        logger.info(f"Output directory: {output_dir}")
    
    def generate_summary(self, output_dir, total_files, successful_files, total_records):
        """Generate processing summary report."""
        summary_file = os.path.join(output_dir, "tcn_extraction_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("TCN DATA EXTRACTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Files processed: {successful_files}/{total_files}\n")
            f.write(f"Success rate: {successful_files/total_files*100:.1f}%\n")
            f.write(f"Total records: {total_records:,}\n\n")
            
            f.write("EXTRACTED PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            
            for msg_type, fields in self.required_messages.items():
                params = [f for f in fields if f != 'TimeUS']
                if params:
                    f.write(f"{msg_type:8}: {', '.join(params)}\n")
            
            f.write(f"\nOUTPUT FORMAT:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"• One CSV file per input log file\n")
            f.write(f"• Filename format: [original_name]_tcn_data.csv\n")
            f.write(f"• Time-synchronized data with interpolation\n")
            f.write(f"• Column format: [MESSAGE_TYPE]_[PARAMETER]\n\n")
            
            f.write("NEXT STEPS FOR TCN TRAINING:\n")
            f.write("-" * 30 + "\n")
            f.write("1. Review CSV files for data quality\n")
            f.write("2. Combine multiple files for larger dataset\n")
            f.write("3. Normalize and scale the data\n")
            f.write("4. Create sliding time windows\n")
            f.write("5. Define target variables (GPS error)\n")
            f.write("6. Split into train/validation/test sets\n")
        
        logger.info(f"Summary saved: {os.path.basename(summary_file)}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract specific parameters from ArduPilot logs for TCN training"
    )
    parser.add_argument("input_dir", help="Directory containing .log files")
    parser.add_argument("--output_dir", help="Output directory (default: input_dir/cleaned_dataset)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    extractor = TCNDataExtractor()
    extractor.process_all_files(args.input_dir, args.output_dir)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETED!")
    print("="*60)
    output_dir = args.output_dir or os.path.join(args.input_dir, "cleaned_dataset")
    print(f"Check output directory: {output_dir}")
    print("• Individual CSV files with all required TCN parameters")
    print("• Time-synchronized data ready for TCN training")
    print("• Summary report with extraction statistics")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())
