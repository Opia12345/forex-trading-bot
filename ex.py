"""
Excel Data Cleaner
Removes duplicate names, filters commission = 1.35, and formats dates without time
"""

import pandas as pd
import os
import sys
from datetime import datetime


def clean_excel_data(input_file: str, output_file: str = None):
    """
    Remove duplicate names, filter commission = 1.35, and format dates without time
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file (optional, auto-generated if None)
    """
    
    print("="*70)
    print("Excel Data Cleaner - Duplicates, Commission & Date Formatting")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Load Excel file
    print(f"\n📂 Loading file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        original_rows = len(df)
        print(f"✅ Loaded successfully!")
        print(f"   Rows: {original_rows}")
        print(f"   Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Show available columns
    print(f"\n📊 Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Find name column (case-insensitive search)
    name_col = None
    possible_names = ['name', 'Name', 'NAME', 'names', 'Names', 'username', 'user', 'User']
    
    for col in df.columns:
        if col in possible_names or 'name' in col.lower():
            name_col = col
            break
    
    # If not found automatically, ask user
    if name_col is None:
        print("\n⚠️  Could not find 'Name' column automatically.")
        col_input = input("Enter the column name or number for names: ").strip()
        
        # Check if user entered a number
        if col_input.isdigit():
            col_index = int(col_input) - 1
            if 0 <= col_index < len(df.columns):
                name_col = df.columns[col_index]
            else:
                print(f"❌ Invalid column number!")
                sys.exit(1)
        else:
            # User entered column name
            if col_input in df.columns:
                name_col = col_input
            else:
                print(f"❌ Column '{col_input}' not found!")
                sys.exit(1)
    
    print(f"\n✅ Using name column: '{name_col}'")
    
    # Show preview of names
    print(f"\n📊 Preview of '{name_col}' column:")
    print(df[name_col].head(10))
    
    # Count duplicates
    duplicate_count = df[name_col].duplicated().sum()
    
    print(f"\n🔍 Found {duplicate_count} duplicate names")
    
    # Remove duplicates (keep first occurrence) - use .copy() to avoid warning
    if duplicate_count > 0:
        print(f"\n🗑️  Removing {duplicate_count} duplicate rows (keeping first occurrence)...")
        df_cleaned = df.drop_duplicates(subset=[name_col], keep='first').copy()
    else:
        print("✅ No duplicate names found.")
        df_cleaned = df.copy()
    
    rows_after_dedup = len(df_cleaned)
    
    # Find commission column
    commission_col = None
    possible_comm_names = ['commission', 'Commission', 'COMMISSION', 'comm', 'Comm']
    
    for col in df_cleaned.columns:
        if col in possible_comm_names or 'commission' in col.lower():
            commission_col = col
            break
    
    # Filter commission = 1.35
    commission_removed = 0
    if commission_col:
        print(f"\n✅ Found commission column: '{commission_col}'")
        
        # Count rows with commission = 1.35
        rows_with_135 = (df_cleaned[commission_col] == 1.35).sum()
        
        if rows_with_135 > 0:
            print(f"🔍 Found {rows_with_135} rows with commission = 1.35")
            print(f"🗑️  Removing rows with commission = 1.35...")
            
            df_cleaned = df_cleaned[df_cleaned[commission_col] != 1.35].copy()
            commission_removed = rows_with_135
            print(f"✅ Removed {commission_removed} rows")
        else:
            print(f"✅ No rows with commission = 1.35 found")
    else:
        print("\n⚠️  Commission column not found - skipping commission filter")
    
    # Format dates - find all date columns
    print(f"\n📅 Searching for date columns...")
    date_columns = []
    
    for col in df_cleaned.columns:
        # Check if column contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
            date_columns.append(col)
        # Check if column might contain dates as strings or objects
        elif df_cleaned[col].dtype == 'object':
            # Sample first non-null value
            sample = df_cleaned[col].dropna().head(1)
            if not sample.empty:
                try:
                    pd.to_datetime(sample.iloc[0])
                    date_columns.append(col)
                except:
                    pass
    
    if date_columns:
        print(f"✅ Found {len(date_columns)} date column(s): {', '.join(date_columns)}")
        
        for col in date_columns:
            print(f"\n📅 Formatting '{col}' column...")
            # Convert to datetime if not already
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            # Format as date only (YYYY-MM-DD)
            df_cleaned[col] = df_cleaned[col].dt.date
            print(f"   ✅ Formatted to date only (no time)")
    else:
        print("⚠️  No date columns detected. If you have date columns, they may already be formatted.")
    
    remaining_rows = len(df_cleaned)
    total_removed = original_rows - remaining_rows
    
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    # Save to Excel
    print(f"\n💾 Saving cleaned file: {output_file}")
    try:
        df_cleaned.to_excel(output_file, index=False)
        print(f"✅ File saved successfully!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)
    
    # Show summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Original file:           {input_file}")
    print(f"Cleaned file:            {output_file}")
    print(f"Original rows:           {original_rows}")
    print(f"Duplicates removed:      {duplicate_count}")
    print(f"Commission 1.35 removed: {commission_removed}")
    print(f"Total rows removed:      {total_removed}")
    print(f"Remaining rows:          {remaining_rows}")
    print(f"Date columns formatted:  {len(date_columns)}")
    if total_removed > 0:
        print(f"Removal rate:            {(total_removed/original_rows*100):.1f}%")
    print("="*70)
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove duplicate names, filter commission = 1.35, and format dates'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input Excel file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output Excel file path (optional)'
    )
    
    args = parser.parse_args()
    
    # If no file provided, ask for it
    if not args.input_file:
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║     Excel Data Cleaner - Duplicates, Commission & Dates         ║")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
        
        input_file = input("Enter Excel file path: ").strip().strip('"').strip("'")
        
        if not input_file:
            print("❌ No file provided!")
            sys.exit(1)
    else:
        input_file = args.input_file
    
    # Run cleaner
    clean_excel_data(input_file, args.output)