"""
Excel Date Formatter
Formats date columns to human-readable dates without time
"""

import pandas as pd
import os
import sys


def format_excel_dates(input_file: str, output_file: str = None):
    """
    Format date columns to remove time and display only the date
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file (optional, auto-generated if None)
    """
    
    print("="*70)
    print("Excel Date Formatter - Remove Time from Dates")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Load Excel file
    print(f"\n📂 Loading file: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✅ Loaded successfully!")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Find all date columns
    print(f"\n📅 Searching for date columns...")
    date_columns = []
    
    for col in df.columns:
        # Check if column contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
        # Check if column might contain dates as strings or objects
        elif df[col].dtype == 'object':
            # Sample first non-null value
            sample = df[col].dropna().head(1)
            if not sample.empty:
                try:
                    pd.to_datetime(sample.iloc[0])
                    date_columns.append(col)
                except:
                    pass
    
    if not date_columns:
        print("⚠️  No date columns detected.")
        print("   The file may not contain date columns or they are already formatted.")
        response = input("\nDo you want to save the file anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled.")
            sys.exit(0)
    else:
        print(f"✅ Found {len(date_columns)} date column(s):")
        for col in date_columns:
            print(f"   • {col}")
        
        # Format each date column
        for col in date_columns:
            print(f"\n📅 Formatting '{col}'...")
            
            # Show sample before
            sample_before = df[col].dropna().head(3)
            if not sample_before.empty:
                print(f"   Before: {sample_before.iloc[0]}")
            
            # Convert to datetime if not already
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Format as date only (YYYY-MM-DD)
            df[col] = df[col].dt.date
            
            # Show sample after
            sample_after = df[col].dropna().head(3)
            if not sample_after.empty:
                print(f"   After:  {sample_after.iloc[0]}")
            
            print(f"   ✅ Formatted successfully!")
    
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_formatted{ext}"
    
    # Save to Excel
    print(f"\n💾 Saving formatted file: {output_file}")
    try:
        df.to_excel(output_file, index=False)
        print(f"✅ File saved successfully!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)
    
    # Show summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Original file:          {input_file}")
    print(f"Formatted file:         {output_file}")
    print(f"Date columns formatted: {len(date_columns)}")
    if date_columns:
        print(f"Columns formatted:      {', '.join(date_columns)}")
    print("="*70)
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Format date columns in Excel to remove time'
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
        print("║            Excel Date Formatter - Remove Time                   ║")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
        
        input_file = input("Enter Excel file path: ").strip().strip('"').strip("'")
        
        if not input_file:
            print("❌ No file provided!")
            sys.exit(1)
    else:
        input_file = args.input_file
    
    # Run formatter
    format_excel_dates(input_file, args.output)