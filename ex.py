"""
Excel Commission Filter
Deletes rows where commission is exactly 1.35 and saves the cleaned file
"""

import pandas as pd
import os
import sys


def filter_commission(input_file: str, output_file: str = None):
    """
    Delete rows where commission is exactly 1.35
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file (optional, auto-generated if None)
    """
    
    print("="*70)
    print("Excel Commission Filter - Delete rows with commission = 1.35")
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
    
    # Find commission column (case-insensitive search)
    commission_col = None
    possible_names = ['commission', 'Commission', 'COMMISSION', 'comm', 'Comm']
    
    for col in df.columns:
        if col in possible_names or 'commission' in col.lower():
            commission_col = col
            break
    
    # If not found automatically, ask user
    if commission_col is None:
        print("\n⚠️  Could not find 'Commission' column automatically.")
        col_input = input("Enter the column name or number for commission: ").strip()
        
        # Check if user entered a number
        if col_input.isdigit():
            col_index = int(col_input) - 1
            if 0 <= col_index < len(df.columns):
                commission_col = df.columns[col_index]
            else:
                print(f"❌ Invalid column number!")
                sys.exit(1)
        else:
            # User entered column name
            if col_input in df.columns:
                commission_col = col_input
            else:
                print(f"❌ Column '{col_input}' not found!")
                sys.exit(1)
    
    print(f"\n✅ Using column: '{commission_col}'")
    
    # Show preview of commission values
    print(f"\n📊 Preview of '{commission_col}' column:")
    print(df[commission_col].head(10))
    
    # Count rows with commission = 1.35
    rows_to_delete = df[commission_col] == 1.35
    delete_count = rows_to_delete.sum()
    
    print(f"\n🔍 Found {delete_count} rows with commission = 1.35")
    
    if delete_count == 0:
        print("✅ No rows to delete. File is already clean.")
        return
    
    # Delete rows where commission = 1.35
    print(f"\n🗑️  Deleting {delete_count} rows...")
    df_filtered = df[df[commission_col] != 1.35]
    
    remaining_rows = len(df_filtered)
    
    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"
    
    # Save to Excel
    print(f"\n💾 Saving filtered file: {output_file}")
    try:
        df_filtered.to_excel(output_file, index=False)
        print(f"✅ File saved successfully!")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)
    
    # Show summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Original file:     {input_file}")
    print(f"Filtered file:     {output_file}")
    print(f"Original rows:     {original_rows}")
    print(f"Deleted rows:      {delete_count}")
    print(f"Remaining rows:    {remaining_rows}")
    print(f"Deletion rate:     {(delete_count/original_rows*100):.1f}%")
    print("="*70)
    print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Delete rows where commission is exactly 1.35'
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
        print("║       Excel Commission Filter - Delete Commission = 1.35         ║")
        print("╚══════════════════════════════════════════════════════════════════╝\n")
        
        input_file = input("Enter Excel file path: ").strip().strip('"').strip("'")
        
        if not input_file:
            print("❌ No file provided!")
            sys.exit(1)
    else:
        input_file = args.input_file
    
    # Run filter
    filter_commission(input_file, args.output)