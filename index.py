import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


sns.set_style('whitegrid')

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    if df is None:
        print("Error: No data to clean.")
        return None
  
    df.rename(columns={'Channel ': 'Channel'}, inplace=True)

    def clean_gender(gender):
        if pd.isna(gender):
            return np.nan
        gender = str(gender).strip().lower()
        if gender in ['women', 'w', 'female']:
            return 'Female'
        elif gender in ['men', 'm', 'male']:
            return 'Male'
        else:
            return np.nan

    df['Gender'] = df['Gender'].apply(clean_gender)

    # Clean Qty column
    qty_map = {'one': 1, 'two': 2, 'three': 3, 'One': 1, 'Two': 2, 'Three': 3}
    df['Qty'] = df['Qty'].apply(lambda x: qty_map.get(str(x).lower(), x) if isinstance(x, str) else x)
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

    # Clean Channel column
    df['Channel'] = df['Channel'].str.strip().str.title().replace('', np.nan)

    # Ensure numeric and datetime columns
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', unit='D')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Create Age Group column
    def get_age_group(age):
        if pd.isna(age):
            return np.nan
        elif 3 <= age <= 18:
            return 'Teenager'
        elif 19 <= age <= 64:
            return 'Adult'
        elif age >= 65:
            return 'Senior'
        else:
            return 'Other'

    df['Age Group'] = df['Age'].apply(get_age_group)

    print("\nDataset Info After Cleaning:")
    df.info()
    return df

def run_statistical_tests(df):
    if df is None or df.empty:
        print("Error: No data for statistical testing.")
        return

    print("\n🔬 Running Statistical Tests:")

    # Test 1: T-Test - Compare sales between two genders
    print("\nT-Test between Female and Male Sales:")
    female_sales = df[df['Gender'] == 'Female']['Amount'].dropna()
    male_sales = df[df['Gender'] == 'Male']['Amount'].dropna()

    if len(female_sales) > 0 and len(male_sales) > 0:
        t_stat, p_val = stats.ttest_ind(female_sales, male_sales, equal_var=False)
        
        print(f"   T-statistic = {t_stat:.4f}")
        print(f"   P-value = {p_val:.4f}")
        
        if p_val < 0.05:
            print("   → Statistically significant difference in sales between genders.")
        else:
            print("   → No significant difference found in sales between genders.")
    else:
        print("   No sufficient data for gender comparison.")

    # Test 2: T-Test - Compare sales between top two categories
    top_categories = df.groupby('Category')['Amount'].sum().nlargest(2).index.tolist()
    
    if len(top_categories) >= 2:
        cat1, cat2 = top_categories[0], top_categories[1]
        print(f"\nT-Test between {cat1} and {cat2} Categories:")
        
        cat1_sales = df[df['Category'] == cat1]['Amount'].dropna()
        cat2_sales = df[df['Category'] == cat2]['Amount'].dropna()
        
        if len(cat1_sales) > 0 and len(cat2_sales) > 0:
            t_stat, p_val = stats.ttest_ind(cat1_sales, cat2_sales, equal_var=False)
            
            print(f"   T-statistic = {t_stat:.4f}")
            print(f"   P-value = {p_val:.4f}")
            
            if p_val < 0.05:
                print(f"   → Statistically significant difference in sales between {cat1} and {cat2}.")
            else:
                print(f"   → No significant difference found in sales between categories.")
        else:
            print("   No sufficient data for category comparison.")
    else:
        print("   Not enough categories for comparison.")

    # Test 3: Z-Test - Compare channel sales to overall mean
    top_channel = df['Channel'].value_counts().index[0] if 'Channel' in df.columns and not df['Channel'].empty else None
    
    if top_channel:
        print(f"\n📌 Z-Test for {top_channel} vs overall sales mean:")
        
        channel_sales = df[df['Channel'] == top_channel]['Amount'].dropna()
        overall_mean = df['Amount'].mean()
        overall_std = df['Amount'].std()
        sample_size = len(channel_sales)
        
        if sample_size > 0:
            z_score = (channel_sales.mean() - overall_mean) / (overall_std / np.sqrt(sample_size))
            # Two-tailed p-value
            p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"   Z-score = {z_score:.4f}")
            print(f"   P-value = {p_val:.4f}")
            
            if p_val < 0.05:
                print(f"   → {top_channel} has significantly different sales from the overall average.")
            else:
                print(f"   → No significant difference from the overall sales average.")
        else:
            print(f"   No data found for channel: {top_channel}")
    else:
        print("   No channel data available for Z-test.")

    # Test 4: Z-Test - Compare state sales to overall mean
    if 'ship-state' in df.columns:
        top_state = df['ship-state'].value_counts().index[0]
        print(f"\n📌 Z-Test for {top_state} vs overall sales mean:")
        
        state_sales = df[df['ship-state'] == top_state]['Amount'].dropna()
        overall_mean = df['Amount'].mean()
        overall_std = df['Amount'].std()
        sample_size = len(state_sales)
        
        if sample_size > 0:
            z_score = (state_sales.mean() - overall_mean) / (overall_std / np.sqrt(sample_size))
            # Two-tailed p-value
            p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"   Z-score = {z_score:.4f}")
            print(f"   P-value = {p_val:.4f}")
            
            if p_val < 0.05:
                print(f"   → {top_state} has significantly different sales from the overall average.")
            else:
                print(f"   → No significant difference from the overall sales average.")
        else:
            print(f"   No data found for state: {top_state}")

def visualize_data(df):
    if df is None or df.empty:
        print("Error: No data for visualization.")
        return

    # Formatter for K format
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}K'
        return f'{int(x)}'

    # Objective 1: Total Sales by Ship State (Top 10)
    top_states = df.groupby('ship-state')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='ship-state', y='Amount', data=top_states, palette='Blues_d')
    plt.title('Top 10 Ship States by Sales Amount', fontsize=14)
    plt.xlabel('Ship State', fontsize=12)
    plt.ylabel('Total Sales (INR)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tight_layout()
    plt.show()

    # Objective 2: Sales Distribution by Gender
    sales_by_gender = df.groupby('Gender')['Amount'].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(sales_by_gender, labels=sales_by_gender.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Sales Distribution by Gender', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Objective 3: Category-wise Sales Distribution
    sales_by_category = df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Category', y='Amount', data=sales_by_category, palette='Greens_d')
    plt.title('Sales Distribution by Product Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Sales (INR)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tight_layout()
    plt.show()
    
    # NEW VISUALIZATION: Box Plot of Sales Amount by Age Group
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Age Group', y='Amount', data=df, palette='viridis')
    plt.title('Distribution of Sales Amount by Age Group', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Sales Amount (INR)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Objective 4: Channel-wise Order Volume
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Channel', data=df, order=df['Channel'].value_counts().index, palette='Purples_d')
    plt.title('Order Volume by Sales Channel', fontsize=14)
    plt.xlabel('Number of Orders', fontsize=12)
    plt.ylabel('Channel', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Objective 5: Order Status Breakdown
    status_counts = df['Status'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
    plt.title('Order Status Breakdown', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # NEW VISUALIZATION: Correlation Heatmap
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) >= 2:  # Need at least 2 numeric columns for correlation
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Create mask for upper triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Heatmap of Numeric Variables', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns for correlation heatmap.")

def main():
    
    file_path = r'E:\Python Project 1\Vrinda Store Data Analysis.xlsx'
    
    df = load_data(file_path)
    
    df = clean_data(df)
    
    run_statistical_tests(df)  # Updated function name

    visualize_data(df)
    
    print("\nData cleaning, statistical testing, and visualizations completed successfully.")

if __name__ == "__main__":
    main()
