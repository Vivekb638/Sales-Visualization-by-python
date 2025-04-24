import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, levene
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

def hypothesis_testing(df):
    if df is None or df.empty:
        print("Error: No data for hypothesis testing.")
        return

    print("\n=== Hypothesis Testing ===")

    # Test 1: Difference in Sales Amount Between Genders (T-test)
    print("\nTest 1: Difference in Sales Amount Between Genders")
    female_sales = df[df['Gender'] == 'Female']['Amount'].dropna()
    male_sales = df[df['Gender'] == 'Male']['Amount'].dropna()

    if len(female_sales) > 0 and len(male_sales) > 0:
      
        levene_stat, levene_p = levene(female_sales, male_sales)
        print(f"Levene's Test for Equal Variances: p-value = {levene_p:.4f}")
        equal_var = levene_p > 0.05

        # Perform t-test
        t_stat, p_value = ttest_ind(female_sales, male_sales, equal_var=equal_var)
        print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("Result: Reject null hypothesis - Significant difference in sales between genders.")
        else:
            print("Result: Fail to reject null hypothesis - No significant difference in sales between genders.")
    else:
        print("Error: Insufficient data for gender sales test.")

    # Test 2: Difference in Sales Across Top Categories (ANOVA)
    print("\nTest 2: Difference in Sales Across Top Categories")
    top_categories = df.groupby('Category')['Amount'].sum().nlargest(3).index
    category_sales = [df[df['Category'] == cat]['Amount'].dropna() for cat in top_categories]

    if all(len(sales) > 0 for sales in category_sales):
        # Check equal variances
        levene_stat_anova, levene_p_anova = levene(*category_sales)
        print(f"Levene's Test for Equal Variances: p-value = {levene_p_anova:.4f}")

        # Perform ANOVA
        f_stat, p_value_anova = f_oneway(*category_sales)
        print(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value_anova:.4f}")
        if p_value_anova < 0.05:
            print(f"Result: Reject null hypothesis - Significant difference in sales across categories {top_categories.tolist()}.")
        else:
            print(f"Result: Fail to reject null hypothesis - No significant difference in sales across categories {top_categories.tolist()}.")
    else:
        print("Error: Insufficient data for category sales test.")

    # Test 3: Association Between Channel and Order Status (Chi-square)
    print("\nTest 3: Association Between Channel and Order Status")
    contingency_table = pd.crosstab(df['Channel'], df['Status'])
    if not contingency_table.empty:
        chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-square Test: Chi2-statistic = {chi2_stat:.4f}, p-value = {p_value_chi2:.4f}, degrees of freedom = {dof}")
        if (expected < 5).any():
            print("Warning: Some expected counts are less than 5, Chi-square results may be unreliable.")
        if p_value_chi2 < 0.05:
            print("Result: Reject null hypothesis - Significant association between channel and order status.")
        else:
            print("Result: Fail to reject null hypothesis - No significant association between channel and order status.")
    else:
        print("Error: Insufficient data for channel-status test.")

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

def main():
    
    file_path = r'E:\Python Project 1\Vrinda Store Data Analysis.xlsx'
    

    df = load_data(file_path)
    
    df = clean_data(df)
    
    hypothesis_testing(df)

    visualize_data(df)
    
    print("\nData cleaning, hypothesis testing, and visualizations completed successfully.")

if __name__ == "__main__":
    main()


