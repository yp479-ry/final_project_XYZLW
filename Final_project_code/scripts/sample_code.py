"""
=================================================
Flight Delay Analysis and Prediction - DS2500 Final Project
=================================================
 Created On: Sat, Nov 2, 2024, 12:44:03

**Team Member:** Shengbin Lei, Yufan Peng, Ziqin Ma
**Instructor:** Professor Rush Sanghrajka
**Course:** DS2500

=================================================

## Project Overview:
This project investigates flight delays in the United States to uncover patterns, contributing factors, and actionable insights.
It leverages machine learning models and statistical analysis to provide recommendations for travelers and airline operations.

### Objectives:
1. **Understand Delay Factors**: Explore the reasons for flight delays and their impact across airlines and routes.
2. **Enhance Traveler Decisions**: Use data insights to help passengers make better travel choices.
3. **Support Airline Operations**: Predict delays to improve airline efficiency and customer satisfaction.

### Tools and Technologies:
- **Libraries**: pandas, seaborn, matplotlib, scikit-learn
- **Data Source**:
  U.S. Department of Transportation (Bureau of Transportation Statistics)

### Dataset Details:
- **Coverage**: Flight data from August 2023 to August 2024.
- **Key Variables**:
  - `DEP_DELAY`: The difference (in minutes) between the scheduled and actual departure time. Negative values indicate early departures.
  - `DEP_DELAY_NEW`: The difference (in minutes) between scheduled and actual departure time, with early departures set to 0.
  - `DEP_DEL15`: Indicates if a flight was delayed by 15 minutes or more (1 = Yes).
  - **Delay Breakdown by Reason**:
    1. **Carrier Delay**: Caused by airline-controlled issues such as crew, maintenance, or scheduling problems.
    2. **Weather Delay**: Results from adverse weather conditions affecting flight safety or operations.
    3. **NAS Delay**: Arises from National Airspace System (NAS) issues, including air traffic control, airport congestion, or airspace management.
    4. **Security Delay**: Due to enhanced security measures or disruptions at checkpoints.
    5. **Late Aircraft Delay**: Occurs when a previous flight arrives late, delaying the next scheduled flight.
  - Distance and flight duration metrics, such as `DISTANCE` (miles flown) and `AIR_TIME` (actual flight time in minutes).

=================================================

## Project Steps:
1. **Data Loading and Cleaning**:
   - Combine all monthly datasets.
   - Handle missing values and clean the delay-related columns appropriately.

2. **Exploratory Data Analysis**:
   - Analyze trends in delays across airlines, routes, and months.
   - Visualize relationships between delay factors using bar plots, pie charts, and heatmaps.

3. **Airline Performance Evaluation**:
   - Identify the top-performing airlines based on delay ratios and average delay time.
   - Categorize airlines into best and worst performers.

4. **Prediction Models**:
   - **Random Forest Classifier**: Predict if a flight will be delayed by 15+ minutes.
   - **Linear Regression**: Estimate the delay duration based on contributing factors.

5. **Visualization**:
   - Generate actionable visualizations, including scatter plots, bar charts, and density plots.

=================================================

## Key Outputs:
1. **Descriptive Statistics**:
   - Overall trends in flight delays (e.g., average delay time and early departure rates).
   - Breakdown of delays by reason and distance.

2. **Airline Performance**:
   - Top 5 best and worst-performing airlines based on delay metrics.
   - Detailed delay contribution analysis for each airline.

3. **Predictive Insights**:
   - Feature importance for predicting delays.
   - Delay likelihood predictions for individual airlines.

4. **Actionable Visualizations**:
   - Delay factor trends across distances and time.
   - Density plots of actual vs. predicted delays.
   - Heatmaps for delay factor correlations.

=================================================
"""

import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
    r2_score
)


# Define required columns globally
REQUIRED_COLUMNS = [
    "YEAR", "QUARTER", "MONTH", "OP_UNIQUE_CARRIER", "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_NM", "DEST_CITY_NAME", "DEST_STATE_NM", "DEP_DELAY",
    "DEP_DELAY_NEW", "DEP_DEL15", "DEP_DELAY_GROUP", "ACTUAL_ELAPSED_TIME",
    "AIR_TIME", "DISTANCE", "CARRIER_DELAY", "WEATHER_DELAY",
    "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
]


def load_and_prepare_data(directory):
    """
    Load and combine CSV files from a specified directory into a single DataFrame.

    Parameters:
    - directory (str): Path to the directory containing CSV files.

    Returns:
    - pandas.DataFrame: Combined DataFrame with all data.

    Raises:
    - FileNotFoundError: If no CSV files are found in the directory.
    """
    csv_files = glob.glob(f"{directory}/*.csv")
    print(f"Files detected: {csv_files}")

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows of data.")
    return combined_df


def clean_data(df, delay_columns):
    """
    Clean the dataset by handling missing values and ensuring valid delay data.

    Parameters:
    - df (pandas.DataFrame): Input dataset.
    - delay_columns (list): Columns to check for missing and invalid values.

    Returns:
    - pandas.DataFrame: Cleaned dataset with:
        - NaN rows in delay columns removed.
        - Non-negative values enforced for delay factor columns.
    """
    df_cleaned = df.dropna(subset=delay_columns)

    delay_factors = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    for col in delay_factors:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(lambda x: max(x, 0))

    return df_cleaned


def overall_delay_analysis(df):
    """
    Calculate averages and percentages for key metrics in a dataset.

    Parameters:
    - df (pandas.DataFrame): Dataset with relevant metrics.

    Returns:
    - tuple: Averages and percentages for specified metrics.
    """
    avg_delay_incl_neg = df['DEP_DELAY'].mean()
    avg_delay_excl_neg = df['DEP_DELAY_NEW'].mean()
    early_departures = (df['DEP_DELAY'] < 0).sum()
    total_flights = len(df)
    early_departure_pct = (early_departures / total_flights) * 100

    return avg_delay_incl_neg, avg_delay_excl_neg, early_departure_pct


def analyze_grouped_counts(df):
    """
    Count flights by delay group.

    Parameters:
    - df (pandas.DataFrame): DataFrame with flight data.

    Returns:
    - pandas.Series: Counts of flights by delay group, sorted by group.
    """
    return df['DEP_DELAY_GROUP'].value_counts().sort_index()


def compare_short_long_haul(df, delay_columns):
    """
    Compare average delays for short-haul and long-haul flights.

    Parameters:
    - df (pandas.DataFrame): Flight dataset.
    - delay_columns (list): Columns representing delay factors.

    Returns:
    - tuple: (Short-haul average delays, Long-haul average delays).
    """
    short_haul = df[df['DISTANCE'] <= 500]
    long_haul = df[df['DISTANCE'] > 1500]
    short_avg = short_haul[delay_columns].mean()
    long_avg = long_haul[delay_columns].mean()
    return short_avg, long_avg


def delay_factor_pie_charts(df, delay_columns):
    """
    Create pie charts for overall and high-severity delay factors.

    Parameters:
    - df (pandas.DataFrame): Flight dataset.
    - delay_columns (list): Columns representing delay factors.

    Notes:
    - DEP_DELAY_GROUP >= 4 represents flights with significant delays of 61 minutes or more.
    """
    overall_delay_factors = df[delay_columns].sum()
    plt.figure(figsize=(8, 8))
    overall_delay_factors.plot(kind='pie', autopct='%1.1f%%', startangle=140, title='Overall Delay Factors')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    high_severity_delays = df[df['DEP_DELAY_GROUP'] >= 4][delay_columns].sum()
    plt.figure(figsize=(8, 8))
    high_severity_delays.plot(kind='pie', autopct='%1.1f%%', startangle=140, title='High-Severity Delay Factors (DEP_DELAY_GROUP >= 4)')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


def top_cities_by_delay(df, delay_columns):
    """
    Identify the top city for each delay reason.

    Parameters:
    - df (pandas.DataFrame): Flight dataset.
    - delay_columns (list): Columns representing delay reasons.

    Returns:
    - pandas.DataFrame: Transposed DataFrame with the top city and total delay for each reason.
    """
    top_cities = {}
    for delay_reason in delay_columns:
        delay_by_city = df.groupby('ORIGIN_CITY_NAME')[delay_reason].sum()
        max_city = delay_by_city.idxmax()
        max_delay = delay_by_city.max()
        top_cities[delay_reason] = {"City": max_city, "Total Delay (Minutes)": max_delay}
    return pd.DataFrame(top_cities).T


def analyze_weather_delay_in_dallas_by_quarter(df):
    """
    Summarize weather delays in Dallas by quarter.

    Parameters:
    - df (pandas.DataFrame): Flight dataset.

    Returns:
    - pandas.Series: Total weather delays per quarter for Dallas.
    """
    if 'QUARTER' not in df.columns:
        print("Error: 'QUARTER' column is missing in the dataset.")
        return None

    dallas_weather_delay = df[df['ORIGIN_CITY_NAME'] == "Dallas/Fort Worth, TX"]
    return dallas_weather_delay.groupby('QUARTER')['WEATHER_DELAY'].sum()


def calculate_airline_delay_metrics(df):
    """
    Analyze airline performance based on various delay metrics.

    Parameters:
    - df (pandas.DataFrame): Dataset containing flight delay data.

    Returns:
    - pandas.DataFrame: Aggregated metrics for each airline.
    """
    airline_performance = df.groupby('OP_UNIQUE_CARRIER').agg({
        'DEP_DEL15': 'mean',
        'CARRIER_DELAY': 'sum',
        'WEATHER_DELAY': 'sum',
        'NAS_DELAY': 'sum',
        'SECURITY_DELAY': 'sum',
        'LATE_AIRCRAFT_DELAY': 'sum',
        'DEP_DELAY_NEW': 'mean',
    }).reset_index()

    airline_performance['Total_Flights'] = df.groupby('OP_UNIQUE_CARRIER').size().values
    airline_performance['Delay_Ratio (%)'] = airline_performance['DEP_DEL15'] * 100
    airline_performance['Avg_Delay_Per_Flight (min)'] = airline_performance['DEP_DELAY_NEW']
    airline_performance['Total_Delay (min)'] = (
        airline_performance[['CARRIER_DELAY', 'WEATHER_DELAY',
                             'NAS_DELAY', 'SECURITY_DELAY',
                             'LATE_AIRCRAFT_DELAY']].sum(axis=1)
    )
    airline_performance['Delay_Contribution (%)'] = (
        airline_performance['Total_Delay (min)'] /
        airline_performance['Total_Delay (min)'].sum()
    ) * 100

    return airline_performance


def best_and_worst_airlines(airline_performance):
    """
    Identify the top 5 best and worst-performing airlines based on delay ratio.

    Parameters:
    - airline_performance (pandas.DataFrame): DataFrame with airline performance metrics.

    Returns:
    - tuple: (best_airlines, worst_airlines) DataFrames.
    """
    best_airlines = airline_performance.nsmallest(5, 'Delay_Ratio (%)')
    worst_airlines = airline_performance.nlargest(5, 'Delay_Ratio (%)')
    return best_airlines, worst_airlines


def airline_performance_by_factor(best_airlines, worst_airlines, delay_columns):
    """
    Extract delay factors for best and worst airlines.

    Parameters:
    - best_airlines (pandas.DataFrame): Best-performing airlines.
    - worst_airlines (pandas.DataFrame): Worst-performing airlines.
    - delay_columns (list): Delay factor columns.

    Returns:
    - tuple: DataFrames for delay factors of best and worst airlines.
    """
    best_factors = best_airlines[['OP_UNIQUE_CARRIER'] + delay_columns]
    worst_factors = worst_airlines[['OP_UNIQUE_CARRIER'] + delay_columns]
    return best_factors, worst_factors


def visualize_airline_performance_metrics(airline_performance):
    """
    Create visualizations for airline delay metrics:
    - Bar plot for delay ratio by airline.
    - Bar plot for average delay per flight.
    - Heatmap for correlation between delay types.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=airline_performance.sort_values('Delay_Ratio (%)', ascending=False),
                x='OP_UNIQUE_CARRIER', y='Delay_Ratio (%)')
    plt.title('Delay Ratio (%) by Airline')
    plt.xlabel('Airline')
    plt.ylabel('Delay Ratio (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=airline_performance.sort_values('Avg_Delay_Per_Flight (min)', ascending=False),
                x='OP_UNIQUE_CARRIER', y='Avg_Delay_Per_Flight (min)')
    plt.title('Average Delay Per Flight (Minutes) by Airline')
    plt.xlabel('Airline')
    plt.ylabel('Average Delay (Minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    delay_corr = airline_performance[['CARRIER_DELAY', 'WEATHER_DELAY',
                                      'NAS_DELAY', 'SECURITY_DELAY',
                                      'LATE_AIRCRAFT_DELAY']].corr()
    sns.heatmap(delay_corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Different Delay Types')
    plt.tight_layout()
    plt.show()


def train_and_evaluate_classifier(df, best_airlines, worst_airlines):
    """
    Train a Random Forest model to classify and predict flight delays.

    Parameters:
    - df (pandas.DataFrame): Dataset with flight delay features.
    - best_airlines (pandas.DataFrame): DataFrame of top-performing airlines.
    - worst_airlines (pandas.DataFrame): DataFrame of worst-performing airlines.

    Returns:
    - RandomForestClassifier: Trained Random Forest model.
    """
    delay_features = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    df['Delay_Class'] = df['DEP_DEL15'].apply(lambda x: 1 if x > 0.15 else 0)

    X = df[delay_features]
    y = df['Delay_Class']
    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    feature_importance = pd.DataFrame({
        'Feature': delay_features,
        'Importance': rfc.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance for Classification:")
    print(feature_importance)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Delay Classification')
    plt.tight_layout()
    plt.show()

    print("\nPredicting Delay Likelihood for Airlines:")
    for airline in best_airlines['OP_UNIQUE_CARRIER']:
        airline_data = df[df['OP_UNIQUE_CARRIER'] == airline][delay_features].fillna(0)
        if not airline_data.empty:
            predictions = rfc.predict_proba(airline_data)
            avg_likelihood = predictions[:, 1].mean()
            print(f"Airline {airline}: Average Delay Likelihood = {avg_likelihood:.2%}")

    for airline in worst_airlines['OP_UNIQUE_CARRIER']:
        airline_data = df[df['OP_UNIQUE_CARRIER'] == airline][delay_features].fillna(0)
        if not airline_data.empty:
            predictions = rfc.predict_proba(airline_data)
            avg_likelihood = predictions[:, 1].mean()
            print(f"Airline {airline}: Average Delay Likelihood = {avg_likelihood:.2%}")

    return rfc


def linear_regression_analysis(df, delay_columns, target_column='DEP_DELAY_NEW'):
    """
    Perform a linear regression analysis to understand the relationship
    between delay factors and the overall departure delay.

    Parameters:
    - df: DataFrame containing the dataset.
    - delay_columns: List of independent variables (delay factors).
    - target_column: Dependent variable (e.g., 'DEP_DELAY_NEW').

    Returns:
    - model: Trained Linear Regression model.
    """
    print("\n=== Linear Regression Analysis ===")

    regression_data = df[delay_columns + [target_column]].dropna()

    X = regression_data[delay_columns]
    y = regression_data[target_column]

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    y_pred = lr_model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared (Explained Variance): {r2:.2f}")

    coef_df = pd.DataFrame({
        'Feature': delay_columns,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', ascending=False)
    print("\nFeature Coefficients:")
    print(coef_df)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, color='blue', edgecolor='k')
    plt.title("Actual vs Predicted Departure Delays")
    plt.xlabel("Actual Delay (Minutes)")
    plt.ylabel("Predicted Delay (Minutes)")
    plt.tight_layout()
    plt.show()

    return lr_model


def main():
    directory = "/Users/macbookpro/Desktop/DS 2500/Project /Dataset"
    delay_columns = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

    print("\n=== Step 1: Loading and Cleaning Data ===")
    df = load_and_prepare_data(directory)
    df_cleaned = clean_data(df, delay_columns)

    print("\n=== Step 2: Overall Departure Delay Analysis ===")
    avg_delay_incl_neg, avg_delay_excl_neg, early_departure_pct = overall_delay_analysis(df)
    print(f"Average Delay (including negatives): {avg_delay_incl_neg:.2f} minutes")
    print(f"Average Delay (excluding negatives): {avg_delay_excl_neg:.2f} minutes")
    print(f"Percentage of early departures: {early_departure_pct:.2f}%")

    print("\n=== Step 3: Flight Counts by Delay Group ===")
    delay_group_counts = analyze_grouped_counts(df)
    print(delay_group_counts)

    print("\n=== Step 4: Delay Factor Pie Charts ===")
    delay_factor_pie_charts(df_cleaned, delay_columns)

    print("\n=== Step 5: Short vs Long Haul Delays ===")
    short_avg, long_avg = compare_short_long_haul(df_cleaned, delay_columns)
    print("Short-Haul Flights Average Delays:")
    print(short_avg)
    print("\nLong-Haul Flights Average Delays:")
    print(long_avg)

    print("\n=== Step 6: Visualizing Short vs Long Haul Comparison ===")
    comparison_df = pd.DataFrame({
        "Short-Haul": short_avg,
        "Long-Haul": long_avg
    }).T
    comparison_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Average Delay by Distance Category")
    plt.ylabel("Average Delay (Minutes)")
    plt.xlabel("Distance Category")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\n=== Step 7: Top City for Each Delay Reason ===")
    top_cities_df = top_cities_by_delay(df_cleaned, delay_columns)
    print(top_cities_df)

    print("\n=== Step 8: Weather Delay in Dallas/Fort Worth, TX by Quarter ===")
    weather_delay_by_quarter = analyze_weather_delay_in_dallas_by_quarter(df_cleaned)
    if weather_delay_by_quarter is not None:
        print(weather_delay_by_quarter)
        plt.figure(figsize=(8, 5))
        weather_delay_by_quarter.plot(kind='bar', color='steelblue')
        plt.title("Weather Delay in Dallas/Fort Worth, TX by Quarter")
        plt.xlabel("Quarter")
        plt.ylabel("Total Weather Delay (Minutes)")
        plt.tight_layout()
        plt.show()

    print("\n=== Step 9: Airline Performance Analysis ===")
    airline_performance = calculate_airline_delay_metrics(df)

    print("\nTop 5 Airlines with Lowest Delay Ratio:")
    best_airlines, worst_airlines = best_and_worst_airlines(airline_performance)
    print(best_airlines[['OP_UNIQUE_CARRIER', 'Delay_Ratio (%)', 'Avg_Delay_Per_Flight (min)']])
    print("\nTop 5 Airlines with Highest Delay Ratio:")
    print(worst_airlines[['OP_UNIQUE_CARRIER', 'Delay_Ratio (%)', 'Avg_Delay_Per_Flight (min)']])

    print("\n=== Step 10: Airline Performance by Delay Factor ===")
    best_factors, worst_factors = airline_performance_by_factor(best_airlines, worst_airlines, delay_columns)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
        print("\nPerformance of Top 5 Airlines by Delay Factor:")
        print(best_factors)
        print("\nPerformance of Worst 5 Airlines by Delay Factor:")
        print(worst_factors)

    print("\n=== Step 11: Visualizing Airline Performance ===")
    visualize_airline_performance_metrics(airline_performance)

    print("\n=== Step 12: Classification and Prediction ===")
    train_and_evaluate_classifier(df, best_airlines, worst_airlines)

    print("\n=== Step 13: Linear Regression Analysis ===")
    linear_regression_analysis(df_cleaned, delay_columns)


if __name__ == "__main__":
    main()
