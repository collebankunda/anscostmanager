import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt  # Required for Matplotlib plots
from fpdf import FPDF
from io import BytesIO
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime

st.title("Uganda Road Project Management Dashboard")

# Create navigation tabs (7 tabs in total)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Cost Analysis", "📉 CPI Graph", "📈 SPI",
    "⚠️ Risk Impact", "📅 Trend Analysis", "📂 Reports", "🧩 Scenario Analysis"
])

# Sidebar for file uploads
st.sidebar.header("Upload Data Files")
budget_file = st.sidebar.file_uploader("Upload Budgeted Cost File (Excel)", type=["xlsx"])
actual_file = st.sidebar.file_uploader("Upload Actual Cost File (Excel)", type=["xlsx"])
schedule_file = st.sidebar.file_uploader("Upload Schedule Data File (Excel)", type=["xlsx"])
risk_file = st.sidebar.file_uploader("Upload Risk Analysis File (Excel)", type=["xlsx"])

# Function to read Excel files
def read_excel(file):
    return pd.read_excel(file) if file else None

# Read data files
budget_df = read_excel(budget_file)
actual_df = read_excel(actual_file)
schedule_df = read_excel(schedule_file)
risk_df = read_excel(risk_file)

# Enhanced multiselect function with "Select All" support
def multiselect_with_select_all(label, options, default=None):
    full_options = ["Select All"] + options
    if default is None:
        default = ["Select All"]
    if "Select All" in default:
        default = full_options
    selected_options = st.sidebar.multiselect(label, full_options, default=default)
    if "Select All" in selected_options:
        return options
    return selected_options

# Proceed only if Budget and Actual cost files are uploaded
if budget_df is not None and actual_df is not None:
    # Standardize column names
    budget_df.columns = budget_df.columns.str.strip()
    actual_df.columns = actual_df.columns.str.strip()

    if "Cost Category" in budget_df.columns and "Cost Category" in actual_df.columns:
        # Merge budget and actual data with an outer merge
        merged_df = pd.merge(
            budget_df,
            actual_df,
            on="Cost Category",
            suffixes=("_Budget", "_Actual"),
            how="outer"
        )
        merged_df["Amount_Budget"] = merged_df["Amount_Budget"].fillna(0)
        merged_df["Amount_Actual"] = merged_df["Amount_Actual"].fillna(0)
        merged_df["Variance"] = merged_df["Amount_Actual"] - merged_df["Amount_Budget"]

        def safe_cpi(row):
            if row["Amount_Actual"] == 0:
                return float("inf") if row["Amount_Budget"] > 0 else 1.0
            else:
                return row["Amount_Budget"] / row["Amount_Actual"]

        merged_df["CPI"] = merged_df.apply(safe_cpi, axis=1)

        # KPI Summary (Baseline)
        total_budget = merged_df["Amount_Budget"].sum()
        total_actual = merged_df["Amount_Actual"].sum()
        total_variance = merged_df["Variance"].sum()
        avg_cpi = merged_df["CPI"].mean()

        st.markdown("## Project Summary (Baseline)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Budget (UGX)", f"{total_budget:,.0f}")
        col2.metric("Total Actual (UGX)", f"{total_actual:,.0f}")
        col3.metric("Total Variance (UGX)", f"{total_variance:,.0f}")
        col4.metric("Average CPI", f"{avg_cpi:.2f}")

        # Sidebar Filters
        st.sidebar.header("Interactive Filters")
        min_budget = float(merged_df["Amount_Budget"].min())
        max_budget = float(merged_df["Amount_Budget"].max())
        budget_threshold = st.sidebar.slider(
            "Select Budget Threshold (UGX)",
            min_value=int(min_budget),
            max_value=int(max_budget),
            value=int(min_budget)
        )
        cost_categories = merged_df["Cost Category"].unique().tolist()
        selected_categories = multiselect_with_select_all("Select Cost Categories", options=cost_categories)
        show_variance = st.sidebar.checkbox("Show Variance Analysis", value=True)
        filtered_df = merged_df[
            (merged_df["Amount_Budget"] >= budget_threshold) &
            (merged_df["Cost Category"].isin(selected_categories))
        ]

        # TAB 1: Detailed Cost Comparison
        with tab1:
            st.subheader("Detailed Cost Comparison")
            st.dataframe(filtered_df)

        # TAB 2: Interactive CPI Graph using Plotly
        with tab2:
            st.subheader("Cost Performance Index (CPI) Graph")
            if not filtered_df.empty:
                fig = px.bar(filtered_df, x="Cost Category", y="CPI",
                             labels={"CPI": "CPI", "Cost Category": "Cost Category"},
                             title="Interactive CPI Graph")
                fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Baseline CPI = 1")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")

        # TAB 3: Schedule Performance Index (SPI) using Matplotlib
        with tab3:
            st.subheader("Schedule Performance Index (SPI)")
            if schedule_df is not None and {"Task", "Planned Duration", "Actual Duration"}.issubset(schedule_df.columns):
                schedule_df["SPI"] = schedule_df["Planned Duration"] / schedule_df["Actual Duration"]
                fig, ax = plt.subplots()
                ax.bar(schedule_df["Task"], schedule_df["SPI"], color="orange")
                ax.axhline(1, color="red", linestyle="--", label="Baseline SPI = 1")
                ax.set_ylabel("SPI")
                ax.set_xticklabels(schedule_df["Task"], rotation=45, ha="right")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Schedule data file must contain 'Task', 'Planned Duration', and 'Actual Duration' columns.")

        # TAB 4: Risk Impact Analysis using Matplotlib
        with tab4:
            st.subheader("Risk Impact Analysis")
            if risk_df is not None and {"Risk Factor", "Variance"}.issubset(risk_df.columns):
                fig, ax = plt.subplots()
                ax.bar(risk_df["Risk Factor"], risk_df["Variance"], color="green")
                ax.set_ylabel("Variance (UGX)")
                ax.set_xticklabels(risk_df["Risk Factor"], rotation=45, ha="right")
                st.pyplot(fig)
            else:
                st.warning("Risk analysis file must contain 'Risk Factor' and 'Variance' columns.")

        # TAB 5: Interactive Quarterly Trend Analysis & Forecast using Plotly with Date Range Filter
        with tab5:
            st.subheader("Quarterly Trend Analysis & Forecast")
            if "Date" in actual_df.columns:
                actual_df["Date"] = pd.to_datetime(actual_df["Date"], errors="coerce")
                min_date_raw = actual_df["Date"].min()
                max_date_raw = actual_df["Date"].max()
                if pd.isna(min_date_raw):
                    min_date = datetime.date.today()
                else:
                    min_date = min_date_raw.date()
                if pd.isna(max_date_raw):
                    max_date = datetime.date.today()
                else:
                    max_date = max_date_raw.date()
                
                date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_actual = actual_df[(actual_df["Date"].dt.date >= start_date) & (actual_df["Date"].dt.date <= end_date)]
                else:
                    filtered_actual = actual_df

                filtered_actual["Quarter"] = filtered_actual["Date"].dt.to_period("Q")
                quarterly_trend = filtered_actual.groupby("Quarter")["Amount_Actual"].sum().reset_index()
                quarterly_trend["Quarter"] = quarterly_trend["Quarter"].astype(str)
                
                fig = px.line(quarterly_trend, x="Quarter", y="Amount_Actual",
                              title="Quarterly Trend Analysis", markers=True)
                ts = quarterly_trend["Amount_Actual"]
                try:
                    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
                    fit = model.fit()
                    forecast = fit.forecast(3)
                    forecast_quarters = [f"F{i+1}" for i in range(len(forecast))]
                    forecast_df = pd.DataFrame({"Quarter": forecast_quarters, "Forecast": forecast})
                    fig.add_scatter(x=forecast_df["Quarter"], y=forecast_df["Forecast"],
                                    mode="lines+markers", name="Forecast", line=dict(dash="dash"))
                except Exception as e:
                    st.error("Forecasting error: " + str(e))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Actual cost data must contain a 'Date' column for trend analysis.")

        # TAB 6: Generate Reports - Enhanced with Excel Export Option
        with tab6:
            st.subheader("Generate Reports")
            def generate_pdf_report():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Uganda Road Project Cost Analysis Report", ln=True, align="C")
                pdf.ln(10)
                for index, row in filtered_df.iterrows():
                    pdf.cell(
                        200,
                        10,
                        f"{row['Cost Category']}: UGX {row['Amount_Actual']:,} (Variance: UGX {row['Variance']:,})",
                        ln=True
                    )
                return pdf.output(dest="S").encode("latin1")
            st.download_button(label="📥 Download PDF Report",
                               data=generate_pdf_report(),
                               file_name="Cost_Analysis_Report.pdf",
                               mime="application/pdf")
            
            csv_data = merged_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Download CSV Data",
                               data=csv_data,
                               file_name="Merged_Data.csv",
                               mime="text/csv")
            
            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine="openpyxl")
                df.to_excel(writer, index=False, sheet_name="Sheet1")
                writer.close()  # Use close() to finish writing
                processed_data = output.getvalue()
                return processed_data
            
            excel_data = to_excel(merged_df)
            st.download_button(label="📥 Download Excel Data",
                               data=excel_data,
                               file_name="Merged_Data.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # TAB 7: Scenario Analysis with Normalized Weights Applied Only to Actual Costs
        with tab7:
            st.subheader("Scenario Analysis: Weighted Actual Costs with Normalized Weights")
            st.markdown(
                """
                Assign weights to each cost category. These weights are normalized so that their total is 1.0 (100%), 
                reflecting the relative importance of each cost component in terms of execution, risk, and financial burden.
                """
            )
            # Input weights for each cost category
            weights = {}
            for cat in cost_categories:
                weights[cat] = st.number_input(f"Weight for {cat}", min_value=0.0, value=1.0, step=0.1)
            
            # Normalize the weights so they sum to 1.0
            total_input_weight = sum(weights.values())
            if total_input_weight == 0:
                st.error("The sum of weights cannot be zero. Please assign non-zero weights.")
            else:
                normalized_weights = {cat: weight / total_input_weight for cat, weight in weights.items()}
                st.markdown("### Normalized Weights")
                st.write(normalized_weights)
                
                # Compute weighted actual cost using normalized weights (Budget remains unchanged)
                merged_df["Normalized_Weighted_Actual"] = merged_df.apply(
                    lambda row: row["Amount_Actual"] * normalized_weights.get(row["Cost Category"], 0), axis=1
                )
                merged_df["Weighted_Variance"] = merged_df["Normalized_Weighted_Actual"] - merged_df["Amount_Budget"]
                
                def safe_weighted_cpi(row):
                    if row["Normalized_Weighted_Actual"] == 0:
                        return float("inf") if row["Amount_Budget"] > 0 else 1.0
                    else:
                        return row["Amount_Budget"] / row["Normalized_Weighted_Actual"]
                
                merged_df["Weighted_CPI"] = merged_df.apply(safe_weighted_cpi, axis=1)
                
                # Weighted KPI Summary
                total_weighted_actual = merged_df["Normalized_Weighted_Actual"].sum()
                total_weighted_variance = merged_df["Weighted_Variance"].sum()
                avg_weighted_cpi = merged_df["Weighted_CPI"].mean()
                
                st.markdown("### Weighted Project Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Budget (UGX)", f"{merged_df['Amount_Budget'].sum():,.0f}")
                col2.metric("Total Weighted Actual (UGX)", f"{total_weighted_actual:,.0f}")
                col3.metric("Total Weighted Variance (UGX)", f"{total_weighted_variance:,.0f}")
                col4.metric("Weighted CPI", f"{avg_weighted_cpi:.2f}")
                
                st.markdown("### Detailed Weighted Data")
                st.dataframe(merged_df[[
                    "Cost Category", "Amount_Budget", "Amount_Actual",
                    "Normalized_Weighted_Actual", "Weighted_Variance", "Weighted_CPI"
                ]])
    else:
        st.warning("The required column 'Cost Category' is missing in the uploaded files.")
else:
    st.warning("Please upload both Budgeted and Actual Cost files to proceed.")

# Feedback Section
st.markdown("## Feedback")
feedback = st.text_area("Please enter your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
