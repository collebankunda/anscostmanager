import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime

st.title("The Armpass Project Cost Manager")

# Tabs: Expense Entry first, then the rest
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📝 Expense Entry", "📊 Cost Analysis", "📉 CPI Graph", "📈 SPI",
    "⚠️ Risk Impact", "📅 Trend Analysis", "📂 Reports", "🧩 Scenario Analysis"
])

# Sidebar for file uploads
st.sidebar.header("Upload Data Files")
budget_file = st.sidebar.file_uploader("Upload Budgeted Cost File (Excel)", type=["xlsx"])
actual_file = st.sidebar.file_uploader("Upload Actual Cost File (Excel)", type=["xlsx"])
schedule_file = st.sidebar.file_uploader("Upload Schedule Data File (Excel)", type=["xlsx"])
risk_file = st.sidebar.file_uploader("Upload Risk Analysis File (Excel)", type=["xlsx"])

def read_excel(file):
    return pd.read_excel(file) if file else None

budget_df = read_excel(budget_file)
actual_df_uploaded = read_excel(actual_file)
schedule_df = read_excel(schedule_file)
risk_df = read_excel(risk_file)

# SESSION STATE for expense entry
if "expenses_data" not in st.session_state:
    st.session_state["expenses_data"] = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])

# If actual file uploaded, use that; else use session state
if actual_df_uploaded is not None:
    actual_df = actual_df_uploaded
else:
    if not st.session_state["expenses_data"].empty:
        actual_df = st.session_state["expenses_data"]
    else:
        actual_df = None

# STANDARDIZE COLUMN NAMES
if budget_df is not None:
    budget_df.columns = budget_df.columns.str.strip()
    if "Category" in budget_df.columns:
        budget_df.rename(columns={"Category": "Cost Category"}, inplace=True)
    if "Amount" in budget_df.columns:
        budget_df.rename(columns={"Amount": "Amount_Budget"}, inplace=True)

if actual_df is not None:
    actual_df.columns = actual_df.columns.str.strip()
    if "Category" in actual_df.columns:
        actual_df.rename(columns={"Category": "Cost Category"}, inplace=True)
    if "Amount" in actual_df.columns:
        actual_df.rename(columns={"Amount": "Amount_Actual"}, inplace=True)

###################################
# TAB 1: Expense Entry
###################################
with tab1:
    st.subheader("Enter Actual Expenses")
    st.markdown("Record project expenses if no Actual Cost file is uploaded.")
    with st.form("expense_entry_form", clear_on_submit=True):
        expense_date = st.date_input("Date of Expense", datetime.date.today())
        description = st.text_input("Description")
        amount = st.number_input("Amount (UGX)", min_value=0.0, step=1000.0)
        category_options = [
            "Materials", "Labour", "Equipment", "Administrative & Overhead",
            "Maintenance & Warranty", "Permits & Legal Fees", "Quality Assurance & Testing",
            "Environmental Fees", "Utilities Relocation", "Contingency",
            "Insurance & Performance", "Transportation", "Bidding",
            "Site Preparation & Demobilisation", "Bid Guarantee Fees",
            "Subcontractor Cost (Meals)", "Other"
        ]
        category = st.selectbox("Cost Category", category_options)
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            if description.strip() == "":
                st.warning("Please enter a valid description.")
            else:
                # Create new row as DataFrame
                new_row_df = pd.DataFrame([{
                    "Date": expense_date,
                    "Description": description,
                    "Amount": amount,
                    "Category": category
                }])
                # Reset indexes on both existing data and new row to ensure unique indexing
                new_row_df.reset_index(drop=True, inplace=True)
                st.session_state["expenses_data"] = pd.concat(
                    [st.session_state["expenses_data"].reset_index(drop=True), new_row_df],
                    ignore_index=True
                )
                st.success("Expense added successfully!")
    st.markdown("### Current Entered Expenses")
    if not st.session_state["expenses_data"].empty:
        st.dataframe(st.session_state["expenses_data"], use_container_width=True)
    else:
        st.info("No expenses have been recorded yet.")

    # Download Options
    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine="openpyxl")
        df.to_excel(writer, index=False, sheet_name="Expenses")
        writer.close()
        return output.getvalue()

    if not st.session_state["expenses_data"].empty:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = to_csv(st.session_state["expenses_data"])
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="actual_expenses_entered.csv",
                mime="text/csv"
            )
        with col2:
            excel_data = to_excel(st.session_state["expenses_data"])
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="actual_expenses_entered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

###################################
# ONLY MERGE IF BOTH DF EXISTS
###################################
if budget_df is not None and actual_df is not None:
    if "Cost Category" in budget_df.columns and "Cost Category" in actual_df.columns:
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
            return row["Amount_Budget"] / row["Amount_Actual"]

        merged_df["CPI"] = merged_df.apply(safe_cpi, axis=1)
        total_budget = merged_df["Amount_Budget"].sum()
        total_actual = merged_df["Amount_Actual"].sum()
        total_variance = merged_df["Variance"].sum()
        avg_cpi = merged_df["CPI"].mean()

        ###################################
        # TAB 2: Cost Analysis
        ###################################
        with tab2:
            st.subheader("Detailed Cost Comparison")
            st.markdown("A side-by-side comparison of budgeted vs actual costs by category.")
            st.metric("Total Budget (UGX)", f"{total_budget:,.0f}")
            st.metric("Total Actual (UGX)", f"{total_actual:,.0f}")
            st.metric("Total Variance (UGX)", f"{total_variance:,.0f}")
            st.metric("Average CPI", f"{avg_cpi:.2f}")
            st.dataframe(merged_df, use_container_width=True)

        ###################################
        # TAB 3: CPI Graph
        ###################################
        with tab3:
            st.subheader("Cost Performance Index (CPI) Graph")
            if not merged_df.empty:
                fig = px.bar(
                    merged_df,
                    x="Cost Category",
                    y="CPI",
                    labels={"CPI": "CPI", "Cost Category": "Cost Category"},
                    title="Interactive CPI Graph"
                )
                fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Baseline CPI = 1")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available.")

        ###################################
        # TAB 4: SPI (Schedule)
        ###################################
        with tab4:
            st.subheader("Schedule Performance Index (SPI)")
            if schedule_df is not None and {"Task", "Planned Duration", "Actual Duration"}.issubset(schedule_df.columns):
                schedule_df["SPI"] = schedule_df["Planned Duration"] / schedule_df["Actual Duration"]
                fig, ax = plt.subplots()
                ax.bar(schedule_df["Task"], schedule_df["SPI"], color="orange")
                # Set explicit tick positions and labels to avoid warnings
                ax.set_xticks(range(len(schedule_df["Task"])))
                ax.set_xticklabels(schedule_df["Task"], rotation=45, ha="right")
                ax.axhline(1, color="red", linestyle="--", label="Baseline SPI = 1")
                ax.set_ylabel("SPI")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Please upload a valid schedule file.")

        ###################################
        # TAB 5: Risk Impact
        ###################################
        with tab5:
            st.subheader("Risk Impact Analysis")
            if risk_df is not None and {"Risk Factor", "Variance"}.issubset(risk_df.columns):
                fig, ax = plt.subplots()
                ax.bar(risk_df["Risk Factor"], risk_df["Variance"], color="green")
                ax.set_xticks(range(len(risk_df["Risk Factor"])))
                ax.set_xticklabels(risk_df["Risk Factor"], rotation=45, ha="right")
                ax.set_ylabel("Variance (UGX)")
                st.pyplot(fig)
            else:
                st.warning("Please upload a valid risk analysis file.")

        ###################################
        # TAB 6: Trend Analysis
        ###################################
        with tab6:
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

                date_range = st.date_input("Select Date Range", [min_date, max_date])
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_actual = actual_df[
                        (actual_df["Date"].dt.date >= start_date) & (actual_df["Date"].dt.date <= end_date)
                    ]
                else:
                    filtered_actual = actual_df

                filtered_actual["Quarter"] = filtered_actual["Date"].dt.to_period("Q")
                quarterly_trend = filtered_actual.groupby("Quarter")["Amount_Actual"].sum().reset_index()
                quarterly_trend["Quarter"] = quarterly_trend["Quarter"].astype(str)

                fig = px.line(
                    quarterly_trend,
                    x="Quarter",
                    y="Amount_Actual",
                    title="Quarterly Trend Analysis",
                    markers=True
                )
                ts = quarterly_trend["Amount_Actual"]
                try:
                    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
                    fit = model.fit()
                    forecast = fit.forecast(3)
                    forecast_quarters = [f"F{i+1}" for i in range(len(forecast))]
                    forecast_df = pd.DataFrame({"Quarter": forecast_quarters, "Forecast": forecast})
                    fig.add_scatter(
                        x=forecast_df["Quarter"],
                        y=forecast_df["Forecast"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(dash="dash")
                    )
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'Date' column found in actual data.")

        ###################################
        # TAB 7: Reports
        ###################################
        with tab7:
            st.subheader("Generate Reports")
            def generate_pdf_report():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Project Cost Analysis Report", ln=True, align="C")
                pdf.ln(10)
                for idx, row in merged_df.iterrows():
                    pdf.cell(
                        200,
                        10,
                        f"{row['Cost Category']}: UGX {row['Amount_Actual']:,} (Variance: UGX {row['Variance']:,})",
                        ln=True
                    )
                return pdf.output(dest="S").encode("latin1")

            st.download_button(
                label="📥 Download PDF Report",
                data=generate_pdf_report(),
                file_name="Cost_Analysis_Report.pdf",
                mime="application/pdf"
            )

            csv_data = merged_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_data,
                file_name="Merged_Data.csv",
                mime="text/csv"
            )

            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine="openpyxl")
                df.to_excel(writer, index=False, sheet_name="Sheet1")
                writer.close()
                return output.getvalue()

            excel_data = to_excel(merged_df)
            st.download_button(
                label="📥 Download Excel Data",
                data=excel_data,
                file_name="Merged_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        ###################################
        # TAB 8: Scenario Analysis
        ###################################
        with tab8:
            st.subheader("Scenario Analysis")
            st.markdown("Apply normalized weights to actual costs.")
            cost_categories = merged_df["Cost Category"].unique().tolist()
            weights = {}
            for cat in cost_categories:
                weights[cat] = st.number_input(f"Weight for {cat}", min_value=0.0, value=1.0, step=0.1)
            total_input_weight = sum(weights.values())
            if total_input_weight == 0:
                st.error("Sum of weights cannot be zero.")
            else:
                normalized_weights = {cat: w / total_input_weight for cat, w in weights.items()}
                merged_df["Normalized_Weighted_Actual"] = merged_df.apply(
                    lambda row: row["Amount_Actual"] * normalized_weights.get(row["Cost Category"], 0), axis=1
                )
                merged_df["Weighted_Variance"] = merged_df["Normalized_Weighted_Actual"] - merged_df["Amount_Budget"]

                def safe_weighted_cpi(row):
                    if row["Normalized_Weighted_Actual"] == 0:
                        return float("inf") if row["Amount_Budget"] > 0 else 1.0
                    return row["Amount_Budget"] / row["Normalized_Weighted_Actual"]

                merged_df["Weighted_CPI"] = merged_df.apply(safe_weighted_cpi, axis=1)
                st.dataframe(merged_df)

else:
    st.info("Upload a Budget file and provide Actual data (file or entered expenses) to see analysis.")

# FEEDBACK SECTION
st.markdown("## Feedback")
feedback = st.text_area("Please enter your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
