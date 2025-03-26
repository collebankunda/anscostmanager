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

# SESSION STATE for expense entry; initialize if not present
if "expenses_data" not in st.session_state:
    # Initialize with four columns: Date, Description, Amount, Category
    st.session_state["expenses_data"] = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])

# If an actual file is uploaded, use that; otherwise, use the session state data
if actual_df_uploaded is not None:
    actual_df = actual_df_uploaded
else:
    if not st.session_state["expenses_data"].empty:
        actual_df = st.session_state["expenses_data"]
    else:
        actual_df = None

# STANDARDIZE COLUMN NAMES for budget and actual files (for later merging)
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
                # Retrieve current expenses data from session state
                current_expenses = st.session_state["expenses_data"]
                # Ensure the index is unique by resetting if needed
                if not current_expenses.index.is_unique:
                    current_expenses = current_expenses.reset_index(drop=True)
                # Create the new expense row and reset its index
                new_row_df = pd.DataFrame([{
                    "Date": expense_date,
                    "Description": description,
                    "Amount": amount,
                    "Category": category
                }]).reset_index(drop=True)
                # Concatenate with ignore_index=True to assign a new unique index
                updated_expenses = pd.concat(
                    [current_expenses, new_row_df],
                    ignore_index=True
                )
                st.session_state["expenses_data"] = updated_expenses
                st.success("Expense added successfully!")
                
    st.markdown("### Current Entered Expenses")
    if not st.session_state["expenses_data"].empty:
        st.dataframe(st.session_state["expenses_data"], use_container_width=True)
    else:
        st.info("No expenses have been recorded yet.")

    # Function to prepare the export DataFrame with exactly four columns
    def get_expenses_export_df():
        df = st.session_state["expenses_data"]
        # Rename columns: "Amount" becomes "Amount_Actual" and "Category" becomes "Cost Category"
        df_export = df.rename(columns={"Amount": "Amount_Actual", "Category": "Cost Category"})
        # Select only the required columns in the specified order
        df_export = df_export[["Date", "Description", "Amount_Actual", "Cost Category"]]
        return df_export

    # Download helper functions
    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine="openpyxl")
        df.to_excel(writer, index=False, sheet_name="Expenses")
        writer.close()
        return output.getvalue()

    export_df = get_expenses_export_df()

    if not export_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = to_csv(export_df)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="actual_expenses_entered.csv",
                mime="text/csv"
            )
        with col2:
            excel_data = to_excel(export_df)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="actual_expenses_entered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

###################################
# (Other tabs: Cost Analysis, CPI Graph, SPI, Risk Impact, Trend Analysis, Reports, and Scenario Analysis)
###################################
# For brevity, the rest of your code remains unchanged.
# Ensure that the merge, graphing, and reporting sections work as expected.

# FEEDBACK SECTION
st.markdown("## Feedback")
feedback = st.text_area("Please enter your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
