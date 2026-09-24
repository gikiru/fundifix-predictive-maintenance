"""
Build Great Expectations Data Docs (HTML report) for clean_wpdx.
Run from repo root: python src/data/build_ge_docs.py
Output: reports/gx/uncommitted/data_docs/local_site/index.html
"""
import duckdb
import great_expectations as gx

con = duckdb.connect("data/processed/fundifix.duckdb", read_only=True)
df = con.execute("SELECT * FROM clean_wpdx").df()
con.close()

context = gx.get_context(project_root_dir="reports")
ds = context.sources.add_or_update_pandas("wpdx_clean")
asset = ds.add_dataframe_asset("clean_wpdx")
batch_request = asset.build_batch_request(dataframe=df)
context.add_or_update_expectation_suite("fundifix_clean_suite")
v = context.get_validator(batch_request=batch_request,
                          expectation_suite_name="fundifix_clean_suite")

v.expect_table_row_count_to_be_between(min_value=15000, max_value=25000)
for col in ["target", "status_clean", "lat_deg", "lon_deg"]:
    v.expect_column_to_exist(col)
v.expect_column_values_to_be_in_set("target", ["Functional", "Needs Repair", "Non-Functional"])
v.expect_column_values_to_not_be_null("target")
v.expect_column_values_to_not_be_null("lat_deg", mostly=0.99)
v.expect_column_values_to_not_be_null("lon_deg", mostly=0.99)
v.expect_column_values_to_be_between("lat_deg", min_value=-5.0, max_value=5.0)
v.expect_column_values_to_be_between("lon_deg", min_value=33.5, max_value=42.5)
v.expect_column_values_to_be_between("install_year", min_value=1950, max_value=2026, mostly=0.95)
v.save_expectation_suite(discard_failed_expectations=False)

checkpoint = context.add_or_update_checkpoint(
    name="fundifix_checkpoint",
    validations=[{"batch_request": batch_request,
                  "expectation_suite_name": "fundifix_clean_suite"}],
)
result = checkpoint.run()
context.build_data_docs()
print("Success:", result.success)
