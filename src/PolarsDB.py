#    MetExtract II
#    Copyright (C) 2015
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from __future__ import absolute_import, division, print_function
import io
import logging
import os
import shutil
import zipfile
from json import dumps, loads
import polars as pl
import xlsxwriter

# Disable polars dtype warnings
# copied from https://github.com/ToucanToco/fastexcel/issues/326#issuecomment-2615748271


# This will disable all messages with a level <= WARNING
logging.getLogger("fastexcel.types.dtype").setLevel(logging.ERROR)

# This will completely disable the logger
logging.getLogger("fastexcel.types.dtype").disabled = True


class PolarsDB:
    """
    Helper class to manage multiple tables in various formats.

    Supports:
    - 'parquet': ZIP archive with Parquet files (one per table) - default
    - 'XLSX': Excel file with sheets (one per table)
    - 'TSV': ZIP archive with TSV files (one per table)
    - None: Auto-detect from file extension

    Usage examples:
        # Auto-detect format from file extension
        db = PolarsDB("data.meii")  # Uses parquet format
        db = PolarsDB("data.xlsx")       # Uses XLSX format
        db = PolarsDB("data.tsv.zip")    # Uses TSV format

        # Explicitly specify format
        db = PolarsDB("mydata.custom", format="parquet")
        db = PolarsDB("mydata.custom", format = "parquet")
        db = PolarsDB("mydata.custom", format="TSV")

        # Create and work with tables
        db.create_table("config", {"key": pl.Utf8, "value": pl.Utf8})
        db.insert_row("config", {"key": "version", "value": "1.0"})
        df = db.get_table("config")
        db.commit()  # Save changes
        db.close()   # Commit and cleanup
    """

    def __init__(self, filepath, format=None, load_all_tables=True):
        self.filepath = filepath
        self.tables = {}
        self.format = self._determine_format(format)
        if load_all_tables:
            self.load_all_tables()

    def _determine_format(self, format):
        """Determine the format from parameter or file extension."""
        if format is not None:
            if format.lower() not in ["parquet", "xlsx"]:  ## Warning: TODO: currently only parquet format is fully supported (i.e., use of datatypes)!
                raise ValueError(f"Unsupported format: {format}. Use 'parquet'")
            return format.lower()

        # Auto-detect from file extension
        lower_path = self.filepath.lower()
        if lower_path.endswith(".meii") or lower_path.endswith(".parquet.zip"):
            return "parquet"
        elif lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
            return "xlsx"
        # elif lower_path.endswith(".tsv.zip"):
        #     return "tsv"
        else:
            # Try to detect from zip contents if it's a zip file
            if os.path.exists(self.filepath) and zipfile.is_zipfile(self.filepath):
                try:
                    with zipfile.ZipFile(self.filepath, "r") as zf:
                        filenames = zf.namelist()
                        if any(f.endswith(".parquet") for f in filenames):
                            return "parquet"
                        elif any(f.endswith(".tsv") or f.endswith(".txt") for f in filenames):
                            return "tsv"
                except Exception:
                    pass
            # Default to parquet
            return "parquet"

    def load_all_tables(self):
        """Load all tables from the file if it exists."""
        if not os.path.exists(self.filepath):
            return

        try:
            if self.format == "parquet":
                self._load_parquet_tables()
            elif self.format == "xlsx":
                self._load_xlsx_tables()
            elif self.format == "tsv":
                self._load_tsv_tables()

        except Exception as e:
            print(f"Error: Could not load tables from {self.filepath}: {e}")
            raise ValueError(f"Could not load tables from {self.filepath}: {e}")

    def _load_parquet_tables(self):
        """Load tables from a ZIP archive containing Parquet files."""
        with zipfile.ZipFile(self.filepath, "r") as zf:
            for filename in zf.namelist():
                if filename.endswith(".parquet"):
                    table_name = filename[:-8]  # Remove '.parquet'
                    with zf.open(filename) as f:
                        parquet_data = f.read()
                        df = pl.read_parquet(io.BytesIO(parquet_data))
                        self.tables[table_name] = df

    def _load_xlsx_tables(self):
        """Load tables from an Excel file (each sheet is a table)."""
        # TODO: make sure to load correct dtypes
        import fastexcel as fe

        wb = fe.read_excel(self.filepath)
        dTypes_df = None
        if "__dTypes__" in wb.sheet_names:
            dTypes_df = pl.read_excel(self.filepath, sheet_name="__dTypes__")
        for sheet_name in wb.sheet_names:
            # Read the sheet into polars
            if sheet_name != "__dTypes__":
                # get schema for this table
                schema = None
                if dTypes_df is not None:
                    dTypes_row = dTypes_df.filter(pl.col("table_name") == sheet_name)
                    if dTypes_row.height > 0:
                        schema_str = dTypes_row[0, "schema"]
                        schema = self.__convert_json_to_polarsSchema(schema_str)  # Convert string back to dict
                try:
                    df = pl.read_excel(self.filepath, sheet_name=sheet_name, schema_overrides=schema)
                except (KeyError, Exception):
                    # schema_overrides may reference columns not present in this sheet
                    # (e.g. annotation step added columns not stored in __dTypes__); retry without it
                    df = pl.read_excel(self.filepath, sheet_name=sheet_name)
                self.tables[sheet_name] = df

    def _load_tsv_tables(self):
        """Load tables from a ZIP archive containing TSV files."""
        # TODO: make sure to load correct dtypes
        with zipfile.ZipFile(self.filepath, "r") as zf:
            for filename in zf.namelist():
                if filename.endswith(".tsv") or filename.endswith(".txt"):
                    if filename.lower() != "__dTypes__.tsv" and filename.lower() != "__dTypes__.txt":
                        # Remove extension to get table name
                        if filename.endswith(".tsv"):
                            table_name = filename[:-4]
                        else:
                            table_name = filename[:-4]

                        with zf.open(filename) as f:
                            tsv_data = f.read()
                            df = pl.read_csv(io.BytesIO(tsv_data), separator="\t")
                            self.tables[table_name] = df
                # TODO import dTypes if available

    def list_tables(self):
        return self.tables.keys()

    def create_table(self, table_name, schema):
        """Create a new empty table with given schema."""
        if self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' already exists.")
        self.tables[table_name] = pl.DataFrame(schema=schema)

    def insert_table(self, table_name, df):
        """Insert a whole DataFrame as a table."""
        if self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' already exists. Use insert_row() to add rows.")
        self.tables[table_name] = df

    def set_table(self, table_name, df):
        self.tables[table_name] = df

    def insert_row(self, table_name, row_dict):
        """Insert a single row into a table."""
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist. Create it first with create_table().")

        # Get the table's schema
        table_schema = self.tables[table_name].schema

        # Cast values to match the schema types
        casted_dict = {}
        for col_name, col_type in table_schema.items():
            if col_name in row_dict:
                value = row_dict[col_name]
                # Convert value to appropriate type
                if value is None:
                    casted_dict[col_name] = None
                elif col_type == pl.Int64:
                    casted_dict[col_name] = int(value) if value is not None else None
                elif col_type == pl.Float64:
                    casted_dict[col_name] = float(value) if value is not None else None
                elif col_type == pl.Utf8:
                    casted_dict[col_name] = str(value) if value is not None else None
                else:
                    casted_dict[col_name] = value
            else:
                # Column not provided, use None
                casted_dict[col_name] = None

        # Create DataFrame with explicit schema casting
        new_row = pl.DataFrame([casted_dict], schema=table_schema)
        self.tables[table_name] = pl.concat([self.tables[table_name], new_row], how="vertical")

    def get_table(self, table_name):
        """Get a table as a Polars DataFrame."""
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist.")
        return self.tables.get(table_name, pl.DataFrame())

    def has_table(self, table_name):
        """Check if a table exists."""
        return table_name in self.tables

    def remove_table(self, table_name):
        """Remove a table entirely so it is no longer written out on the next commit()/close()."""
        self.tables.pop(table_name, None)

    def delete_rows(self, table_name, condition):
        """Delete rows from a table based on a condition (Polars expression)."""
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist.")
        self.tables[table_name] = self.tables[table_name].filter(~condition)

    def update_rows(self, table_name, condition, updates):
        """Update rows in a table based on a condition."""
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist.")
        df = self.tables[table_name]
        for col, value in updates.items():
            df = df.with_columns(pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col))
        self.tables[table_name] = df

    def execute(self, query_desc, params=None):
        """
        Execute a query-like operation. This is for compatibility with the sqlite interface.
        query_desc should be a description or function that operates on the tables.
        """
        # This is a compatibility method - actual operations should use polars directly
        pass

    def cursor(self):
        """Return self for compatibility with SQLite interface."""
        return self

    def commit(self):
        """Save all tables to file in the appropriate format."""
        if not self.tables:
            return

        # copy self.filepath to a temporary file to avoid overwriting in case of errors
        bkp_filepath = self.filepath + ".bkp"
        if os.path.exists(bkp_filepath):
            os.remove(bkp_filepath)
            shutil.copyfile(self.filepath, bkp_filepath)

        # save tables
        if self.format == "parquet":
            self._save_parquet_tables()
        elif self.format == "xlsx":
            self._save_xlsx_tables()
        elif self.format == "tsv":
            self._save_tsv_tables()

        # remove backup
        if os.path.exists(bkp_filepath):
            os.remove(bkp_filepath)

    def _save_parquet_tables(self):
        """Save all tables as Parquet files in a ZIP archive."""
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            for table_name, df in self.tables.items():
                buffer = io.BytesIO()
                df.write_parquet(buffer, compression="snappy")
                buffer.seek(0)
                zf.writestr(f"{table_name}.parquet", buffer.getvalue())

    def _save_xlsx_tables(self):
        """Save all tables as sheets in an Excel file."""
        with xlsxwriter.Workbook(self.filepath) as wb:
            dTypes_table = {}
            for table_name, df in self.tables.items():
                if table_name != "__dTypes__":
                    # save dTypes of each row
                    dTypes_table[table_name] = df.schema
                    df = self._dedupe_columns_case_insensitive(df, table_name)
                    df.write_excel(workbook=wb, worksheet=table_name, autofit=True, float_precision=4)
            # save dTypes table as json object
            dTypes_df = pl.DataFrame([{"table_name": tname, "schema": self.__convert_polarsSchema_to_json(schema)} for tname, schema in dTypes_table.items()])
            dTypes_df.write_excel(workbook=wb, worksheet="__dTypes__")

    @staticmethod
    def _dedupe_columns_case_insensitive(df, table_name=""):
        """Rename columns so no two column names are equal case-insensitively.

        xlsxwriter's Excel table headers must be unique case-insensitively; a collision
        (e.g. "Collision_energy" and "COLLISION_ENERGY") makes it silently skip writing the
        whole table (only a "Duplicate header name" warning is emitted), leaving the sheet
        empty. Colliding columns are renamed here (keeping the first occurrence unchanged)."""
        seen_lower = set()
        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower not in seen_lower:
                seen_lower.add(lower)
                continue
            new_name = col
            suffix = 2
            while new_name.lower() in seen_lower:
                new_name = f"{col}_{suffix}"
                suffix += 1
            rename_map[col] = new_name
            seen_lower.add(new_name.lower())
        if rename_map:
            logging.warning(f"Sheet '{table_name}': renamed column(s) with case-insensitively duplicate names to avoid an empty Excel table: {rename_map}")
            df = df.rename(rename_map)
        return df

    def _save_tsv_tables(self):
        """Save all tables as TSV files in a ZIP archive."""
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            dTypes_table = {}
            for table_name, df in self.tables.items():
                if table_name != "__dTypes__":
                    # save dTypes of each row
                    dTypes_table[table_name] = df.schema
                    buffer = io.BytesIO()
                    # Write to buffer as TSV
                    df.write_csv(buffer, separator="\t", has_header=True, quote_char='"', escape_char="\\", null_value="", encoding="utf-8", line_terminator="\n", quote_style="necessary")
                    buffer.seek(0)
                    zf.writestr(f"{table_name}.tsv", buffer.getvalue())
            # save dTypes table
            dTypes_df = pl.DataFrame([{"table_name": tname, "schema": self.__convert_polarsSchema_to_json(schema)} for tname, schema in dTypes_table.items()])
            buffer = io.BytesIO()
            dTypes_df.write_csv(buffer, separator="\t", has_header=True, quote_char='"', escape_char="\\", null_value="", encoding="utf-8", line_terminator="\n", quote_style="necessary")
            buffer.seek(0)
            zf.writestr("__dTypes__.tsv", buffer.getvalue())

    def close(self):
        """Close the database (commit and cleanup)."""
        self.commit()
        self.tables = {}

    def __convert_polarsSchema_to_json(self, schema):
        """Convert Polars schema to JSON-serializable format."""
        json_schema = {}
        for col_name, col_type in schema.items():
            if col_type == pl.Int64:
                json_schema[col_name] = "Int64"
            elif col_type == pl.Float64:
                json_schema[col_name] = "Float64"
            elif col_type == pl.Utf8 or col_type is None or col_type == pl.Null:
                # do not include, default
                pass
            else:
                raise ValueError(f"Unsupported column type: {col_type} for column '{col_name}'")
        return dumps(json_schema)

    def __convert_json_to_polarsSchema(self, json_schema):
        """Convert JSON-serializable schema back to Polars schema."""
        schema = loads(json_schema)
        for col_name, col_type_str in schema.items():
            if col_type_str == "Int64":
                schema[col_name] = pl.Int64
            elif col_type_str == "Float64":
                schema[col_name] = pl.Float64
            elif col_type_str == "Utf8":
                schema[col_name] = pl.Utf8
            else:
                raise ValueError(f"Unsupported column type: {col_type_str}")
        return schema
