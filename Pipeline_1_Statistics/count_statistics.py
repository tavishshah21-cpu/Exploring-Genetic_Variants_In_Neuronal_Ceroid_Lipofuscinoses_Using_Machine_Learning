import pandas as pd

# CHANGE PATH
FILE = r"D:\tavish_shah\PES\capstone\Pipeline_1_Statistics\ML_READY.csv"

print("Loading...")

df = pd.read_csv(
    FILE,
    low_memory=False
)

print("\n====================")
print("DATASET SUMMARY")
print("====================")

print("Rows:", len(df))
print("Columns:", len(df.columns))


summary = []

for col in df.columns:

    try:

        unique_count = df[col].nunique(
            dropna=False
        )

        missing = df[col].isna().sum()

        sample = (
            df[col]
            .dropna()
            .astype(str)
            .unique()[:10]
        )

        summary.append({

            "Column":

                col,

            "Unique Values":

                unique_count,

            "Missing":

                missing,

            "Sample Values":

                ", ".join(
                    sample
                )
        })

    except:

        summary.append({

            "Column":

                col,

            "Unique Values":

                "ERROR",

            "Missing":

                "ERROR",

            "Sample Values":

                ""
        })


summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(

    "dataset_column_summary.csv",

    index=False
)

print("\nSaved:")
print("dataset_column_summary.csv")

print("\nPreview:\n")

print(
    summary_df.head(20)
)