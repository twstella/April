import pandas as pd
import glob

target_files = glob.glob("sampled/*_keywords.csv")

for file in target_files:
    df = pd.read_csv(file)

    def normalize_category(cat):
        if pd.isna(cat):  # NaN 처리
            return ""
        tmp = str(cat).lower().replace("\"", "")
        if tmp.startswith("compar"):
            return "Compare"
        elif tmp.startswith("defin"):
            return "Definition"
        elif "fact" in tmp:
            return "Fact"
        elif "policy" in tmp:
            return "Policy"
        elif "value" in tmp:
            return "Value"
        elif "definition" in tmp:
            return "Definition"
        elif "conclusion" in tmp:
            return "Conclusion"
        else:
            return tmp.capitalize()

    df["question_category"] = df["question_category"].apply(normalize_category)
    df.to_csv(file, index=False)
    print(f"✅ {file} 저장 완료.")
