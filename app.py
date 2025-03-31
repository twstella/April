import streamlit as st
import pandas as pd
import plotly.express as px
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import os
from sklearn.cluster import KMeans
from collections import defaultdict
import random
import ast

# 환경설정
os.environ["PYTORCH_NO_PROXY"] = "*"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

st.title("Keyword Analysis & Clustering")

uploaded_file = st.file_uploader("[csv file upload]", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File upload finished")

    st.subheader("Step 1: Keyword embedding and Clustering")

    keyword_set = set()
    keyword_counts = defaultdict(int)
    keyword_category_count = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        try:
            q_keywords = ast.literal_eval(row["question_keywords"])
            a_keywords = ast.literal_eval(row["answer_keywords"])
        except:
            continue
        category = str(row["question_category"]).strip().lower()
        keywords = q_keywords + a_keywords
        for kw in keywords:
            keyword_set.add(kw)
            keyword_counts[kw] += 1
            keyword_category_count[kw][category] += 1

    keyword_list = list(keyword_set)
    keyword_list = sorted(keyword_list)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(keyword_list, show_progress_bar=True)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True)
    labels = clusterer.fit_predict(scaled_embeddings)

    reducer = umap.UMAP(random_state=42, metric='cosine')
    reduced = reducer.fit_transform(scaled_embeddings)

    kmeans = KMeans(n_clusters=8, random_state=42)
    k_labels = kmeans.fit_predict(scaled_embeddings)

    cluster_df = pd.DataFrame({
        "keyword": keyword_list,
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "cluster": labels
    })
    st.plotly_chart(px.scatter(cluster_df, x="x", y="y",
                               color=cluster_df["cluster"].astype(str),
                               hover_name="keyword",
                               title="HDBSCAN clustering"))
    kmean_cluster = pd.DataFrame({
        "keyword": keyword_list,
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "cluster": k_labels
    })
    st.plotly_chart(px.scatter(kmean_cluster, x="x", y="y",
                               color=kmean_cluster["cluster"].astype(str),
                               hover_name="keyword",
                               title="kmeans clustering"))

    st.subheader("Step 2: Visualize category")
    df["category"] = df["question_category"]
    category_count = df["category"].value_counts().reset_index()
    category_count.columns = ["category", "count"]
    fig = px.bar(category_count, x="category", y="count", color="category",
                 title="categories", labels={"count": "quantity of questions"})
    st.plotly_chart(fig)

    st.subheader("Step 3: Categories per Keywords")
    all_categories = sorted(list({cat for counts in keyword_category_count.values() for cat in counts}))

    rows = []
    for kw in keyword_list:
        row = {"keyword": kw, "total": keyword_counts[kw]}
        for cat in all_categories:
            row[cat.capitalize()] = keyword_category_count[kw].get(cat, 0)
        rows.append(row)

    keyword_cat_df = pd.DataFrame(rows)
    keyword_cat_df = keyword_cat_df.sort_values(by="total", ascending=False)
    st.dataframe(keyword_cat_df, use_container_width=True)
    st.subheader("Step 4: Dialog dataset Generation")
    input_keyword = st.text_input("Enter a keyword for dialog generation")

    if input_keyword:
        st.markdown("### In-depth Dialog Conversation")

        selected_rows = []
        categories_order = ["definition", "fact", "value", "compare", "policy", "conclusion"]

        for cat in categories_order:
            filtered = df[
                (df["question_category"].str.lower() == cat) &
                (df["question_keywords"].apply(lambda x: input_keyword in str(x)))
            ]
            if len(filtered) > 0:
                selected_rows.append(filtered.sample(1))

        if selected_rows:
            result_df = pd.concat(selected_rows)
            for _, row in result_df.iterrows():
                question = str(row["question"])
                answer = str(row["answer"])
                category = str(row["question_category"])
                with st.chat_message("user", avatar="🤖"):
                    st.markdown(f"**[{category}]** Q: {question}")
                with st.chat_message("assistant"):
                    st.markdown(f"**A:** {answer}")
            with st.chat_message("user", avatar="🤖"):
                st.markdown(f"**Q:** <Make a follow-up question>")
        else:
            st.warning("Failed to generate a dialog with that keyword. Please enter another keyword.")
        st.markdown("### In-Breadth Dialog Conversation")
        breadth_chain = []
        used_keywords = set()
        current_keyword = input_keyword.strip()
        used_keywords.add(current_keyword)

        category_sequence = ["definition", "fact", "value", "compare", "policy", "conclusion"]

        def safe_eval(x):
            try:
                if pd.isna(x) or str(x).strip() == "":
                    return []
                return ast.literal_eval(x)
            except Exception:
                return []

        for i, cat in enumerate(category_sequence):
            filtered = df[
                (df["question_category"].str.lower() == cat) &
                (df["question_keywords"].apply(lambda x: current_keyword in safe_eval(x))) &
                (df["question_keywords"].apply(lambda x: len(safe_eval(x)) > 0)) &
                (df["answer_keywords"].apply(lambda x: len(safe_eval(x)) > 0))
            ]

            if len(filtered) > 0:
                selected_row = filtered.sample(1).iloc[0]
                question = selected_row["question"]
                answer = selected_row["answer"]
                a_keywords = safe_eval(selected_row["answer_keywords"])
                print(a_keywords)
                # 다음 카테고리 내 등장 키워드 집합 (현재 단계가 마지막이 아니면)
                if i + 1 < len(category_sequence):
                    next_cat = category_sequence[i + 1]
                    next_q_keywords = set()
                    next_df = df[df["question_category"].str.lower() == next_cat]
                    for kw_list in next_df["question_keywords"].dropna():
                        next_q_keywords.update(safe_eval(kw_list))

                    # a_keywords 중에서 다음 질문과 겹치는 키워드만 남김
                    valid_next_keywords = [kw for kw in a_keywords if kw not in used_keywords and kw in next_q_keywords]
                else:
                    valid_next_keywords = [kw for kw in a_keywords if kw not in used_keywords]

                breadth_chain.append((cat.capitalize(), current_keyword, question, answer))

                if valid_next_keywords:
                    current_keyword = random.choice(valid_next_keywords)
                    used_keywords.add(current_keyword)
                else:
                    break
            else:
                break

        if breadth_chain:
            for i, (category, keyword, question, answer) in enumerate(breadth_chain):
                with st.chat_message("user", avatar="🤖"):
                    st.markdown(f"**[{category}] Q{i+1} (keyword: {keyword})**\n{question}")
                with st.chat_message("assistant"):
                    st.markdown(f"**A:** {answer}")
            with st.chat_message("user", avatar="🤖"):
                st.markdown("**Q:** <Make a follow-up question>")
        else:
            st.warning("No suitable QA chain could be generated with that keyword.")