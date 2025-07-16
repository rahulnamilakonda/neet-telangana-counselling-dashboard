# %%
# Deep Insights for NEET Telangana 2024 vs 2025 Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

# Title
st.title("NEET Telangana 2024 vs 2025 Dashboard for Counselling Support")


# Load data
@st.cache_data
def load_data():
    neet_2024 = pd.read_csv("neet_tg_data_2024.csv")
    cols_2024_rename_map = {
        col: col.lower().replace(".", "").replace("\n", "_")
        for col in neet_2024.columns.values.tolist()
    }
    neet_2024.rename(columns=cols_2024_rename_map, inplace=True)

    neet_2025 = pd.read_csv(
        "neet_tg_data.csv", names=cols_2024_rename_map.values(), header=0
    )
    neet_2024["year"] = 2024
    neet_2025["year"] = 2025

    return neet_2024, neet_2025


neet_2024, neet_2025 = load_data()
neet_combined = pd.concat([neet_2024, neet_2025], ignore_index=True)
neet_combined["score"] = pd.to_numeric(neet_combined["score"], errors="coerce")
neet_combined.dropna(subset=["score"], inplace=True)
neet_combined["category"] = neet_combined["category"].str.replace(
    "OBC-(NCL) As Per Central List", "OBC- NCL (Central List)"
)

# Sidebar Score Check
st.sidebar.header("🔎 Check Your Score Insight")
user_score = st.sidebar.number_input(
    "Enter Your NEET Score", min_value=0, max_value=720, step=1
)
if user_score > 0:
    percentile_2025 = np.round((neet_2025["score"] < user_score).mean() * 100, 2)
    st.sidebar.write(f"Your approximate percentile in 2025: **{percentile_2025:.2f}%**")
    st.sidebar.info(
        "This percentile shows where you stand compared to other 2025 candidates."
    )
    st.sidebar.markdown(
        f"""
    ### Score Insight for {user_score}
    - Your entered score is **{user_score}**.
    - This puts you roughly in the **top {round(100 - percentile_2025)}%** of students in 2025.

    **What this means for you:**
    - 🟢 If you're scoring **above 450**: You're in a strong position, especially for MBBS in state quota.
    - 🟡 If you're in **300–450**: Competition is tighter. Focus on your **category advantage** and smart college choices.
    - 🔴 If below **300**: Focus on **category-specific colleges**, BDS or AYUSH, and backup options like state counseling rounds.
    """
    )

# Score distribution
st.subheader("Score Distribution NEET 2024 vs 2025")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.histplot(
    data=neet_combined, x="score", hue="year", bins=50, kde=True, palette="Set2", ax=ax1
)
ax1.set_title("Score Distribution NEET 2024 vs 2025")
ax1.grid(True)
st.pyplot(fig1)
st.markdown(
    """
#### What This Graph Shows:
- The curve shows how many students scored at each range in 2024 vs 2025.
- 2025 has **fewer students scoring above 400**, meaning **cutoffs may reduce**.
- The majority of students are scoring between **200–400**, so that's where competition is highest.

**Tip:** If your score is near 400, a few marks can mean a big rank jump.
"""
)

# Category-wise Box Plot
st.subheader("Category-wise Score Distribution")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=neet_combined, x="category", y="score", hue="year", palette="Set2", ax=ax2
)
ax2.set_title("Category-wise Score Distribution (2024 vs 2025)")
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True)
st.pyplot(fig2)
st.markdown(
    """
#### What This Graph Shows:
- Each box shows the score range for that category.
- The **middle line** is the **median score** – half the students scored below it.
- In 2025, medians dropped for all categories – especially General and EWS.

**Counseling Tip:** If your score is **above your category's median**, you're doing better than average.
"""
)

# Category-wise Stats
agg_stats = neet_combined.groupby(["year", "category"])["score"].agg(
    ["mean", "median", "std", "min", "max"]
)
st.subheader("Category-wise Stats")
st.dataframe(agg_stats)
st.markdown(
    """
#### 📌 Interpretation:
- All categories saw a drop in average and median scores.
- **Standard deviation** is lower, meaning most students scored close together.

**Implication:** The score difference between two students could be small, but their **ranks may differ a lot**. Every mark matters!
"""
)

# Score >= 400 Category Distribution
st.subheader("400+ Score Category-wise Count")
mark_cutoff = 400
high_scorers = neet_combined[neet_combined["score"] >= mark_cutoff]
high_cat_dist = pd.crosstab(high_scorers["category"], high_scorers["year"])
fig3, ax3 = plt.subplots(figsize=(10, 6))
high_cat_dist.plot(kind="bar", colormap="Set2", ax=ax3)
ax3.set_title("Category Distribution of Students with Score >= 400")
ax3.set_ylabel("Student Count")
ax3.grid(True)
st.pyplot(fig3)
st.markdown(
    """
#### 📌 Why This Matters:
- Shows how many students in each category scored **above 400**.
- Numbers dropped sharply in 2025 — especially for **SC/ST/EWS**.

**🎯 If you scored 400+ this year:** You’re among a **smaller elite group**. Your chances for top colleges are **better than 2024**.
"""
)

# Heatmap - Category vs Score Bin (2025)
st.subheader("Score Bin Distribution for 2025")
neet_2025["score_bin"] = pd.cut(
    neet_2025["score"],
    bins=[0, 100, 200, 300, 400, 500, 600, 700, 720],
    labels=[
        "0-100",
        "100-200",
        "200-300",
        "300-400",
        "400-500",
        "500-600",
        "600-700",
        "700-720",
    ],
)
score_cat_pivot = pd.pivot_table(
    neet_2025,
    index="category",
    columns="score_bin",
    values="name",
    aggfunc="count",
    fill_value=0,
)
fig4, ax4 = plt.subplots(figsize=(15, 6))
sns.heatmap(score_cat_pivot, annot=True, fmt="d", cmap="Set2", ax=ax4)
ax4.set_title("Category Distribution - 2025 Across Score Bins")
st.pyplot(fig4)
st.markdown(
    """
#### Insight from Score Buckets:
- Most students (80%+) scored below 400.
- **OBC-NCL and General** had the most students in 400–600+ range.
- **No one crossed 700** in 2025, unlike earlier years.

**Targeted Strategy:** Focus on **category cutoffs**, and don't panic if your score seems low — most students are in the same range.
"""
)

# Top 1% Gender Insight
st.subheader("Gender Composition in Top 1% Scorers")
top_1_percent_cutoff = np.percentile(neet_2025["score"], 99)
top_1_df = neet_2025[neet_2025["score"] >= top_1_percent_cutoff]
gender_cat_top1 = pd.crosstab(top_1_df["category"], top_1_df["gender"])
fig5, ax5 = plt.subplots(figsize=(10, 6))
gender_cat_top1.plot(kind="bar", colormap="Set2", ax=ax5)
ax5.set_title("Gender Composition in Top 1% Scorers by Category")
ax5.set_ylabel("Number of Students")
ax5.tick_params(axis="x", rotation=45)
ax5.grid(True)
st.pyplot(fig5)
st.markdown(
    """
#### Gender-Based Trends:
- More **girls** than boys made it to top scores in **General and OBC** categories.
- But **SC/ST** top scorers still have more **boys**.

**Why it matters:** Reserved category girls may need **more support, mentoring, or scholarships**. If you’re one of them, you’re already ahead of many peers!
"""
)

# Additional Insights
st.subheader("📌 Additional Insights for Students")
st.markdown(
    """
- 🔻 **Scores Dropped in 2025**: Indicates potential relaxation in cutoffs.
- 🎯 **80% Students Scored Below 400**: More competition in mid-score bands.
- 📉 **High Scorers in 2025 are fewer**: Those with 400+ now have stronger advantage.
- 🔄 **Category-wise Compression**: SC/ST/EWS show tighter IQR—closer competition.
- 🧠 **Tip**: Check if your score is above 2024 median of your category for a stronger chance.
"""
)
