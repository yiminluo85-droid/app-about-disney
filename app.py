import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读数据
df = pd.read_csv("disney_plus_shows.csv")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["score"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
df["runtime"] = df["runtime"].astype(str).str.extract('(\\d+)').astype(float)

# ===== 侧边栏：超丰富筛选 =====
st.sidebar.title("多维度过滤器")

# 1. 类型
content = st.sidebar.radio("类型", ["All", "Movie", "TV Show"])

# 2. 年份区间
y0, y1 = st.sidebar.slider("年份区间", 1920, 2025, (1990, 2010))

# 3. 评分区间
s0, s1 = st.sidebar.slider("评分区间", 0.0, 10.0, (0.0, 10.0))

# 4. 时长区间
r0, r1 = st.sidebar.slider("时长(min)", 0, 300, (0, 300))

# 5. TOP N
top_n = st.sidebar.slider("TOP N", 3, 20, 10)

# 6. 国家关键词
country_kw = st.sidebar.text_input("国家关键词（留空=全部）")

# 7. 标题关键词
title_kw = st.sidebar.text_input("标题关键词（留空=全部）")

# 8. 最低投票人数
min_votes = st.sidebar.number_input("最低投票人数", min_value=0, value=0)

# 9. 题材关键词
genre_kw = st.sidebar.text_input("题材关键词（留空=全部）")

# ===== 数据过滤 =====
d = df.copy()

if content != "All":
    d = d[d["type"] == content]

d = d[
    (d["year"] >= y0) & (d["year"] <= y1) &
    (d["score"] >= s0) & (d["score"] <= s1) &
    (d["runtime"] >= r0) & (d["runtime"] <= r1)
]

if country_kw:
    d = d[d["country"].fillna("").str.contains(country_kw, case=False, na=False)]

if title_kw:
    d = d[d["title"].fillna("").str.contains(title_kw, case=False, na=False)]

if genre_kw:
    d = d[d["genre"].fillna("").str.contains(genre_kw, case=False, na=False)]

if min_votes > 0:
    d = d[d["imdb_votes"].astype(str).str.replace(",", "").astype(float) >= min_votes]

# ===== 展示 =====
st.title("Disney+ 超级筛选仪表盘")

col1, col2, col3, col4 = st.columns(4)
col1.metric("剩余条数", len(d))
col2.metric("平均评分", round(d["score"].mean(), 2))
col3.metric("平均时长(min)", round(d["runtime"].mean(), 1))
col4.metric("最高评分", round(d["score"].max(), 2))

st.subheader("① 类型平均评分")
type_tb = d.groupby("type")["score"].mean().reset_index()
fig1, ax1 = plt.subplots()
sns.barplot(data=type_tb, x="type", y="score", ax=ax1)
st.pyplot(fig1)

st.subheader("② 年份-评分折线")
year_tb = d.groupby("year")["score"].mean().dropna().reset_index()
fig2, ax2 = plt.subplots()
sns.lineplot(data=year_tb, x="year", y="score", marker="o", ax=ax2)
plt.xticks(rotation=90)
st.pyplot(fig2)

st.subheader("③ TOP 国家/地区平均评分")
country_tb = (d.dropna(subset=["country"])
                .assign(country=lambda x: x["country"].str.split(",").str[0])
                .groupby("country")["score"]
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
fig3, ax3 = plt.subplots()
sns.barplot(data=country_tb, x="score", y="country", ax=ax3)
st.pyplot(fig3)

st.subheader("④ 评分分布")
fig4, ax4 = plt.subplots()
sns.histplot(d["score"].dropna(), bins=20, kde=True, ax=ax4)
st.pyplot(fig4)

st.subheader("⑤ 时长-评分散点")
fig5, ax5 = plt.subplots()
sns.scatterplot(data=d, x="runtime", y="score", ax=ax5)
st.pyplot(fig5)

st.subheader("⑥ 题材词频（前 20）")
genre_list = d["genre"].dropna().str.split(",").explode().str.strip()
genre_top = genre_list.value_counts().head(20).reset_index()
genre_top.columns = ["genre", "count"]
fig6, ax6 = plt.subplots()
sns.barplot(data=genre_top, x="count", y="genre", ax=ax6)
st.pyplot(fig6)

st.subheader("⑦ TOP 作品表")
top_df = d.nlargest(top_n, "score")[["title", "year", "score", "type", "runtime"]]
st.dataframe(top_df)

with st.expander("⑧ 查看原始样本（前 100）"):
    st.dataframe(d.head(100))