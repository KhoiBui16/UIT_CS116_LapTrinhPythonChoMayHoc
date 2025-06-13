import streamlit as st
import pandas as pd
from models.recommendation_model import RecommendationModel

st.set_page_config(page_title="🎯 Hybrid Recommendation System", layout="centered")


# Hàm reverse encode
def reverse_encode_category(df):
    category_columns = [col for col in df.columns if col.startswith("Category_")]
    if category_columns:
        df["Category"] = (
            df[category_columns].idxmax(axis=1).str.replace("Category_", "")
        )
    return df


def reverse_encode_review_rating(df):
    if "Review Rating" in df.columns:
        df["Review Rating"] = df["Review Rating"].astype(float)
    return df


# Load cả hai file: processed để lọc, updated để lấy rating gốc
df_proc = pd.read_csv("data/shopping_behavior_processed.csv")
df_raw = pd.read_csv("data/shopping_behavior_updated.csv")

# Reverse encode, giải mã rating
df_proc = reverse_encode_category(df_proc)
df_proc = reverse_encode_review_rating(df_proc)
df_raw = reverse_encode_review_rating(df_raw)

# Min/max rating
min_rating, max_rating = 0.0, 5.0

# Giao diện
st.sidebar.title("📦 Product Recommendation System")

page = st.sidebar.radio("📄 Chọn trang", ["Danh sách sản phẩm", "Gợi ý sản phẩm"])

if page == "Danh sách sản phẩm":
    st.title("📋 DANH SÁCH SẢN PHẨM")

    unique_products = (
        df_proc.groupby("Item Purchased")
        .agg(
            {
                "Category": lambda x: ", ".join(sorted(x.dropna().unique())),
                "Size": lambda x: ", ".join(sorted(x.dropna().unique())),
                "Color": lambda x: ", ".join(sorted(x.dropna().unique())),
            }
        )
        .reset_index()
    )

    mean_ratings = (
        df_raw.groupby("Item Purchased")["Review Rating"]
        .mean()
        .round(1)
        .reset_index(name="Review Rating")
    )
    unique_products = unique_products.merge(mean_ratings, on="Item Purchased")
    st.subheader("🎯 Thông tin sản phẩm hiện có:")
    st.dataframe(unique_products)

elif page == "Gợi ý sản phẩm":
    st.markdown(
        "<h1 style='text-align: center;'>🎯 GỢI Ý SẢN PHẨM</h1>", unsafe_allow_html=True
    )
    filters = {}

    # mode = st.radio("Chọn phương pháp gợi ý:", ("Theo cá nhân")) # ("Theo Customers", "Theo cá nhân"))
    # cust_id = None
    # if mode.startswith("Theo Customers"):
    # pass
    # customer_ids = sorted(df_proc['Customer ID'].dropna().unique().astype(int).tolist())
    # cust_sel = st.selectbox("Chọn Customer ID:", [''] + [str(i) for i in customer_ids])
    # cust_id = int(cust_sel) if cust_sel else None

    # if not cust_id:
    #     st.warning("Vui lòng chọn Customer ID.")
    # else:

    allowed_columns = ["Category", "Location", "Season"]
    selected_cols = st.multiselect("Chọn thông tin mô tả:", allowed_columns)

    for col in selected_cols:
        values = df_proc[col].dropna().unique().tolist()
        filters[col] = st.selectbox(f"{col}:", values, key=col)

    if "Category" in selected_cols:
        sub_attrs = st.multiselect(
            "Chọn thêm thuộc tính:",
            [
                "Size",
                "Color",
                "Review Rating",
                "Subscription Status",
                "Shipping Type",
                "Payment Method",
            ],
        )
        for attr in sub_attrs:
            if attr == "Review Rating":
                filters[attr] = st.slider(
                    "Rating tối thiểu:", min_rating, max_rating, min_rating, 0.1
                )
            else:
                values = df_proc[attr].dropna().unique().tolist()
                filters[attr] = st.selectbox(f"{attr}:", values, key=attr + "_sub")

    if st.button("🚀 Đề xuất"):
        model = RecommendationModel(df_proc, df_raw)
        recs = []

        # if mode.startswith("Theo Customers") and cust_id:
        #     try:
        #         recs = model.get_recommendations(customer_id=cust_id, top_n=5)
        #     except Exception as e:
        #         st.error(f"Lỗi CF: {e}")

        if filters:
            try:
                recs = model.get_recommendations(filters=filters, top_n=5)
            except Exception as e:
                st.error(f"Lỗi content-based: {e}")
        else:
            st.warning("Cần chọn thông tin để gợi ý.")
            st.stop()

        positive_recs = [(item, score) for item, score in recs if score > 0]

        if not positive_recs:
            st.warning("Không có sản phẩm nào phù hợp.")
        else:
            st.subheader("🔮 Top đề xuất:")
            total = sum(score for _, score in positive_recs)
            for i, (item, score) in enumerate(positive_recs, 1):
                pct = round(score / total * 100, 1)
                st.markdown(
                    f"""
                    <div class="recommendation-item" style="
                        background-color:#e9ecef;
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 4px;">
                        <strong>{i}. {item}</strong> — ({pct}%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
<style>
    /* Đổi background màu nhạt */
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Tùy chỉnh container */
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
    }

    /* Các thẻ gợi ý */
    .recommendation-item {
        background-color: #e9ecef;
        padding: 10px;`
        border-radius: 5px;
        margin: 10px 0;
        transition: 0.3s ease;
    }

    .recommendation-item:hover {
        background-color: #d0e0ff;
    }
</style>
""",
        unsafe_allow_html=True,
    )
