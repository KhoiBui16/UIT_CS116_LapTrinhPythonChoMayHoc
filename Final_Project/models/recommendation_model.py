import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationModel:
    def __init__(self, proc_df, raw_df=None):
        """
        proc_df: DataFrame đã one-hot + có cột Interaction_Score
        raw_df: nếu cần lấy ngưỡng Review Rating gốc
        """
        self.proc = proc_df
        self.raw = raw_df
        self.user_item = self._make_ui_matrix()

    def _make_ui_matrix(self):
        return self.proc.pivot_table(
            index="Customer ID",
            columns="Item Purchased",
            values="Interaction_Score",
            aggfunc="mean"
        ).fillna(0)

    def get_recommendations(self, customer_id=None, filters=None, top_n=5):
        # --- Collaborative Filtering ---
        if customer_id is not None:
            if customer_id not in self.user_item.index:
                raise ValueError(f"Không tìm thấy Customer ID {customer_id}")
            sim = cosine_similarity(self.user_item, self.user_item.loc[[customer_id]])
            sim = pd.Series(sim.ravel(), index=self.user_item.index)
            sim = sim.drop(customer_id)
            top_users = sim.sort_values(ascending=False).index

            bought = set(self.user_item.loc[customer_id].gt(0).pipe(lambda s: s[s].index))
            scores = {}
            for u in top_users:
                for item, sc in self.user_item.loc[u].items():
                    if sc > 0 and item not in bought:
                        scores[item] = scores.get(item, 0) + sc

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # --- Content/Attribute-based Filtering ---
        elif filters:
            df = self.proc.copy()
            thr = filters.pop("Review Rating", None) if "Review Rating" in filters else None

            for col, val in filters.items():
                if col in df.columns:
                    df = df[df[col] == val]

            agg = (
                df.groupby("Item Purchased")["Interaction_Score"]
                .sum()
                .sort_values(ascending=False)
            )

            if thr is not None and self.raw is not None:
                avg_rating = self.raw.groupby("Item Purchased")["Review Rating"].mean()
                valid_items = avg_rating[avg_rating >= thr].index
                agg = agg[agg.index.isin(valid_items)]

            return list(agg.head(top_n).items())

        else:
            return []
