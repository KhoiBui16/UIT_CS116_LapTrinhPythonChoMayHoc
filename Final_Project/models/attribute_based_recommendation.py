import pandas as pd

class AttributeBasedRecommendation:
    def __init__(self, data):
        """
        Khởi tạo hệ thống gợi ý dựa trên thuộc tính.
        :param data: DataFrame chứa thông tin sản phẩm và các thuộc tính liên quan.
        """
        self.data = data

    def get_recommendations(self, filters, top_n=5):
        """
        Gợi ý sản phẩm dựa trên các thuộc tính được chọn.
        :param filters: Dictionary chứa các thuộc tính và giá trị cần lọc (ví dụ: {'Category': 'Clothing', 'Size': 'M'}).
        :param top_n: Số lượng sản phẩm gợi ý.
        :return: Danh sách các sản phẩm được gợi ý.
        """
        # Lọc dữ liệu dựa trên các thuộc tính
        filtered_data = self.data.copy()
        for attribute, value in filters.items():
            if attribute in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[attribute] == value]

        # Nếu không có sản phẩm nào phù hợp, trả về thông báo
        if filtered_data.empty:
            return []

        # Sắp xếp sản phẩm theo điểm tương tác (Interaction_Score) hoặc một tiêu chí khác
        if 'Interaction_Score' in filtered_data.columns:
            filtered_data = filtered_data.sort_values(by='Interaction_Score', ascending=False)

        # Lấy top N sản phẩm
        recommendations = filtered_data['Item Purchased'].head(top_n).tolist()
        return recommendations