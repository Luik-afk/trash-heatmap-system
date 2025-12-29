import streamlit as st
import pandas as pd
import math
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN, KMeans
from streamlit_folium import st_folium

st.set_page_config(page_title="쓰레기 투기 분석 시스템", layout="wide")

st.title("🗺️ 데이터 기반 쓰레기 분포 분석 시스템")

# ----------------------
# 파일 업로드
# ----------------------
st.sidebar.header("📂 데이터 업로드")
events_file = st.sidebar.file_uploader("쓰레기 투기 데이터 CSV", type="csv")
bins_file = st.sidebar.file_uploader("쓰레기통 위치 CSV", type="csv")

if events_file and bins_file:
    df_events = pd.read_csv(events_file)
    df_bins = pd.read_csv(bins_file)

    lat0 = df_events["lat"].mean()
    lon0 = df_events["lon"].mean()

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371008.8
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        return 2*R*math.asin(math.sqrt(a))

    if st.button("🚀 분석 실행"):

        # ----------------------
        # 1. 히트맵
        # ----------------------
        m1 = folium.Map(location=[lat0, lon0], zoom_start=15)
        heat_data = [[r.lat, r.lon, r.confidence] for _, r in df_events.iterrows()]
        HeatMap(heat_data, radius=18).add_to(m1)

        st.subheader("🔥 쓰레기 투기 히트맵")
        st_folium(m1, width=700, height=500)

        # ----------------------
        # 2. DBSCAN
        # ----------------------
        coords_rad = np.radians(df_events[["lat", "lon"]])
        db = DBSCAN(
            eps=(45/1000)/6371,
            min_samples=10,
            metric="haversine"
        ).fit(coords_rad)

        df_events["cluster"] = db.labels_

        # ----------------------
        # 3. k-means
        # ----------------------
        uncovered = df_events[df_events["cluster"] != -1]
        if len(uncovered) >= 4:
            km = KMeans(n_clusters=4, random_state=42)
            km.fit(uncovered[["lat", "lon"]])
            centers = km.cluster_centers_

            m2 = folium.Map(location=[lat0, lon0], zoom_start=15)
            for _, r in df_bins.iterrows():
                folium.CircleMarker(
                    [r.lat, r.lon],
                    radius=4,
                    popup="기존 쓰레기통",
                    color="blue"
                ).add_to(m2)

            for i, c in enumerate(centers):
                folium.Marker(
                    c.tolist(),
                    popup=f"신규 후보 {i+1}",
                    icon=folium.Icon(color="red")
                ).add_to(m2)

            st.subheader("➕ 신규 쓰레기통 설치 후보")
            st_folium(m2, width=700, height=500)

else:
    st.info("왼쪽에서 CSV 파일을 업로드하세요.")
