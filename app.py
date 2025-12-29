import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

# =====================
# 기본 설정
# =====================
st.set_page_config(page_title="쓰레기 투기 분석 시스템", layout="wide")
st.title("🗺️ 데이터 기반 쓰레기 분포 · 쓰레기통 배치 시스템")

K_FIXED = 55

# =====================
# 세션 상태
# =====================
if "run" not in st.session_state:
    st.session_state.run = False

# =====================
# 좌표 변환 (근방 평면 근사)
# =====================
def ll_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111_000 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 111_000
    return x, y

def xy_to_ll(x, y, lat0, lon0):
    lon = x / (111_000 * math.cos(math.radians(lat0))) + lon0
    lat = y / 111_000 + lat0
    return lat, lon

# =====================
# Voronoi 유한 다각형
# =====================
def voronoi_finite_polygons(vor, radius=8000):
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far = vor.vertices[v2] + direction * radius

            new_vertices.append(far.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# =====================
# 파일 업로드
# =====================
st.sidebar.header("📂 데이터 업로드")
events_file = st.sidebar.file_uploader("쓰레기 투기 데이터 CSV", type="csv")
bins_file = st.sidebar.file_uploader("기존 쓰레기통 CSV", type="csv")

if events_file and bins_file:
    df_events = pd.read_csv(events_file)
    df_bins = pd.read_csv(bins_file)

    if "confidence" not in df_events.columns:
        df_events["confidence"] = 1.0

    lat0 = df_events["lat"].mean()
    lon0 = df_events["lon"].mean()

    if st.button("🚀 분석 실행"):
        st.session_state.run = True

    if st.session_state.run:

        # =====================
        # 히트맵 + K=55 쓰레기통 배치
        # =====================
        pts_xy = np.array([
            ll_to_xy(r.lat, r.lon, lat0, lon0)
            for _, r in df_events.iterrows()
        ])
        weights = df_events["confidence"].to_numpy()

        km = KMeans(n_clusters=K_FIXED, random_state=42, n_init=20)
        km.fit(pts_xy, sample_weight=weights)
        centers_xy = km.cluster_centers_

        # =====================
        # Voronoi
        # =====================
        vor = Voronoi(centers_xy)
        regions, vertices = voronoi_finite_polygons(vor)

        # =====================
        # 지도 생성
        # =====================
        m = folium.Map(location=[lat0, lon0], zoom_start=15)

        heat_data = [[r.lat, r.lon, r.confidence] for _, r in df_events.iterrows()]
        HeatMap(heat_data, radius=18, blur=16).add_to(m)

        palette = [
            "red", "blue", "green", "purple", "orange",
            "darkred", "cadetblue", "darkgreen"
        ]

        for i, region in enumerate(regions):
            poly = vertices[region]
            ll_poly = [xy_to_ll(x, y, lat0, lon0) for x, y in poly]

            folium.Polygon(
                locations=ll_poly,
                color=palette[i % len(palette)],
                fill=True,
                fill_opacity=0.12,
                weight=1,
                popup=f"관할 영역 H{i+1}"
            ).add_to(m)

        # 신규 쓰레기통
        for i, (x, y) in enumerate(centers_xy):
            la, lo = xy_to_ll(x, y, lat0, lon0)
            folium.CircleMarker(
                [la, lo],
                radius=3,
                color="red",
                fill=True,
                popup=f"신규 쓰레기통 H{i+1}"
            ).add_to(m)

        # 기존 쓰레기통
        for _, r in df_bins.iterrows():
            folium.CircleMarker(
                [r.lat, r.lon],
                radius=4,
                color="blue",
                fill=True,
                popup="기존 쓰레기통"
            ).add_to(m)

        st.subheader("🔥 히트맵 기반 쓰레기통 55개 + NVD 관할 구역")
        st_folium(m, width=1200, height=650)

else:
    st.info("왼쪽에서 CSV 파일을 업로드하세요.")
