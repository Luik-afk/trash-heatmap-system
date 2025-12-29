import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from streamlit_folium import st_folium

# =====================
# 설정
# =====================
st.set_page_config(page_title="쓰레기 투기 NVD 분석", layout="wide")
st.title("🗺️ 히트맵 기반 쓰레기통 배치 + 보로노이 분석")

K_FIXED = 55

LAT0 = 36 + 21/60 + 52.765/3600
LON0 = 127 + 21/60 + 13.525/3600


# =====================
# 좌표 변환
# =====================
def ll_to_xy(lat, lon):
    x = (lon - LON0) * 111_000 * math.cos(math.radians(LAT0))
    y = (lat - LAT0) * 111_000
    return x, y

def xy_to_ll(x, y):
    lon = x / (111_000 * math.cos(math.radians(LAT0))) + LON0
    lat = y / 111_000 + LAT0
    return lat, lon


# =====================
# Voronoi 보정
# =====================
def voronoi_finite_polygons_2d(vor, radius=8000):
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        new_region = [v for v in region if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1]-c[1], vs[:,0]-c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


# =====================
# 업로드
# =====================
st.sidebar.header("📂 투기 이벤트 CSV 업로드")
events_file = st.sidebar.file_uploader("event_id, lat, lon, confidence", type="csv")

if events_file:
    df = pd.read_csv(events_file)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0

    pts_xy = np.array([ll_to_xy(r.lat, r.lon) for _, r in df.iterrows()])
    weights = df["confidence"].values

    if st.button("🚀 분석 실행"):

        # =====================
        # 1. KMeans (K=55)
        # =====================
        km = KMeans(n_clusters=K_FIXED, random_state=1416, n_init=20)
        km.fit(pts_xy, sample_weight=weights)
        bins_xy = km.cluster_centers_

        # =====================
        # 2. Voronoi
        # =====================
        vor = Voronoi(bins_xy)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        # =====================
        # 3. 지도 생성
        # =====================
        m = folium.Map(location=[LAT0, LON0], zoom_start=15)

        HeatMap(
            [[r.lat, r.lon, r.confidence] for _, r in df.iterrows()],
            radius=18, blur=16, min_opacity=0.3
        ).add_to(m)

        for i, reg in enumerate(regions):
            poly = vertices[reg]
            ll_poly = [xy_to_ll(x, y) for x, y in poly]

            folium.Polygon(
                locations=ll_poly,
                fill=True,
                fill_opacity=0.1,
                weight=1,
                popup=f"H{i+1}"
            ).add_to(m)

        for i, (x, y) in enumerate(bins_xy, start=1):
            la, lo = xy_to_ll(x, y)
            folium.CircleMarker(
                [la, lo],
                radius=3,
                popup=f"H{i}",
                fill=True,
                color="red"
            ).add_to(m)

        st.subheader("🧭 히트맵 + Voronoi NVD (K=55)")
        st_folium(m, width=1200, height=700)

else:
    st.info("왼쪽에서 CSV 파일을 업로드하세요.")
