import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

triplets = [
    ("Valve Fault Diagnosis", "includes", "Seat Leakage"),
    ("Valve Fault Diagnosis", "includes", "Packing Leakage"),
    ("Valve Fault Diagnosis", "includes", "Cavitation"),
    ("Valve Fault Diagnosis", "includes", "Loose Fastener"),
    ("Valve Fault Diagnosis", "includes", "Foreign Object"),

    ("Seat Leakage", "has_symptom", "RMS Increase"),
    ("Seat Leakage", "has_symptom", "High Frequency Leakage Sound"),
    ("Seat Leakage", "has_symptom", "Pressure Drop"),
    ("Seat Leakage", "caused_by", "Seat Wear"),
    ("Seat Leakage", "caused_by", "Foreign Object Ingress"),
    ("Seat Leakage", "caused_by", "Disc Damage"),
    ("Seat Leakage", "requires_action", "Pipe Isolation"),
    ("Seat Leakage", "requires_action", "Pressure Release"),
    ("Seat Leakage", "requires_action", "Seat Inspection and Cleaning"),
    ("Seat Leakage", "requires_action", "Leak Test"),

    ("Packing Leakage", "has_symptom", "Stem Leakage"),
    ("Packing Leakage", "has_symptom", "Spectral Centroid Increase"),
    ("Packing Leakage", "caused_by", "Packing Aging"),
    ("Packing Leakage", "caused_by", "Insufficient Gland Tightening"),
    ("Packing Leakage", "requires_action", "Packing Replacement"),
    ("Packing Leakage", "requires_action", "Stem Surface Inspection"),

    ("Cavitation", "has_symptom", "Gravel-like Noise"),
    ("Cavitation", "has_symptom", "FFT Peak Increase"),
    ("Cavitation", "caused_by", "Excessive Pressure Drop"),
    ("Cavitation", "caused_by", "Excessive Flow Velocity"),
    ("Cavitation", "requires_action", "Operating Pressure Adjustment"),
    ("Cavitation", "requires_action", "Valve Selection Review"),

    ("Loose Fastener", "has_symptom", "Low Frequency Vibration Increase"),
    ("Loose Fastener", "caused_by", "Loose Bolting"),
    ("Loose Fastener", "requires_action", "Bolt Tightening Check"),

    ("Foreign Object", "has_symptom", "Intermittent Impact Noise"),
    ("Foreign Object", "caused_by", "Foreign Object Ingress"),
    ("Foreign Object", "requires_action", "Line Flushing"),
]

df = pd.DataFrame(triplets, columns=["subject", "relation", "object"])

print("=" * 80)
print("GraphRAG Triplet 검증")
print("=" * 80)

# 1. Triplet 무결성 검증
empty_rows = df[
    df["subject"].isna() | df["relation"].isna() | df["object"].isna() |
    (df["subject"].str.strip() == "") |
    (df["relation"].str.strip() == "") |
    (df["object"].str.strip() == "")
]

print("Triplet Count:", len(df))
print("Empty / None Row Count:", len(empty_rows))

if len(empty_rows) == 0:
    print("검증 결과 1: Triplet 형식이 정상입니다.")
else:
    print("검증 결과 1: 비어 있는 Triplet이 있습니다.")

# 2. NetworkX 그래프 생성
G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_edge(row["subject"], row["object"], relation=row["relation"])

print("Node Count:", G.number_of_nodes())
print("Edge Count:", G.number_of_edges())

components = list(nx.weakly_connected_components(G))
print("Connected Component Count:", len(components))

if len(components) == 1:
    print("검증 결과 2: 전체 그래프가 하나의 연결 구조로 구성되었습니다.")
else:
    print("검증 결과 2: 그래프가 여러 컴포넌트로 분리되어 있습니다.")

# 3. CSV 저장
df.to_csv("graph_triplets.csv", index=False, encoding="utf-8-sig")

# 4. 그래프 이미지 저장
plt.figure(figsize=(18, 12))
pos = nx.spring_layout(G, seed=42, k=1.0)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2200,
    font_size=8,
    arrows=True
)

edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=7
)

plt.title("ShipLeak-Copilot GraphRAG Network")
plt.tight_layout()
plt.savefig("graph_network.png", dpi=200)

print("저장 완료: graph_triplets.csv")
print("저장 완료: graph_network.png")
print("=" * 80)
print("GraphRAG 검증 완료")
print("=" * 80)