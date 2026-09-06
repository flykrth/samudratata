#!/usr/bin/env python3
"""
visualize_graph.py
==================
Publication-Quality Visualization Suite for Coastal Graph Topology.

Uses NetworkX and Matplotlib to plot the finalized PyTorch Geometric coastal graph
(`south_india_coastal_graph.pt`) overlaid on geographic coordinates, visually
proving node connectivity, bidirectional linear chains, and directional
seasonal longshore drift edges across all 5 study zones:
- Chellanam (SW Coast, Arabian Sea, Kerala)
- Alappuzha (SW Coast, Arabian Sea, Kerala)
- Nagapattinam (SE Coast, Bay of Bengal, Tamil Nadu)
- Cuddalore (SE Coast, Bay of Bengal, Tamil Nadu)
- Visakhapatnam (SE Coast, Bay of Bengal, Andhra Pradesh)

Outputs:
--------
- data/coastal_graph_topology.png: Multi-panel high-resolution visualization figure.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from shapely.geometry import LineString, shape
import torch
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("VisualizeCoastalGraph")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_PT = BASE_DIR / "data" / "south_india_coastal_graph.pt"
DEFAULT_COASTLINE_GEOJSON = BASE_DIR / "data" / "coastline" / "south_india_coastline.geojson"
DEFAULT_OUTPUT_PNG = BASE_DIR / "data" / "coastal_graph_topology.png"

ZONE_METRICS = {
    "Chellanam": {"color": "#1f77b4", "coast": "SW Coast (Arabian Sea)", "monsoon": "SW Monsoon (S → N)"},
    "Alappuzha": {"color": "#2ca02c", "coast": "SW Coast (Arabian Sea)", "monsoon": "SW Monsoon (S → N)"},
    "Nagapattinam": {"color": "#ff7f0e", "coast": "SE Coast (Bay of Bengal)", "monsoon": "NE Monsoon (N → S)"},
    "Cuddalore": {"color": "#9467bd", "coast": "SE Coast (Bay of Bengal)", "monsoon": "NE Monsoon (N → S)"},
    "Visakhapatnam": {"color": "#d62728", "coast": "SE Coast (Bay of Bengal)", "monsoon": "NE Monsoon (N → S)"},
}


def create_networkx_graph(pyg_data: Data) -> nx.DiGraph:
    """Converts a PyG Data object into a NetworkX DiGraph with GIS node & edge attributes."""
    G = nx.DiGraph()
    N = pyg_data.num_nodes
    pos = pyg_data.pos.numpy()
    elev = pyg_data.elevation.numpy()
    veg_dist = pyg_data.veg_dist.numpy()
    bathy_depth = pyg_data.bathymetric_depth.numpy()
    zone_idx = pyg_data.zone_idx.numpy()

    for i in range(N):
        G.add_node(
            i,
            pos=(float(pos[i, 0]), float(pos[i, 1])),
            lon=float(pos[i, 0]),
            lat=float(pos[i, 1]),
            elevation=float(elev[i]),
            veg_dist=float(veg_dist[i]),
            bathy_depth=float(bathy_depth[i]),
            zone_idx=int(zone_idx[i]),
        )

    edge_index = pyg_data.edge_index.numpy()
    edge_attr = pyg_data.edge_attr.numpy()
    E = edge_index.shape[1]

    for e in range(E):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        dist_m = float(edge_attr[e, 0])
        bathy_grad = float(edge_attr[e, 1])
        drift_flag = float(edge_attr[e, 2])
        G.add_edge(
            u, v,
            distance_m=dist_m,
            bathy_grad=bathy_grad,
            drift_flag=drift_flag,
        )

    return G


def render_graph_visualization(
    graph_pt_path: Path = DEFAULT_GRAPH_PT,
    coastline_geojson_path: Path = DEFAULT_COASTLINE_GEOJSON,
    output_png_path: Path = DEFAULT_OUTPUT_PNG,
    dpi: int = 300,
):
    """
    Generates a multi-panel visual proof of coastal graph topology and connectivity.
    """
    log.info("Loading PyG graph from: %s", graph_pt_path)
    pyg_data: Data = torch.load(graph_pt_path, map_location="cpu", weights_only=False)

    log.info("Converting to NetworkX DiGraph...")
    G = create_networkx_graph(pyg_data)
    pos_dict = nx.get_node_attributes(G, "pos")

    # Load coastline GeoJSON
    with open(coastline_geojson_path, "r") as f:
        coast_geojson = json.load(f)

    zone_geoms: Dict[str, LineString] = {
        feat["properties"]["zone"]: shape(feat["geometry"])
        for feat in coast_geojson["features"]
    }

    zone_names = pyg_data.zone_names
    zone_idx_arr = pyg_data.zone_idx.numpy()

    # Setup publication layout: 3 columns x 3 rows
    fig = plt.figure(figsize=(20, 16), facecolor="#fdfdfd")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

    # -------------------------------------------------------------------------
    # Panel 1: Regional South India Overview (Left Column, spans 2 rows)
    # -------------------------------------------------------------------------
    ax_macro = fig.add_subplot(gs[0:2, 0])
    ax_macro.set_facecolor("#f0f4f8")

    # Draw peninsular India outline approx
    ax_macro.plot([76.0, 75.0, 76.5, 77.5, 79.8, 80.3, 83.3, 85.0],
                  [12.5, 14.0, 8.1, 8.0, 9.8, 13.0, 17.7, 19.5],
                  color="#b0bec5", linestyle="--", linewidth=1.2, alpha=0.7, label="Peninsular Coastline")

    # Plot nodes and connections for all 5 zones on regional scale
    for z_idx, z_name in enumerate(zone_names):
        z_nodes = [n for n in G.nodes if G.nodes[n]["zone_idx"] == z_idx]
        z_lons = [pos_dict[n][0] for n in z_nodes]
        z_lats = [pos_dict[n][1] for n in z_nodes]

        color = ZONE_METRICS[z_name]["color"]
        ax_macro.scatter(z_lons, z_lats, s=25, color=color, zorder=5, label=f"{z_name} ({len(z_nodes)} nodes)")

        # Zone annotation box
        mid_lon, mid_lat = float(np.mean(z_lons)), float(np.mean(z_lats))
        offset_x = -1.2 if "SW" in ZONE_METRICS[z_name]["coast"] else 0.8
        ax_macro.annotate(
            f"{z_name}\n({len(z_nodes)} nodes)",
            xy=(mid_lon, mid_lat),
            xytext=(mid_lon + offset_x, mid_lat + 0.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            fontsize=8, fontweight="bold",
        )

    # Regional hydrodynamic drift vectors
    # SW Monsoon on West Coast (Arabian Sea): South to North
    ax_macro.annotate(
        "SW Monsoon Drift (S → N)\n[Arabian Sea]",
        xy=(75.8, 10.5), xytext=(75.0, 8.8),
        arrowprops=dict(facecolor="#0288d1", edgecolor="#01579b", width=3, headwidth=8),
        fontsize=9, fontweight="bold", color="#01579b", ha="center"
    )

    # NE Monsoon Drift on East Coast (Bay of Bengal): North to South
    ax_macro.annotate(
        "NE Monsoon Drift (N → S)\n[Bay of Bengal]",
        xy=(81.0, 11.5), xytext=(82.2, 16.5),
        arrowprops=dict(facecolor="#d84315", edgecolor="#bf360c", width=3, headwidth=8),
        fontsize=9, fontweight="bold", color="#bf360c", ha="center"
    )

    ax_macro.set_title("A. South India Coastal Graph Overview\n(5 Study Zones | N=718 Nodes | E=2134 Edges)",
                       fontsize=11, fontweight="bold", pad=10)
    ax_macro.set_xlabel("Longitude (°E)", fontsize=9)
    ax_macro.set_ylabel("Latitude (°N)", fontsize=9)
    ax_macro.grid(True, linestyle=":", alpha=0.5)
    ax_macro.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
    ax_macro.set_xlim(74.0, 85.5)
    ax_macro.set_ylim(7.5, 19.0)

    # -------------------------------------------------------------------------
    # Panels 2-6: Zone-by-Zone Detailed Graph Topology Plots
    # -------------------------------------------------------------------------
    zone_subplots = [
        (gs[0, 1], "Chellanam", "B1"),
        (gs[0, 2], "Alappuzha", "B2"),
        (gs[1, 1], "Nagapattinam", "B3"),
        (gs[1, 2], "Cuddalore", "B4"),
        (gs[2, 0], "Visakhapatnam", "B5"),
    ]

    for cell, z_name, label_p in zone_subplots:
        ax = fig.add_subplot(cell)
        z_idx = zone_names.index(z_name)
        z_nodes = [n for n in G.nodes if G.nodes[n]["zone_idx"] == z_idx]
        subG = G.subgraph(z_nodes)

        z_color = ZONE_METRICS[z_name]["color"]
        drift_desc = ZONE_METRICS[z_name]["monsoon"]

        # Draw official coastline line
        if z_name in zone_geoms:
            cline = zone_geoms[z_name]
            cx, cy = zip(*list(cline.coords))
            ax.plot(cx, cy, color="#78909c", linewidth=2.5, alpha=0.6, label="Official Coastline (10m)")

        # Draw bidirectional chain edges (adjacent: distance ~ 100m)
        chain_edges = [
            (u, v) for u, v, d in subG.edges(data=True)
            if d.get("distance_m", 0.0) <= 150.0
        ]
        for u, v in chain_edges:
            p1, p2 = pos_dict[u], pos_dict[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#37474f", linewidth=1.2, alpha=0.7, zorder=2)

        # Draw directional longshore drift edges (2-hop transport: distance > 150m)
        drift_edges = [
            (u, v) for u, v, d in subG.edges(data=True)
            if d.get("distance_m", 0.0) > 150.0 and d.get("drift_flag", 0.0) == 1.0
        ]
        # Plot subset of drift arrows to maintain visual clarity
        stride_arrow = max(1, len(drift_edges) // 18)
        for idx_e, (u, v) in enumerate(drift_edges):
            if idx_e % stride_arrow == 0:
                p1, p2 = pos_dict[u], pos_dict[v]
                ax.annotate(
                    "",
                    xy=(p2[0], p2[1]),
                    xytext=(p1[0], p1[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=z_color,
                        lw=1.8,
                        mutation_scale=12,
                        alpha=0.85,
                    ),
                    zorder=3,
                )

        # Draw nodes colored by elevation
        z_lons = np.array([pos_dict[n][0] for n in z_nodes])
        z_lats = np.array([pos_dict[n][1] for n in z_nodes])
        z_elev = np.array([G.nodes[n]["elevation"] for n in z_nodes])

        sc = ax.scatter(
            z_lons, z_lats,
            c=z_elev, cmap="viridis", s=30, edgecolors="#263238", linewidth=0.5, zorder=4
        )

        # Annotate start node (South) and end node (North)
        ax.scatter(z_lons[0], z_lats[0], color="#00e676", s=70, marker="o", edgecolors="black", zorder=6)
        ax.scatter(z_lons[-1], z_lats[-1], color="#d50000", s=70, marker="s", edgecolors="black", zorder=6)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Elevation (m)", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"{label_p}. {z_name} ({ZONE_METRICS[z_name]['coast']})\n"
                     f"{len(z_nodes)} Transects | {drift_desc}",
                     fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("Longitude (°E)", fontsize=8)
        ax.set_ylabel("Latitude (°N)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel 7: Node Degree Distribution Histogram (Row 2, Col 1)
    # -------------------------------------------------------------------------
    ax_deg = fig.add_subplot(gs[2, 1])
    edge_index = pyg_data.edge_index.numpy()
    N = pyg_data.num_nodes
    out_deg = np.bincount(edge_index[0], minlength=N)
    in_deg = np.bincount(edge_index[1], minlength=N)
    tot_deg = out_deg + in_deg

    deg_bins = np.arange(2.5, 7.5, 1.0)
    ax_deg.hist(tot_deg, bins=deg_bins, color="#455a64", edgecolor="black", rwidth=0.7, alpha=0.85)
    ax_deg.axvline(tot_deg.mean(), color="#e53935", linestyle="--", linewidth=1.5,
                   label=f"Mean Total Degree = {tot_deg.mean():.2f}")

    ax_deg.set_title("C1. Node Degree Distribution\n(Linear Chains + Directional Drift)",
                     fontsize=9.5, fontweight="bold", pad=6)
    ax_deg.set_xlabel("Total Degree (In + Out)", fontsize=8.5)
    ax_deg.set_ylabel("Node Frequency", fontsize=8.5)
    ax_deg.tick_params(labelsize=7.5)
    ax_deg.grid(True, linestyle=":", alpha=0.5)
    ax_deg.legend(loc="upper left", fontsize=7.5)

    # -------------------------------------------------------------------------
    # Panel 8: Edge Distance & Bathymetric Gradient Distribution (Row 2, Col 2)
    # -------------------------------------------------------------------------
    ax_attr = fig.add_subplot(gs[2, 2])
    edge_attr = pyg_data.edge_attr.numpy()
    dists = edge_attr[:, 0]
    grads = edge_attr[:, 1]

    ax_attr2 = ax_attr.twinx()

    # Histogram of edge Euclidean distances
    n1, bins1, _ = ax_attr.hist(dists, bins=25, color="#1976d2", alpha=0.6,
                                label="Edge Distance (m)", edgecolor="none")
    # KDE / line of bathymetric gradients on twin axis
    n2, bins2, _ = ax_attr2.hist(grads * 1000.0, bins=25, color="#f57c00", alpha=0.5,
                                 label="Bathymetric Gradient (‰)", edgecolor="none")

    ax_attr.set_title("C2. Edge Attribute Distributions\n(Euclidean Distance & Bathymetry Slope)",
                      fontsize=9.5, fontweight="bold", pad=6)
    ax_attr.set_xlabel("Edge Distance (m)", fontsize=8.5)
    ax_attr.set_ylabel("Count (Distance)", fontsize=8.5, color="#1976d2")
    ax_attr2.set_ylabel("Count (Gradient ‰)", fontsize=8.5, color="#f57c00")
    ax_attr.tick_params(axis="y", labelcolor="#1976d2", labelsize=7.5)
    ax_attr2.tick_params(axis="y", labelcolor="#f57c00", labelsize=7.5)
    ax_attr.grid(True, linestyle=":", alpha=0.4)

    # Unified Legend for the overall figure
    legend_elements = [
        Line2D([0], [0], color="#78909c", lw=2, label="Official Coastline (10m)"),
        Line2D([0], [0], color="#37474f", lw=1.2, label="Adjacent Chain Edge (100m)"),
        Line2D([0], [0], color="#d32f2f", marker=">", lw=1.5, label="Directional Longshore Drift Edge (200m)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00e676", markeredgecolor="k", markersize=8, label="South Transect (Start)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d50000", markeredgecolor="k", markersize=8, label="North Transect (End)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995), frameon=True, facecolor="white", edgecolor="#cfd8dc")

    # Save to disk
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log.info("Saved graph visualization figure to: %s (DPI: %d)", output_png_path, dpi)


def main():
    parser = argparse.ArgumentParser(description="Visualize PyTorch Geometric coastal graph topology")
    parser.add_argument("--graph-pt", type=Path, default=DEFAULT_GRAPH_PT)
    parser.add_argument("--coastline-geojson", type=Path, default=DEFAULT_COASTLINE_GEOJSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    render_graph_visualization(
        graph_pt_path=args.graph_pt,
        coastline_geojson_path=args.coastline_geojson,
        output_png_path=args.output_png,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
