import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller.connection_graph import ConnectionGraph
import matplotlib.pyplot as plt  # Optional for visualization
import networkx as nx

NUM_AGENTS = 30

def test_graph_construction(num_agents=NUM_AGENTS, seed=42, graph_type="erdos-renyi"):
    # Create num agents agents
    agents = [f"agent_{i}" for i in range(num_agents)]
    
    # Build community graph
    cg = ConnectionGraph(
        agents,
        seed=seed,
        graph_type=graph_type,
        base_in_social_needs=0.5,
        base_out_social_needs=2.0,
        lognormal_mu=1.0,
        lognormal_sigma=1.0,
        activity_lognormal_mu=1.0,
        activity_lognormal_sigma=1.8,
        group_divisor=5,
        within_group_fraction=0.85,
        cross_group_divisor=5.0,
        reciprocity_within=0.4,
        reciprocity_cross=0.05,
        reciprocity_influencer=0.0,
        influencer_quantile=0.9,
        min_out_degree=2,
        min_in_degree=1,
    )
    
    graph = cg.get_graph()  # out-neighbors
    in_weights = getattr(cg, "node_in_weights", {})
    out_weights = getattr(cg, "node_out_weights", {})
    group_of = getattr(cg, "_group_of", {})
    
    # Console output
    print(f"=== Community Graph Test ({NUM_AGENTS} Agents) ===")
    print(f"Total agents: {len(agents)}")
    print(f"Groups: {len(set(group_of.values()))} (divisor={cg.group_divisor})")
    
    # Group sizes
    from collections import Counter
    group_counts = Counter(group_of.values())
    print(f"Group sizes: {dict(group_counts)}")
    
    # Degrees (directed)
    out_degrees = {a: len(graph[a]) for a in agents}
    in_degrees = {a: 0 for a in agents}
    for src, nbrs in graph.items():
        for dst in nbrs:
            if dst in in_degrees:
                in_degrees[dst] += 1
    avg_out = sum(out_degrees.values()) / len(out_degrees)
    avg_in = sum(in_degrees.values()) / len(in_degrees)
    print(f"Avg out-degree: {avg_out:.2f}")
    print(f"Avg in-degree:  {avg_in:.2f}")
    print(f"Out-degree range: {min(out_degrees.values())} - {max(out_degrees.values())}")
    print(f"In-degree range:  {min(in_degrees.values())} - {max(in_degrees.values())}")
    
    # Within/cross edges + reciprocity
    within_edges = 0
    cross_edges = 0
    reciprocal_edges = 0
    total_edges = 0
    for a, nbrs in graph.items():
        for b in nbrs:
            total_edges += 1
            if group_of.get(a) == group_of.get(b):
                within_edges += 1
            else:
                cross_edges += 1
            if a in graph.get(b, []):
                reciprocal_edges += 1
    within_fraction = within_edges / total_edges if total_edges else 0
    reciprocity = reciprocal_edges / total_edges if total_edges else 0
    print(
        f"Edges: {total_edges} total, {within_edges} within-group ({within_fraction:.2%}), {cross_edges} cross-group"
    )
    print(f"Reciprocity (edge-level): {reciprocity:.2%}")
    
    # Influence-graph semantics:
    #   out-degree = how many others this node influences (reach)
    #   in-degree  = how many inputs this node receives (feed size)
    top_by_outdeg = sorted(agents, key=lambda a: (out_degrees[a], out_weights.get(a, 0.0)), reverse=True)[:5]
    top_by_indeg = sorted(agents, key=lambda a: (in_degrees[a], in_weights.get(a, 0.0)), reverse=True)[:5]

    print("Top 5 influencers (by out-degree):")
    for i, agent in enumerate(top_by_outdeg):
        print(
            f"  {i+1}. {agent}: out={out_degrees[agent]}, out_w={out_weights.get(agent, 0.0):.2f}, in={in_degrees[agent]}, in_w={in_weights.get(agent, 0.0):.2f}"
        )

    print("Top 5 biggest feeds (by in-degree):")
    for i, agent in enumerate(top_by_indeg):
        print(
            f"  {i+1}. {agent}: in={in_degrees[agent]}, in_w={in_weights.get(agent, 0.0):.2f}, out={out_degrees[agent]}, out_w={out_weights.get(agent, 0.0):.2f}"
        )
    
    # Visualize
    visualize_graph(cg, agents, group_of)


def test_erdos_renyi_graph_construction():
    agents = [f"agent_{i}" for i in range(NUM_AGENTS)]

    cg = ConnectionGraph(
        agents,
        seed=42,
        graph_type="erdos-renyi",
        avg_degree_target=6.0,
    )

    graph = cg.get_graph()
    nx_graph = cg.to_networkx()

    out_degrees = {a: len(graph[a]) for a in agents}
    in_degrees = {a: 0 for a in agents}
    for src, nbrs in graph.items():
        for dst in nbrs:
            if dst in in_degrees:
                in_degrees[dst] += 1

    avg_out = sum(out_degrees.values()) / len(out_degrees)
    avg_in = sum(in_degrees.values()) / len(in_degrees)

    assert nx.is_strongly_connected(nx_graph)
    assert 5.0 <= avg_out <= 7.0
    assert abs(avg_out - avg_in) < 1e-9
    assert all(deg >= 1 for deg in out_degrees.values())

    print(f"=== Erd\"os-R\"enyi Graph Test ({NUM_AGENTS} Agents) ===")
    print(f"Avg out-degree: {avg_out:.2f}")
    print(f"Avg in-degree:  {avg_in:.2f}")
    print(f"Out-degree range: {min(out_degrees.values())} - {max(out_degrees.values())}")
    print(f"In-degree range:  {min(in_degrees.values())} - {max(in_degrees.values())}")

def visualize_graph(cg, agents, group_of, save_path=None, show=True):
    # Convert to NetworkX
    G = cg.to_networkx()
    
    # Color nodes by group
    group_colors = {}
    color_map = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for agent, group in group_of.items():
        group_colors[agent] = color_map[group % len(color_map)]
    
    node_colors = [group_colors.get(node, 'black') for node in G.nodes()]
    
    # Node sizes by in-degree (influence)
    out_degrees = dict(G.out_degree()) if G.is_directed() else dict(G.degree())
    node_sizes = [150 + 60 * out_degrees.get(node, 0) for node in G.nodes()]
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    draw_labels = len(G.nodes()) <= 10

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='gray',
        alpha=0.6,
        arrows=G.is_directed(),
        arrowstyle='-|>',
        arrowsize=14,
        connectionstyle='arc3,rad=0.05',
        width=1.0,
    )
    if draw_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Legend for groups
    unique_groups = sorted(set(group_of.values()))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[g % len(color_map)], 
                                  markersize=10, label=f'Group {g}') for g in unique_groups]
    plt.legend(handles=legend_elements, loc='upper right')
    
    title = (
        f"{'Directed ' if G.is_directed() else ''}Community Graph Visualization ({len(G.nodes())} Agents)\n"
        "Node size = in-degree (influence), Color = group"
    )
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def test_and_save_grpah():
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    for agent_size in [30, 60]:
        agents = [f"agent_{i}" for i in range(agent_size)]
        for graph_type in ["community", "erdos-renyi", "chung-lu"]:
            for i in range(5):
                seed = 42 + i
                cg = ConnectionGraph(
                    agents,
                    seed=seed,
                    graph_type=graph_type,
                    avg_degree_target=6.0,
                    base_in_social_needs=0.5,
                    base_out_social_needs=2.0,
                    lognormal_mu=1.0,
                    lognormal_sigma=1.0,
                    activity_lognormal_mu=1.0,
                    activity_lognormal_sigma=1.8,
                    group_divisor=5,
                    within_group_fraction=0.85,
                    cross_group_divisor=5.0,
                    reciprocity_within=0.4,
                    reciprocity_cross=0.05,
                    reciprocity_influencer=0.0,
                    influencer_quantile=0.9,
                    min_out_degree=2,
                    min_in_degree=1,
                )

                group_of = getattr(cg, "_group_of", {})

                if graph_type == "community":
                    fname = f"SBM_{agent_size}_run_{i+1}.png"
                elif graph_type == "erdos-renyi":
                    fname = f"erdos_renyi_{agent_size}_run_{i+1}.png"
                else:
                    fname = f"chung_lu_{agent_size}_run_{i+1}.png"

                save_path = os.path.join(plots_dir, fname)
                visualize_graph(cg, agents, group_of, save_path=save_path, show=False)
                print(f"Saved graph plot: {save_path}")

if __name__ == "__main__":
    test_graph_construction()
    test_erdos_renyi_graph_construction()
    test_and_save_grpah()