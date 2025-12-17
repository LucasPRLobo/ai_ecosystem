"""
Entity interaction graph visualization for Prompted Town.

Tracks and visualizes interactions between agents:
- Conversations
- Recruitments
- Trust relationships
"""

from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from ..core.types import AgentRole
from ..core.world_state import WorldState
from ..core.agent_spec import AgentSpec


# =============================================================================
# INTERACTION TRACKING
# =============================================================================

@dataclass
class Interaction:
    """Record of a single interaction."""
    initiator: str
    target: str
    interaction_type: str  # "conversation", "recruitment", "arrest"
    success: bool = True
    tick: int = 0
    day: int = 0


@dataclass
class InteractionTracker:
    """
    Tracks interactions between agents over time.

    Use this to build up interaction data during training,
    then visualize with create_interaction_graph().
    """
    interactions: list[Interaction] = field(default_factory=list)
    conversation_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    recruitment_attempts: dict[tuple[str, str], int] = field(default_factory=dict)
    recruitment_successes: dict[tuple[str, str], int] = field(default_factory=dict)
    arrests: dict[tuple[str, str], int] = field(default_factory=dict)

    def record_conversation(
        self,
        initiator: str,
        target: str,
        tick: int = 0,
        day: int = 0,
    ):
        """Record a conversation between agents."""
        self.interactions.append(Interaction(
            initiator=initiator,
            target=target,
            interaction_type="conversation",
            success=True,
            tick=tick,
            day=day,
        ))

        key = (initiator, target)
        self.conversation_counts[key] = self.conversation_counts.get(key, 0) + 1

    def record_recruitment(
        self,
        initiator: str,
        target: str,
        success: bool,
        tick: int = 0,
        day: int = 0,
    ):
        """Record a recruitment attempt."""
        self.interactions.append(Interaction(
            initiator=initiator,
            target=target,
            interaction_type="recruitment",
            success=success,
            tick=tick,
            day=day,
        ))

        key = (initiator, target)
        self.recruitment_attempts[key] = self.recruitment_attempts.get(key, 0) + 1
        if success:
            self.recruitment_successes[key] = self.recruitment_successes.get(key, 0) + 1

    def record_arrest(
        self,
        guard: str,
        arrested: str,
        tick: int = 0,
        day: int = 0,
    ):
        """Record an arrest."""
        self.interactions.append(Interaction(
            initiator=guard,
            target=arrested,
            interaction_type="arrest",
            success=True,
            tick=tick,
            day=day,
        ))

        key = (guard, arrested)
        self.arrests[key] = self.arrests.get(key, 0) + 1

    def get_interaction_matrix(self, agent_ids: list[str]) -> np.ndarray:
        """Get a matrix of total interactions between agents."""
        n = len(agent_ids)
        matrix = np.zeros((n, n))
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

        for (a, b), count in self.conversation_counts.items():
            if a in id_to_idx and b in id_to_idx:
                matrix[id_to_idx[a], id_to_idx[b]] += count

        return matrix

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_interactions": len(self.interactions),
            "total_conversations": sum(self.conversation_counts.values()),
            "total_recruitment_attempts": sum(self.recruitment_attempts.values()),
            "successful_recruitments": sum(self.recruitment_successes.values()),
            "total_arrests": sum(self.arrests.values()),
        }


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_interaction_graph(
    tracker: InteractionTracker,
    agent_specs: dict[str, AgentSpec],
    include_conversations: bool = True,
    include_recruitments: bool = True,
    include_arrests: bool = True,
    min_interactions: int = 1,
) -> "nx.DiGraph":
    """
    Create a NetworkX directed graph from interactions.

    Args:
        tracker: Interaction tracker with recorded data
        agent_specs: Agent specifications for role information
        include_conversations: Include conversation edges
        include_recruitments: Include recruitment edges
        include_arrests: Include arrest edges
        min_interactions: Minimum interactions to create an edge

    Returns:
        NetworkX DiGraph
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph visualization. Install with: pip install networkx")

    G = nx.DiGraph()

    # Add nodes with attributes
    for agent_id, spec in agent_specs.items():
        G.add_node(
            agent_id,
            role=spec.public_role.value,
            faction=spec.true_faction.value,
            name=spec.name,
        )

    # Add conversation edges
    if include_conversations:
        for (a, b), count in tracker.conversation_counts.items():
            if count >= min_interactions:
                if G.has_edge(a, b):
                    G[a][b]['conversations'] = count
                else:
                    G.add_edge(a, b, conversations=count, weight=count)

    # Add recruitment edges
    if include_recruitments:
        for (a, b), count in tracker.recruitment_attempts.items():
            successes = tracker.recruitment_successes.get((a, b), 0)
            if count >= min_interactions:
                if G.has_edge(a, b):
                    G[a][b]['recruitment_attempts'] = count
                    G[a][b]['recruitment_successes'] = successes
                else:
                    G.add_edge(
                        a, b,
                        recruitment_attempts=count,
                        recruitment_successes=successes,
                        weight=count,
                    )

    # Add arrest edges
    if include_arrests:
        for (a, b), count in tracker.arrests.items():
            if count >= min_interactions:
                if G.has_edge(a, b):
                    G[a][b]['arrests'] = count
                else:
                    G.add_edge(a, b, arrests=count, weight=count)

    return G


# =============================================================================
# GRAPH VISUALIZATION
# =============================================================================

ROLE_COLORS = {
    "farmer": "#4CAF50",  # Green
    "guard": "#2196F3",   # Blue
    "rebel": "#F44336",   # Red (but appears as farmer publicly)
}

FACTION_COLORS = {
    "loyalist": "#2196F3",
    "rebel": "#F44336",
    "neutral": "#9E9E9E",
}


def plot_interaction_graph(
    G: "nx.DiGraph",
    figsize: tuple = (12, 10),
    show_weights: bool = True,
    color_by: str = "role",  # "role" or "faction"
    layout: str = "spring",  # "spring", "circular", "kamada_kawai"
    title: str = "Agent Interactions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the interaction graph.

    Args:
        G: NetworkX graph from create_interaction_graph
        figsize: Figure size
        show_weights: Show edge weights as labels
        color_by: Color nodes by "role" or "faction"
        layout: Graph layout algorithm
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required. Install with: pip install networkx")

    fig, ax = plt.subplots(figsize=figsize)

    # Get layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors based on role or faction
    if color_by == "faction":
        colors = FACTION_COLORS
        node_colors = [colors.get(G.nodes[n].get('faction', 'neutral'), '#9E9E9E') for n in G.nodes()]
    else:
        colors = ROLE_COLORS
        node_colors = [colors.get(G.nodes[n].get('role', 'farmer'), '#9E9E9E') for n in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=1500,
        alpha=0.9,
    )

    # Draw labels
    labels = {n: f"{G.nodes[n].get('name', n)}\n({n})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    # Separate edge types by color
    conversation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('conversations', 0) > 0]
    recruitment_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('recruitment_successes', 0) > 0]
    arrest_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('arrests', 0) > 0]

    # Draw edges with different colors
    if conversation_edges:
        conv_weights = [G[u][v].get('conversations', 1) for u, v in conversation_edges]
        max_w = max(conv_weights) if conv_weights else 1
        conv_widths = [1 + 3 * (w / max_w) for w in conv_weights]
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=conversation_edges,
            edge_color='#888888',
            width=conv_widths,
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
        )

    if recruitment_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=recruitment_edges,
            edge_color='#F44336',
            width=3,
            alpha=0.8,
            arrows=True,
            arrowsize=20,
            style='solid',
            connectionstyle="arc3,rad=0.15",
        )

    if arrest_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=arrest_edges,
            edge_color='#2196F3',
            width=3,
            alpha=0.8,
            arrows=True,
            arrowsize=20,
            style='dashed',
            connectionstyle="arc3,rad=0.2",
        )

    # Edge labels if requested
    if show_weights:
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            parts = []
            if d.get('conversations', 0) > 0:
                parts.append(f"c:{d['conversations']}")
            if d.get('recruitment_attempts', 0) > 0:
                parts.append(f"r:{d.get('recruitment_successes', 0)}/{d['recruitment_attempts']}")
            if d.get('arrests', 0) > 0:
                parts.append(f"a:{d['arrests']}")
            if parts:
                edge_labels[(u, v)] = '\n'.join(parts)

        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)

    # Legend
    legend_patches = []
    if color_by == "faction":
        for faction, color in FACTION_COLORS.items():
            legend_patches.append(mpatches.Patch(color=color, label=faction.title()))
    else:
        for role, color in ROLE_COLORS.items():
            legend_patches.append(mpatches.Patch(color=color, label=role.title()))

    # Edge type legend
    legend_patches.append(mpatches.Patch(color='#888888', label='Conversation'))
    legend_patches.append(mpatches.Patch(color='#F44336', label='Recruitment'))
    legend_patches.append(mpatches.Patch(color='#2196F3', label='Arrest'))

    ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_trust_heatmap(
    state: WorldState,
    agent_ids: list[str],
    figsize: tuple = (10, 8),
    title: str = "Trust Relationships",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap of trust relationships between agents.

    Args:
        state: World state with relationships
        agent_ids: List of agent IDs to include
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    n = len(agent_ids)
    trust_matrix = np.zeros((n, n))

    for i, a in enumerate(agent_ids):
        for j, b in enumerate(agent_ids):
            if i != j:
                trust_matrix[i, j] = state.get_trust(a, b)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(trust_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_ids, rotation=45, ha='right')
    ax.set_yticklabels(agent_ids)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Trust Level', rotation=270, labelpad=15)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = ax.text(j, i, f'{trust_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(trust_matrix[i, j]) > 0.5 else 'black')

    ax.set_xlabel('Target Agent')
    ax.set_ylabel('Source Agent')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
