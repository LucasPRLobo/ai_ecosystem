"""
Main application window for Prompted Town GUI.

Run with:
    python -m prompted_town.gui
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import queue
import os
from typing import Optional
from dataclasses import dataclass

from ..core.types import Location, TimeOfDay, AgentRole, ConversationIntent
from ..core.agent_spec import AgentSpec
from ..rl import (
    EnvConfig,
    PromptedTownEnv,
    QLearningConfig,
    MultiAgentQLearning,
)
from ..simulation import conversation_action
from ..ai import create_backend, ConversationEngine


# =============================================================================
# THEME & COLORS
# =============================================================================

class Theme:
    """Application color theme."""
    # Main colors
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    ACCENT = "#e94560"
    ACCENT_LIGHT = "#ff6b6b"
    TEXT = "#eaeaea"
    TEXT_DIM = "#a0a0a0"

    # Role colors
    FARMER = "#4ade80"
    GUARD = "#60a5fa"
    REBEL = "#f87171"

    # Status colors
    SUCCESS = "#4ade80"
    WARNING = "#fbbf24"
    ERROR = "#f87171"
    INFO = "#60a5fa"

    # Time of day colors
    DAWN = "#fcd34d"
    DAY = "#fef3c7"
    DUSK = "#fb923c"
    NIGHT = "#1e3a5f"


LOCATION_EMOJI = {
    Location.FARM: "🌾",
    Location.MARKET: "🏪",
    Location.TAVERN: "🍺",
    Location.HOME: "🏠",
    Location.GATE: "🚪",
    Location.TOWN_SQUARE: "⛲",
}

ROLE_COLORS = {
    "farmer": Theme.FARMER,
    "guard": Theme.GUARD,
    "rebel": Theme.REBEL,
}

TIME_COLORS = {
    TimeOfDay.DAWN: Theme.DAWN,
    TimeOfDay.DAY: Theme.DAY,
    TimeOfDay.DUSK: Theme.DUSK,
    TimeOfDay.NIGHT: Theme.NIGHT,
}


# =============================================================================
# CUSTOM STYLED WIDGETS
# =============================================================================

class StyledFrame(tk.Frame):
    """A styled frame with rounded appearance."""
    def __init__(self, parent, **kwargs):
        bg = kwargs.pop('bg', Theme.BG_MEDIUM)
        super().__init__(parent, bg=bg, **kwargs)


class StyledLabel(tk.Label):
    """A styled label."""
    def __init__(self, parent, **kwargs):
        bg = kwargs.pop('bg', Theme.BG_MEDIUM)
        fg = kwargs.pop('fg', Theme.TEXT)
        super().__init__(parent, bg=bg, fg=fg, **kwargs)


class StyledButton(tk.Button):
    """A styled button with hover effects."""
    def __init__(self, parent, **kwargs):
        bg = kwargs.pop('bg', Theme.BG_LIGHT)
        fg = kwargs.pop('fg', Theme.TEXT)
        activebackground = kwargs.pop('activebackground', Theme.ACCENT)
        super().__init__(
            parent, bg=bg, fg=fg,
            activebackground=activebackground,
            activeforeground=Theme.TEXT,
            relief=tk.FLAT,
            cursor="hand2",
            padx=15, pady=8,
            **kwargs
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._default_bg = bg

    def _on_enter(self, e):
        self.config(bg=Theme.ACCENT)

    def _on_leave(self, e):
        self.config(bg=self._default_bg)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class PromptedTownApp:
    """Main GUI application for Prompted Town."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Prompted Town")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=Theme.BG_DARK)

        # Simulation state
        self.env: Optional[PromptedTownEnv] = None
        self.agents: Optional[MultiAgentQLearning] = None
        self.is_running = False
        self.step_delay = 800  # ms between steps
        self.use_ai = False
        self.ai_initialized = False

        # Message queue for thread-safe updates
        self.message_queue = queue.Queue()

        # Configure fonts
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.header_font = font.Font(family="Helvetica", size=11, weight="bold")
        self.body_font = font.Font(family="Consolas", size=10)
        self.small_font = font.Font(family="Helvetica", size=9)

        # Build UI
        self._create_header()
        self._create_main_layout()
        self._create_status_bar()

        # Initialize simulation
        self._init_simulation()

        # Start message processor
        self._process_messages()

    # =========================================================================
    # UI CREATION
    # =========================================================================

    def _create_header(self):
        """Create header bar."""
        header = StyledFrame(self.root, bg=Theme.BG_DARK, height=60)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        header.pack_propagate(False)

        # Title
        title = StyledLabel(
            header, text="⚔️ Prompted Town",
            font=self.title_font, bg=Theme.BG_DARK, fg=Theme.ACCENT
        )
        title.pack(side=tk.LEFT, padx=10)

        # Subtitle
        subtitle = StyledLabel(
            header, text="Multi-Agent RL Simulation with AI Personalities",
            font=self.small_font, bg=Theme.BG_DARK, fg=Theme.TEXT_DIM
        )
        subtitle.pack(side=tk.LEFT, padx=5)

        # Right side controls
        controls = StyledFrame(header, bg=Theme.BG_DARK)
        controls.pack(side=tk.RIGHT, padx=10)

        # AI toggle with indicator
        self.ai_indicator = StyledLabel(
            controls, text="●", fg=Theme.TEXT_DIM, bg=Theme.BG_DARK,
            font=("Helvetica", 16)
        )
        self.ai_indicator.pack(side=tk.LEFT, padx=5)

        self.ai_var = tk.BooleanVar(value=False)
        ai_check = tk.Checkbutton(
            controls, text="AI Conversations", variable=self.ai_var,
            command=self._toggle_ai, bg=Theme.BG_DARK, fg=Theme.TEXT,
            selectcolor=Theme.BG_MEDIUM, activebackground=Theme.BG_DARK,
            activeforeground=Theme.TEXT, font=self.small_font
        )
        ai_check.pack(side=tk.LEFT, padx=5)

    def _create_main_layout(self):
        """Create main layout with panels."""
        main_frame = StyledFrame(self.root, bg=Theme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left column - World view
        left_col = StyledFrame(main_frame, bg=Theme.BG_DARK, width=500)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_world_panel(left_col)
        self._create_control_panel(left_col)

        # Right column - Info panels
        right_col = StyledFrame(main_frame, bg=Theme.BG_DARK, width=500)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_agent_panel(right_col)
        self._create_conversation_panel(right_col)

    def _create_world_panel(self, parent):
        """Create world view panel."""
        panel = StyledFrame(parent, bg=Theme.BG_MEDIUM)
        panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Panel header
        header = StyledFrame(panel, bg=Theme.BG_LIGHT, height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        StyledLabel(
            header, text="🗺️ World Map", font=self.header_font, bg=Theme.BG_LIGHT
        ).pack(side=tk.LEFT, padx=15, pady=8)

        # Time/day indicator
        self.time_frame = StyledFrame(header, bg=Theme.BG_LIGHT)
        self.time_frame.pack(side=tk.RIGHT, padx=15)

        self.day_label = StyledLabel(
            self.time_frame, text="Day 0", font=self.small_font, bg=Theme.BG_LIGHT
        )
        self.day_label.pack(side=tk.LEFT, padx=5)

        self.time_label = StyledLabel(
            self.time_frame, text="🌅 DAWN", font=self.small_font, bg=Theme.BG_LIGHT
        )
        self.time_label.pack(side=tk.LEFT, padx=5)

        # Canvas
        canvas_frame = StyledFrame(panel, bg=Theme.BG_MEDIUM)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame, bg=Theme.BG_MEDIUM, highlightthickness=0,
            width=480, height=320
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Stats bar
        stats_bar = StyledFrame(panel, bg=Theme.BG_LIGHT, height=50)
        stats_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        stats_bar.pack_propagate(False)

        stats_inner = StyledFrame(stats_bar, bg=Theme.BG_LIGHT)
        stats_inner.pack(expand=True)

        # Stats items
        self._create_stat_item(stats_inner, "tick_label", "⏱️ Tick", "0")
        self._create_stat_item(stats_inner, "recruited_label", "🤝 Recruited", "0")
        self._create_stat_item(stats_inner, "arrests_label", "⛓️ Arrests", "0")
        self._create_stat_item(stats_inner, "convs_label", "💬 Convs", "0")

    def _create_stat_item(self, parent, attr_name, label, value):
        """Create a stat display item."""
        frame = StyledFrame(parent, bg=Theme.BG_LIGHT)
        frame.pack(side=tk.LEFT, padx=20, pady=10)

        StyledLabel(
            frame, text=label, font=self.small_font,
            bg=Theme.BG_LIGHT, fg=Theme.TEXT_DIM
        ).pack()

        val_label = StyledLabel(
            frame, text=value, font=self.header_font, bg=Theme.BG_LIGHT
        )
        val_label.pack()
        setattr(self, attr_name, val_label)

    def _create_control_panel(self, parent):
        """Create control panel."""
        panel = StyledFrame(parent, bg=Theme.BG_MEDIUM, height=100)
        panel.pack(fill=tk.X)
        panel.pack_propagate(False)

        # Panel header
        header = StyledFrame(panel, bg=Theme.BG_LIGHT, height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        StyledLabel(
            header, text="🎮 Controls", font=self.header_font, bg=Theme.BG_LIGHT
        ).pack(side=tk.LEFT, padx=15, pady=5)

        # Controls
        controls = StyledFrame(panel, bg=Theme.BG_MEDIUM)
        controls.pack(fill=tk.X, padx=15, pady=10)

        # Buttons
        btn_frame = StyledFrame(controls, bg=Theme.BG_MEDIUM)
        btn_frame.pack(side=tk.LEFT)

        self.step_btn = StyledButton(btn_frame, text="⏭️ Step", command=self._step_once)
        self.step_btn.pack(side=tk.LEFT, padx=3)

        self.run_btn = StyledButton(btn_frame, text="▶️ Run", command=self._start_running)
        self.run_btn.pack(side=tk.LEFT, padx=3)

        self.pause_btn = StyledButton(btn_frame, text="⏸️ Pause", command=self._stop_running)
        self.pause_btn.pack(side=tk.LEFT, padx=3)
        self.pause_btn.config(state=tk.DISABLED)

        self.reset_btn = StyledButton(btn_frame, text="🔄 Reset", command=self._reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=3)

        # Force conversation button (for testing AI)
        self.conv_btn = StyledButton(
            btn_frame, text="💬 Force Conv", command=self._force_conversation,
            bg=Theme.ACCENT
        )
        self.conv_btn.pack(side=tk.LEFT, padx=3)

        # Speed slider
        speed_frame = StyledFrame(controls, bg=Theme.BG_MEDIUM)
        speed_frame.pack(side=tk.RIGHT)

        StyledLabel(
            speed_frame, text="Speed:", font=self.small_font, bg=Theme.BG_MEDIUM
        ).pack(side=tk.LEFT, padx=5)

        self.speed_var = tk.IntVar(value=800)
        speed_slider = tk.Scale(
            speed_frame, from_=100, to=2000, orient=tk.HORIZONTAL,
            variable=self.speed_var, command=self._update_speed,
            bg=Theme.BG_MEDIUM, fg=Theme.TEXT, highlightthickness=0,
            troughcolor=Theme.BG_LIGHT, length=150
        )
        speed_slider.pack(side=tk.LEFT)

    def _create_agent_panel(self, parent):
        """Create agent list panel."""
        panel = StyledFrame(parent, bg=Theme.BG_MEDIUM, height=250)
        panel.pack(fill=tk.X, pady=(0, 10))
        panel.pack_propagate(False)

        # Panel header
        header = StyledFrame(panel, bg=Theme.BG_LIGHT, height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        StyledLabel(
            header, text="👥 Agents", font=self.header_font, bg=Theme.BG_LIGHT
        ).pack(side=tk.LEFT, padx=15, pady=5)

        # Agent list frame
        list_frame = StyledFrame(panel, bg=Theme.BG_MEDIUM)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Custom agent cards instead of treeview
        self.agent_cards_frame = StyledFrame(list_frame, bg=Theme.BG_MEDIUM)
        self.agent_cards_frame.pack(fill=tk.BOTH, expand=True)

        self.agent_cards = {}

    def _create_conversation_panel(self, parent):
        """Create conversation log panel."""
        panel = StyledFrame(parent, bg=Theme.BG_MEDIUM)
        panel.pack(fill=tk.BOTH, expand=True)

        # Panel header
        header = StyledFrame(panel, bg=Theme.BG_LIGHT, height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        StyledLabel(
            header, text="💬 Conversation Log", font=self.header_font, bg=Theme.BG_LIGHT
        ).pack(side=tk.LEFT, padx=15, pady=5)

        # Clear button
        clear_btn = tk.Button(
            header, text="Clear", command=self._clear_log,
            bg=Theme.BG_LIGHT, fg=Theme.TEXT_DIM, relief=tk.FLAT,
            font=self.small_font, cursor="hand2"
        )
        clear_btn.pack(side=tk.RIGHT, padx=15)

        # Log text area
        log_frame = StyledFrame(panel, bg=Theme.BG_MEDIUM)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.conv_log = tk.Text(
            log_frame, wrap=tk.WORD, font=self.body_font,
            bg=Theme.BG_DARK, fg=Theme.TEXT,
            insertbackground=Theme.TEXT, relief=tk.FLAT,
            padx=10, pady=10
        )

        scrollbar = tk.Scrollbar(log_frame, command=self.conv_log.yview)
        self.conv_log.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.conv_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.conv_log.config(state=tk.DISABLED)

        # Configure tags
        self.conv_log.tag_configure("header", foreground=Theme.ACCENT, font=self.header_font)
        self.conv_log.tag_configure("speaker_farmer", foreground=Theme.FARMER)
        self.conv_log.tag_configure("speaker_guard", foreground=Theme.GUARD)
        self.conv_log.tag_configure("speaker_rebel", foreground=Theme.REBEL)
        self.conv_log.tag_configure("outcome_good", foreground=Theme.SUCCESS)
        self.conv_log.tag_configure("outcome_bad", foreground=Theme.ERROR)
        self.conv_log.tag_configure("system", foreground=Theme.TEXT_DIM, font=self.small_font)
        self.conv_log.tag_configure("dialogue", foreground=Theme.TEXT)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = StyledLabel(
            self.root, textvariable=self.status_var,
            font=self.small_font, bg=Theme.BG_DARK, fg=Theme.TEXT_DIM,
            anchor=tk.W, padx=15
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    # =========================================================================
    # SIMULATION MANAGEMENT
    # =========================================================================

    def _init_simulation(self):
        """Initialize the simulation."""
        # Check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        env_config = EnvConfig(
            num_farmers=2,
            num_guards=1,
            num_rebels=1,
            max_ticks=200,
            max_days=14,
            use_ai_conversations=False,
            random_seed=None,  # Random each time
        )
        self.env = PromptedTownEnv(env_config)

        q_config = QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            num_actions=self.env.get_action_space_size(),
        )
        self.agents = MultiAgentQLearning(self.env.agent_ids, q_config)

        self.env.reset()
        self.total_arrests = 0
        self.total_conversations = 0

        self._create_agent_cards()
        self._update_display()
        self._log_system("Simulation initialized with 4 agents")

        if api_key:
            self._log_system("Anthropic API key detected - AI conversations available")
        else:
            self._log_system("No API key found - using rule-based conversations")

    def _create_agent_cards(self):
        """Create agent card widgets."""
        # Clear existing
        for widget in self.agent_cards_frame.winfo_children():
            widget.destroy()
        self.agent_cards = {}

        for i, aid in enumerate(self.env.agent_ids):
            spec = self.env.agent_specs.get(aid)
            if not spec:
                continue

            role = spec.public_role.value
            color = ROLE_COLORS.get(role, Theme.TEXT)

            # Card frame
            card = StyledFrame(self.agent_cards_frame, bg=Theme.BG_DARK)
            card.pack(fill=tk.X, pady=2)

            # Color indicator
            indicator = tk.Canvas(card, width=4, height=40, bg=color, highlightthickness=0)
            indicator.pack(side=tk.LEFT)

            # Info
            info = StyledFrame(card, bg=Theme.BG_DARK)
            info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

            # Name and role
            name_label = StyledLabel(
                info, text=f"{spec.name}", font=self.header_font,
                bg=Theme.BG_DARK, fg=color
            )
            name_label.pack(anchor=tk.W)

            # Details
            details_label = StyledLabel(
                info, text="Loading...", font=self.small_font,
                bg=Theme.BG_DARK, fg=Theme.TEXT_DIM
            )
            details_label.pack(anchor=tk.W)

            # Status badge
            status_label = StyledLabel(
                card, text="●", font=self.small_font,
                bg=Theme.BG_DARK, fg=Theme.SUCCESS
            )
            status_label.pack(side=tk.RIGHT, padx=10)

            self.agent_cards[aid] = {
                'card': card,
                'name': name_label,
                'details': details_label,
                'status': status_label,
            }

    def _new_simulation(self):
        """Create new simulation."""
        self._stop_running()
        self._init_simulation()

    def _reset_simulation(self):
        """Reset current simulation."""
        self._stop_running()
        if self.env:
            self.env.reset()
            self.total_arrests = 0
            self.total_conversations = 0
            self._update_display()
            self._clear_log()
            self._log_system("Simulation reset")

    def _step_once(self):
        """Execute one simulation step."""
        if not self.env or not self.agents:
            return

        # Get observations
        observations = self.env._get_observations()

        # Select actions
        action_indices = self.agents.select_actions(observations, explore=True)
        actions = {
            aid: self.env._index_to_action(aid, idx)
            for aid, idx in action_indices.items()
        }

        # Step
        next_obs, rewards, done, info = self.env.step(actions)

        # Update Q-values
        self.agents.update_all(
            observations, action_indices, rewards, next_obs, done
        )

        # Track stats
        self.total_arrests += len(info.arrests)
        self.total_conversations += len(info.conversations)

        # Log conversations
        for conv in info.conversations:
            self._log_conversation(conv)

        # Log arrests
        for arrested in info.arrests:
            self._log_system(f"⛓️ ARREST: {arrested} was arrested!")

        # Update display
        self._update_display()

        if done:
            self._stop_running()
            self._log_system("🏁 Episode ended!")
            self.status_var.set("Episode ended")

    def _force_conversation(self):
        """Force a conversation between two nearby agents (for testing AI)."""
        if not self.env:
            return

        state = self.env.state

        # Find rebel and a nearby farmer
        rebel_id = None
        target_id = None

        for aid, spec in self.env.agent_specs.items():
            if spec.true_faction.value == "rebel":
                rebel_id = aid
                break

        if rebel_id:
            nearby = state.get_nearby_agents(rebel_id)
            for nid in nearby:
                spec = self.env.agent_specs.get(nid)
                if spec and spec.public_role.value == "farmer":
                    target_id = nid
                    break

        if not rebel_id or not target_id:
            # Just pick any two agents at same location
            agents_by_loc = {}
            for aid, agent in state.agents.items():
                if agent.can_act():
                    loc = agent.location
                    if loc not in agents_by_loc:
                        agents_by_loc[loc] = []
                    agents_by_loc[loc].append(aid)

            for loc, agents in agents_by_loc.items():
                if len(agents) >= 2:
                    rebel_id = agents[0]
                    target_id = agents[1]
                    break

        if rebel_id and target_id:
            # Create conversation action
            intent = ConversationIntent.RECRUIT if "rebel" in rebel_id else ConversationIntent.CASUAL
            action = conversation_action(rebel_id, target_id, intent)

            # Step with this forced action
            actions = {aid: self.env._index_to_action(aid, 0) for aid in self.env.agent_ids}
            actions[rebel_id] = action

            next_obs, rewards, done, info = self.env.step(actions)

            self.total_conversations += len(info.conversations)

            for conv in info.conversations:
                self._log_conversation(conv)

            self._update_display()

            if not info.conversations:
                self._log_system("No conversation occurred - agents may be too far apart")
        else:
            self._log_system("Could not find two agents at same location")

    def _start_running(self):
        """Start continuous simulation."""
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.DISABLED)
        self.conv_btn.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self._run_step()

    def _stop_running(self):
        """Stop continuous simulation."""
        self.is_running = False
        self.run_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.step_btn.config(state=tk.NORMAL)
        self.conv_btn.config(state=tk.NORMAL)
        self.status_var.set("Paused")

    def _run_step(self):
        """Run one step and schedule next."""
        if self.is_running:
            self._step_once()
            self.root.after(self.step_delay, self._run_step)

    def _update_speed(self, value):
        """Update simulation speed."""
        self.step_delay = int(float(value))

    def _toggle_ai(self):
        """Toggle AI conversations."""
        self.use_ai = self.ai_var.get()

        if self.use_ai:
            if not self.ai_initialized:
                try:
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if api_key:
                        self._log_system("Initializing Anthropic AI backend...")
                        self.env.ai_backend = create_backend(
                            "anthropic",
                            api_key=api_key,
                            model="claude-3-5-haiku-latest"
                        )
                        self.env.conversation_engine = ConversationEngine(self.env.ai_backend)
                        self.env.config.use_ai_conversations = True
                        self.ai_initialized = True
                        self.ai_indicator.config(fg=Theme.SUCCESS)
                        self._log_system("✅ AI conversations enabled!")
                    else:
                        messagebox.showwarning(
                            "API Key Missing",
                            "Set ANTHROPIC_API_KEY environment variable to use AI conversations.\n\n"
                            "Example:\nexport ANTHROPIC_API_KEY=your-key-here"
                        )
                        self.ai_var.set(False)
                        self.use_ai = False
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize AI: {e}")
                    self._log_system(f"❌ AI initialization failed: {e}")
                    self.ai_var.set(False)
                    self.use_ai = False
            else:
                self.env.config.use_ai_conversations = True
                self.ai_indicator.config(fg=Theme.SUCCESS)
                self._log_system("✅ AI conversations enabled")
        else:
            if self.env:
                self.env.config.use_ai_conversations = False
            self.ai_indicator.config(fg=Theme.TEXT_DIM)
            self._log_system("AI conversations disabled")

    # =========================================================================
    # DISPLAY UPDATES
    # =========================================================================

    def _update_display(self):
        """Update all display elements."""
        if not self.env or not self.env.state:
            return

        self._update_canvas()
        self._update_stats()
        self._update_agent_cards()

    def _update_canvas(self):
        """Update the world canvas."""
        self.canvas.delete("all")
        state = self.env.state

        # Get canvas size
        w = self.canvas.winfo_width() or 480
        h = self.canvas.winfo_height() or 320

        # Calculate grid positions
        cols = 3
        rows = 2
        cell_w = w // cols
        cell_h = h // rows
        padding = 10

        locations = list(Location)

        for i, loc in enumerate(locations):
            col = i % cols
            row = i // cols

            x = col * cell_w + padding
            y = row * cell_h + padding
            box_w = cell_w - padding * 2
            box_h = cell_h - padding * 2

            # Background based on time
            time_color = TIME_COLORS.get(state.time_of_day, Theme.BG_LIGHT)

            # Location box
            self.canvas.create_rectangle(
                x, y, x + box_w, y + box_h,
                fill=Theme.BG_DARK, outline=Theme.BG_LIGHT, width=2
            )

            # Location header
            emoji = LOCATION_EMOJI.get(loc, "📍")
            self.canvas.create_text(
                x + box_w // 2, y + 15,
                text=f"{emoji} {loc.value.upper()}",
                fill=Theme.TEXT, font=("Helvetica", 9, "bold")
            )

            # Draw agents at this location
            agents_here = [
                (aid, a) for aid, a in state.agents.items()
                if a.location == loc and a.can_act()
            ]

            for j, (aid, agent) in enumerate(agents_here):
                spec = self.env.agent_specs.get(aid)
                role = spec.public_role.value if spec else "?"
                color = ROLE_COLORS.get(role, Theme.TEXT)

                # Agent position within box
                ax = x + 25 + (j % 3) * 45
                ay = y + 45 + (j // 3) * 35

                # Agent circle with glow for recruited
                if agent.is_recruited:
                    self.canvas.create_oval(
                        ax - 18, ay - 18, ax + 18, ay + 18,
                        fill="", outline=Theme.REBEL, width=2
                    )

                self.canvas.create_oval(
                    ax - 14, ay - 14, ax + 14, ay + 14,
                    fill=color, outline=Theme.BG_DARK, width=2
                )

                # Agent initial
                initial = spec.name[0] if spec else "?"
                self.canvas.create_text(
                    ax, ay, text=initial, fill=Theme.BG_DARK,
                    font=("Helvetica", 11, "bold")
                )

    def _update_stats(self):
        """Update statistics display."""
        state = self.env.state

        # Time indicators
        time_emojis = {
            TimeOfDay.DAWN: "🌅",
            TimeOfDay.DAY: "☀️",
            TimeOfDay.DUSK: "🌆",
            TimeOfDay.NIGHT: "🌙",
        }

        self.day_label.config(text=f"Day {state.day}")
        self.time_label.config(
            text=f"{time_emojis.get(state.time_of_day, '')} {state.time_of_day.value.upper()}"
        )

        # Stats
        self.tick_label.config(text=str(state.tick))
        self.recruited_label.config(text=str(state.total_rebels_recruited))
        self.arrests_label.config(text=str(self.total_arrests))
        self.convs_label.config(text=str(self.total_conversations))

    def _update_agent_cards(self):
        """Update agent card displays."""
        state = self.env.state

        for aid, card_widgets in self.agent_cards.items():
            agent = state.get_agent(aid)
            spec = self.env.agent_specs.get(aid)

            if not agent or not spec:
                continue

            # Update details
            loc_emoji = LOCATION_EMOJI.get(agent.location, "📍")
            details = f"{loc_emoji} {agent.location.value} | ⚡{agent.energy} | 💰{agent.gold}"
            if agent.suspicion_level > 0.1:
                details += f" | ⚠️{agent.suspicion_level:.1f}"

            card_widgets['details'].config(text=details)

            # Update status
            if agent.is_arrested:
                card_widgets['status'].config(text="⛓️", fg=Theme.ERROR)
            elif agent.is_recruited:
                card_widgets['status'].config(text="🤝", fg=Theme.REBEL)
            elif agent.suspicion_level > 0.5:
                card_widgets['status'].config(text="⚠️", fg=Theme.WARNING)
            else:
                card_widgets['status'].config(text="●", fg=Theme.SUCCESS)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log_conversation(self, conv: dict):
        """Log a conversation."""
        self.conv_log.config(state=tk.NORMAL)

        # Get speaker info for coloring
        initiator = conv['initiator']
        target = conv['target']

        init_spec = self.env.agent_specs.get(initiator)
        targ_spec = self.env.agent_specs.get(target)

        init_name = init_spec.name if init_spec else initiator
        targ_name = targ_spec.name if targ_spec else target

        # Header
        intent_emoji = "🤝" if conv['intent'] == "recruit" else "💬"
        self.conv_log.insert(
            tk.END,
            f"\n{intent_emoji} {init_name} → {targ_name} ({conv['intent']})\n",
            "header"
        )

        # Outcome
        outcome = conv['outcome_summary']
        tag = "outcome_good" if "succeeded" in outcome or "friendly" in outcome else "outcome_bad"
        self.conv_log.insert(tk.END, f"   Result: {outcome}\n", tag)

        # Transcript
        transcript = conv.get('transcript', [])
        if transcript:
            self.conv_log.insert(tk.END, "\n", "dialogue")
            for turn in transcript[:8]:
                speaker_id = turn['speaker']
                text = turn['text']

                # Get speaker name and color tag
                speaker_spec = self.env.agent_specs.get(speaker_id)
                speaker_name = speaker_spec.name if speaker_spec else speaker_id
                role = speaker_spec.public_role.value if speaker_spec else "farmer"
                tag = f"speaker_{role}"

                self.conv_log.insert(tk.END, f"   {speaker_name}: ", tag)
                self.conv_log.insert(tk.END, f"{text}\n", "dialogue")

            if len(transcript) > 8:
                self.conv_log.insert(
                    tk.END,
                    f"   ... ({len(transcript) - 8} more turns)\n",
                    "system"
                )
        else:
            self.conv_log.insert(tk.END, "   (No transcript available)\n", "system")

        self.conv_log.insert(tk.END, "\n")
        self.conv_log.see(tk.END)
        self.conv_log.config(state=tk.DISABLED)

    def _log_system(self, message: str):
        """Log a system message."""
        self.conv_log.config(state=tk.NORMAL)
        self.conv_log.insert(tk.END, f"[SYS] {message}\n", "system")
        self.conv_log.see(tk.END)
        self.conv_log.config(state=tk.DISABLED)

    def _clear_log(self):
        """Clear the conversation log."""
        self.conv_log.config(state=tk.NORMAL)
        self.conv_log.delete(1.0, tk.END)
        self.conv_log.config(state=tk.DISABLED)

    # =========================================================================
    # MESSAGE PROCESSING
    # =========================================================================

    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg[0] == "log":
                    self._log_system(msg[1])
                elif msg[0] == "update":
                    self._update_display()
        except queue.Empty:
            pass
        self.root.after(100, self._process_messages)


# =============================================================================
# RUN FUNCTION
# =============================================================================

def run_app():
    """Run the GUI application."""
    root = tk.Tk()

    # Set icon if available
    try:
        root.iconname("Prompted Town")
    except:
        pass

    app = PromptedTownApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()
