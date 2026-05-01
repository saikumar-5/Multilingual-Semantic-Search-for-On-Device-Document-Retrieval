"""
Settings Panel - Clean sidebar with search mode and directory management.
"""

import customtkinter as ctk
from typing import Callable, List


class SettingsFrame(ctk.CTkFrame):
    """Sidebar settings panel."""

    def __init__(
        self,
        parent,
        indexed_dirs: List[str],
        on_add_directory: Callable,
        on_reindex: Callable,
        on_reset: Callable,
        on_mode_change: Callable,
        current_mode: str = "hybrid",
    ):
        super().__init__(parent, fg_color=("#151c2f", "#151c2f"), corner_radius=14)

        self.on_add_directory = on_add_directory
        self.on_reindex = on_reindex
        self.on_reset = on_reset
        self.on_mode_change = on_mode_change
        self.current_mode = current_mode
        self._switches = {}

        self._create_widgets(indexed_dirs, current_mode)

    def _create_widgets(self, indexed_dirs: List[str], current_mode: str):
        # ── Search Mode ─────────────────────────────────────
        mode_header = ctk.CTkLabel(
            self,
            text="Search Mode",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=("#d9e6ff", "#d9e6ff"),
        )
        mode_header.pack(pady=(16, 10), padx=18, anchor="w")

        self.mode_var = ctk.StringVar(value=current_mode)

        modes = [
            ("Hybrid", "hybrid", "Search mode for custom hybrid retrieval."),
            ("Keyword", "keyword", "Keyword search for exact matches."),
            ("Semantic", "semantic", "Semantic search for meaning-based matches."),
        ]

        for label, value, tooltip in modes:
            mode_row = ctk.CTkFrame(
                self,
                fg_color=("#1a2238", "#1a2238"),
                corner_radius=10,
                border_width=1,
                border_color=("#26345a", "#26345a"),
            )
            mode_row.pack(fill="x", padx=16, pady=5)

            text_col = ctk.CTkFrame(mode_row, fg_color="transparent")
            text_col.pack(side="left", fill="x", expand=True, padx=(10, 6), pady=8)

            title = ctk.CTkLabel(
                text_col,
                text=label,
                font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
                text_color=("#e3ecff", "#e3ecff"),
                anchor="w",
            )
            title.pack(fill="x")

            hint = ctk.CTkLabel(
                text_col,
                text=tooltip,
                font=ctk.CTkFont(size=11),
                text_color=("#91a0c6", "#91a0c6"),
                anchor="w",
            )
            hint.pack(fill="x")

            switch = ctk.CTkSwitch(
                mode_row,
                text="",
                width=40,
                command=lambda m=value: self._set_mode(m),
                onvalue=1,
                offvalue=0,
                button_color=("#cfe3ff", "#cfe3ff"),
                progress_color=("#2fb8ff", "#2fb8ff"),
            )
            switch.pack(side="right", padx=(6, 12))
            self._switches[value] = switch

        self._sync_mode_switches(current_mode)

        ctk.CTkFrame(self, height=1, fg_color=("#2b3658", "#2b3658")).pack(
            fill="x", padx=16, pady=14
        )

        # ── Indexed Directories ──────────────────────────────
        dir_header = ctk.CTkLabel(
            self,
            text="Indexed Folders",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=("#d9e6ff", "#d9e6ff"),
        )
        dir_header.pack(pady=(0, 8), padx=16, anchor="w")

        self.dir_list = ctk.CTkTextbox(
            self,
            height=120,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=("#202a44", "#202a44"),
            border_width=1,
            border_color=("#2b3658", "#2b3658"),
            text_color=("#c7d4f4", "#c7d4f4"),
            corner_radius=10,
        )
        self.dir_list.pack(fill="x", padx=16, pady=(0, 8))
        self.update_directories(indexed_dirs)

        # Buttons row
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(0, 6))

        add_btn = ctk.CTkButton(
            btn_frame,
            text="+ Add Folder",
            command=self.on_add_directory,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#00a5ff", "#00a5ff"),
            hover_color=("#068fdb", "#32bcff"),
            text_color="white",
        )
        add_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        reindex_btn = ctk.CTkButton(
            btn_frame,
            text="Re-Index",
            command=self.on_reindex,
            height=36,
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#ff8a00", "#ff8a00"),
            hover_color=("#dd7400", "#ff9f2f"),
            text_color="white",
        )
        reindex_btn.pack(side="right", fill="x", expand=True, padx=(4, 0))

        reset_btn = ctk.CTkButton(
            self,
            text="Reset App",
            command=self.on_reset,
            height=34,
            corner_radius=10,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#c7342c", "#c7342c"),
            hover_color=("#a92a23", "#d94941"),
        )
        reset_btn.pack(fill="x", padx=16, pady=(4, 6))

        ctk.CTkFrame(self, height=1, fg_color=("#2b3658", "#2b3658")).pack(
            fill="x", padx=16, pady=12
        )

        # ── Stats ────────────────────────────────────────────
        self.stats_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#90a0c9", "#90a0c9"),
            justify="left",
        )
        self.stats_label.pack(padx=16, anchor="w")

        # ── About (bottom) ──────────────────────────────────
        spacer = ctk.CTkFrame(self, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        about_frame = ctk.CTkFrame(self, fg_color="transparent")
        about_frame.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkLabel(
            about_frame,
            text="DocSearch v1.0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#6f80aa", "#6f80aa"),
        ).pack(anchor="w")
        # ctk.CTkLabel(
        #     about_frame,
        #     text="CSE3707 - Information Retrieval\nBMU, Batch 2023",
        #     font=ctk.CTkFont(size=10),
        #     text_color="gray40",
        #     justify="left",
        # ).pack(anchor="w")

    def update_directories(self, dirs: List[str]):
        """Update the displayed list of indexed directories."""
        self.dir_list.configure(state="normal")
        self.dir_list.delete("1.0", "end")
        if dirs:
            for d in dirs:
                # Show just the last part of path for brevity
                short = d.replace("\\", "/").split("/")[-1] or d
                self.dir_list.insert("end", f" {short}\n   {d}\n\n")
        else:
            self.dir_list.insert("end", " No folders indexed yet.\n Click '+ Add Folder' to begin.")
        self.dir_list.configure(state="disabled")

    def update_stats(self, num_docs: int, num_terms: int):
        """Update index statistics display."""
        self.stats_label.configure(
            text=f"Documents: {num_docs}\nTerms: {num_terms:,}"
        )

    def _sync_mode_switches(self, active_mode: str):
        for mode, switch in self._switches.items():
            if mode == active_mode:
                switch.select()
            else:
                switch.deselect()

    def _set_mode(self, mode: str):
        # Keep a single active mode (exclusive toggle behavior)
        if self.current_mode == mode and self._switches[mode].get() == 0:
            self._switches[mode].select()
            return

        self.current_mode = mode
        self.mode_var.set(mode)
        self._sync_mode_switches(mode)
        self.on_mode_change(mode)
