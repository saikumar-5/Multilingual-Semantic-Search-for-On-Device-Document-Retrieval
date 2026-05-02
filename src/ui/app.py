"""
Main Application Window - Modern CustomTkinter desktop GUI.

Features:
- Live as-you-type search with debounce
- Modern dark/light theme
- Clickable suggestion chips
- Background indexing with progress
- Sidebar settings panel
"""

import threading
import pickle
import json
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
import customtkinter as ctk
import logging

from src.config import (
    APP_NAME,
    APP_VERSION,
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    FAISS_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH,
    DATA_DIR,
    DEFAULT_TOP_K,
    ENABLE_CROSS_ENCODER_RERANK,
    CROSS_ENCODER_CANDIDATES,
    CROSS_ENCODER_TOP_K,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_MODEL_LOCAL_DIR,
    MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
    MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
    USE_MULTILINGUAL_CROSS_ENCODER,
    OFFLINE_MODE,
    load_settings,
    save_settings,
)
from src.ingestion.file_router import FileRouter
from src.indexer.inverted_index import InvertedIndex
from src.indexer.tfidf import TFIDFEngine
from src.indexer.vector_index import VectorIndex
from src.indexer.clustering import DocumentClustering
from src.search.keyword_search import KeywordSearch
from src.search.semantic_search import SemanticSearch
from src.search.hybrid_search import HybridSearch
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.wildcard import WildcardSearch
from src.search.ranker import Ranker
from src.search.query_processor import QueryProcessor
from src.ui.search_frame import SearchFrame
from src.ui.settings_frame import SettingsFrame
from src.ui.directory_dialog import DirectoryDialog

logger = logging.getLogger(__name__)

# Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DocSearchApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME}")
        self.geometry("1200x750")
        self.minsize(950, 600)

        # State
        self.settings = load_settings()
        self.documents = []
        self.inv_index = InvertedIndex()
        self.tfidf_engine = None
        self.vector_index = VectorIndex()
        self.keyword_search = None
        self.semantic_search = None
        self.hybrid_search = None
        self.cross_encoder_reranker = None
        self.wildcard_search = None
        self.ranker = None
        self.query_processor = QueryProcessor()
        self.search_mode = self.settings.get("search_mode", "hybrid")
        self._index_loaded = False
        self._indexing_in_progress = False
        self.current_page = "dashboard"
        self.nav_buttons = {}
        self.history_path = DATA_DIR / "search_history.json"
        self.search_history = self._load_search_history()

        self._create_layout()
        self._try_load_existing_index()

    def _create_layout(self):
        """Create the main application layout."""
        self.configure(fg_color=("#0c1222", "#0c1222"))

        # Main container
        self.grid_columnconfigure(0, weight=0, minsize=190)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        # Left navigation rail
        self._create_nav_rail()

        # Center content host (page container)
        self.content_host = ctk.CTkFrame(self, fg_color="transparent")
        self.content_host.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=8)
        self.content_host.grid_rowconfigure(0, weight=1)
        self.content_host.grid_columnconfigure(0, weight=1)

        self._create_content_pages()

        # Settings panel (right sidebar)
        self.settings_frame = SettingsFrame(
            self,
            indexed_dirs=self.settings.get("indexed_directories", []),
            on_add_directory=self._add_directory,
            on_reindex=self._reindex,
            on_reset=self._reset_application,
            on_mode_change=self._change_search_mode,
            current_mode=self.search_mode,
        )
        self.settings_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

        # Progress bar (bottom, hidden by default)
        self.progress_frame = ctk.CTkFrame(self, height=44, corner_radius=0)
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        )
        self.progress_label.pack(side="left", padx=16)
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            width=400,
            height=8,
            corner_radius=4,
            progress_color="#1a73e8",
        )
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)

        self._switch_page("dashboard")

    def _resolve_reranker_config(self):
        if not USE_MULTILINGUAL_CROSS_ENCODER:
            return CROSS_ENCODER_MODEL_NAME, CROSS_ENCODER_MODEL_LOCAL_DIR, True

        multilingual_available = (
            MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR.exists() or not OFFLINE_MODE
        )
        if multilingual_available:
            return (
                MULTILINGUAL_CROSS_ENCODER_MODEL_NAME,
                MULTILINGUAL_CROSS_ENCODER_MODEL_LOCAL_DIR,
                False,
            )

        logger.warning(
            "Multilingual reranker enabled but model not available; falling back to English-only reranker."
        )
        return CROSS_ENCODER_MODEL_NAME, CROSS_ENCODER_MODEL_LOCAL_DIR, True

    def _create_content_pages(self):
        self.pages = {}

        # Dashboard page
        self.search_frame = SearchFrame(self.content_host, on_search=self._handle_search)
        self.search_frame.grid(row=0, column=0, sticky="nsew")
        self.pages["dashboard"] = self.search_frame

        # Files page
        self.files_page = ctk.CTkFrame(
            self.content_host,
            fg_color=("#11182a", "#11182a"),
            corner_radius=16,
            border_width=1,
            border_color=("#26345a", "#26345a"),
        )
        self.files_page.grid(row=0, column=0, sticky="nsew")
        self.pages["files"] = self.files_page
        self._build_files_page()

        # History page
        self.history_page = ctk.CTkFrame(
            self.content_host,
            fg_color=("#11182a", "#11182a"),
            corner_radius=16,
            border_width=1,
            border_color=("#26345a", "#26345a"),
        )
        self.history_page.grid(row=0, column=0, sticky="nsew")
        self.pages["history"] = self.history_page
        self._build_history_page()

        # Settings page (main content)
        self.settings_page = ctk.CTkFrame(
            self.content_host,
            fg_color=("#11182a", "#11182a"),
            corner_radius=16,
            border_width=1,
            border_color=("#26345a", "#26345a"),
        )
        self.settings_page.grid(row=0, column=0, sticky="nsew")
        self.pages["settings"] = self.settings_page
        self._build_settings_page()

    def _build_files_page(self):
        header = ctk.CTkFrame(self.files_page, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            header,
            text="Indexed Files",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=("#d9e6ff", "#d9e6ff"),
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="Refresh",
            width=100,
            command=self._refresh_files_page,
            fg_color=("#00a5ff", "#00a5ff"),
            hover_color=("#068fdb", "#32bcff"),
        ).pack(side="right")

        self.files_summary_label = ctk.CTkLabel(
            self.files_page,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#91a0c6", "#91a0c6"),
            anchor="w",
        )
        self.files_summary_label.pack(fill="x", padx=18, pady=(0, 8))

        self.files_list_frame = ctk.CTkScrollableFrame(
            self.files_page,
            fg_color=("#11182a", "#11182a"),
            scrollbar_button_color=("#3b4f82", "#3b4f82"),
            scrollbar_button_hover_color=("#4762a3", "#4762a3"),
        )
        self.files_list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._refresh_files_page()

    def _build_history_page(self):
        header = ctk.CTkFrame(self.history_page, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            header,
            text="Search History",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=("#d9e6ff", "#d9e6ff"),
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="Clear History",
            width=120,
            command=self._clear_search_history,
            fg_color=("#c7342c", "#c7342c"),
            hover_color=("#a92a23", "#d94941"),
        ).pack(side="right")

        self.history_summary_label = ctk.CTkLabel(
            self.history_page,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("#91a0c6", "#91a0c6"),
            anchor="w",
        )
        self.history_summary_label.pack(fill="x", padx=18, pady=(0, 8))

        self.history_list_frame = ctk.CTkScrollableFrame(
            self.history_page,
            fg_color=("#11182a", "#11182a"),
            scrollbar_button_color=("#3b4f82", "#3b4f82"),
            scrollbar_button_hover_color=("#4762a3", "#4762a3"),
        )
        self.history_list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._refresh_history_page()

    def _build_settings_page(self):
        card = ctk.CTkFrame(
            self.settings_page,
            fg_color=("#151f37", "#151f37"),
            corner_radius=14,
            border_width=1,
            border_color=("#2b3658", "#2b3658"),
        )
        card.pack(fill="x", padx=16, pady=(16, 10))

        ctk.CTkLabel(
            card,
            text="Application Settings",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color=("#d9e6ff", "#d9e6ff"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        ctk.CTkLabel(
            card,
            text="Quick actions for index and data management.",
            font=ctk.CTkFont(size=12),
            text_color=("#91a0c6", "#91a0c6"),
        ).pack(anchor="w", padx=14, pady=(0, 10))

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkButton(
            actions,
            text="Add Folder",
            command=self._add_directory,
            fg_color=("#00a5ff", "#00a5ff"),
            hover_color=("#068fdb", "#32bcff"),
            height=34,
            corner_radius=10,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            actions,
            text="Re-Index",
            command=self._reindex,
            fg_color=("#ff8a00", "#ff8a00"),
            hover_color=("#dd7400", "#ff9f2f"),
            height=34,
            corner_radius=10,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            actions,
            text="Reset App",
            command=self._reset_application,
            fg_color=("#c7342c", "#c7342c"),
            hover_color=("#a92a23", "#d94941"),
            height=34,
            corner_radius=10,
        ).pack(side="left")

        info = ctk.CTkFrame(
            self.settings_page,
            fg_color=("#151f37", "#151f37"),
            corner_radius=14,
            border_width=1,
            border_color=("#2b3658", "#2b3658"),
        )
        info.pack(fill="x", padx=16, pady=(0, 10))
        self.settings_info_label = ctk.CTkLabel(
            info,
            text="",
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12),
            text_color=("#c7d4f4", "#c7d4f4"),
        )
        self.settings_info_label.pack(fill="x", padx=14, pady=12)
        self._refresh_settings_page_info()

    def _switch_page(self, page_name: str):
        if page_name not in self.pages:
            return

        self.current_page = page_name
        self.pages[page_name].tkraise()

        # Dashboard uses the right settings sidebar; other pages expand center area.
        if page_name == "dashboard":
            self.content_host.grid_configure(column=1, columnspan=1, padx=(6, 4), pady=8)
            self.settings_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        else:
            self.settings_frame.grid_remove()
            self.content_host.grid_configure(column=1, columnspan=2, padx=(6, 8), pady=8)

        for key, btn in self.nav_buttons.items():
            is_active = key == page_name
            btn.configure(
                fg_color=("#1e2a48", "#1e2a48") if is_active else "transparent",
                text_color=("#32c3ff", "#32c3ff") if is_active else ("#97a7cf", "#97a7cf"),
                font=ctk.CTkFont(size=15, weight="bold" if is_active else "normal"),
            )

        if page_name == "files":
            self._refresh_files_page()
        elif page_name == "history":
            self._refresh_history_page()
        elif page_name == "settings":
            self._refresh_settings_page_info()

    def _refresh_files_page(self):
        if not hasattr(self, "files_list_frame"):
            return

        for widget in self.files_list_frame.winfo_children():
            widget.destroy()

        total = len(self.documents)
        self.files_summary_label.configure(text=f"{total} files currently loaded in memory")

        if total == 0:
            ctk.CTkLabel(
                self.files_list_frame,
                text="No indexed files yet. Add a folder and run Re-Index.",
                text_color=("#91a0c6", "#91a0c6"),
                font=ctk.CTkFont(size=13),
            ).pack(pady=18)
            return

        for doc in self.documents:
            row = ctk.CTkFrame(
                self.files_list_frame,
                fg_color=("#1a2238", "#1a2238"),
                corner_radius=10,
                border_width=1,
                border_color=("#2b3658", "#2b3658"),
            )
            row.pack(fill="x", padx=4, pady=4)

            title = ctk.CTkLabel(
                row,
                text=f"{doc.get('file_name', 'Unknown')}  ({doc.get('file_type', 'unknown').upper()})",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("#d9e6ff", "#d9e6ff"),
                anchor="w",
            )
            title.pack(fill="x", padx=10, pady=(8, 0))

            path_label = ctk.CTkLabel(
                row,
                text=doc.get("file_path", ""),
                font=ctk.CTkFont(size=11),
                text_color=("#91a0c6", "#91a0c6"),
                anchor="w",
            )
            path_label.pack(fill="x", padx=10, pady=(2, 8))

    def _refresh_history_page(self):
        if not hasattr(self, "history_list_frame"):
            return

        for widget in self.history_list_frame.winfo_children():
            widget.destroy()

        self.history_summary_label.configure(text=f"{len(self.search_history)} saved searches")

        if not self.search_history:
            ctk.CTkLabel(
                self.history_list_frame,
                text="No searches yet. Run a few queries to build history.",
                text_color=("#91a0c6", "#91a0c6"),
                font=ctk.CTkFont(size=13),
            ).pack(pady=18)
            return

        for item in self.search_history:
            row = ctk.CTkFrame(
                self.history_list_frame,
                fg_color=("#1a2238", "#1a2238"),
                corner_radius=10,
                border_width=1,
                border_color=("#2b3658", "#2b3658"),
            )
            row.pack(fill="x", padx=4, pady=4)

            ctk.CTkLabel(
                row,
                text=item.get("query", ""),
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=("#d9e6ff", "#d9e6ff"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(8, 0))

            ctk.CTkLabel(
                row,
                text=f"{item.get('time', '')}  |  mode: {item.get('mode', 'hybrid')}  |  results: {item.get('count', 0)}",
                font=ctk.CTkFont(size=11),
                text_color=("#91a0c6", "#91a0c6"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(2, 8))

    def _refresh_settings_page_info(self):
        if not hasattr(self, "settings_info_label"):
            return
        self.settings_info_label.configure(
            text=(
                f"Search mode: {self.search_mode}\n"
                f"Indexed folders: {len(self.settings.get('indexed_directories', []))}\n"
                f"Loaded documents: {len(self.documents)}\n"
                f"Vocabulary terms: {len(self.inv_index.vocabulary) if hasattr(self.inv_index, 'vocabulary') else 0}"
            )
        )

    def _load_search_history(self) -> list:
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("Could not load search history: %s", e)
        return []

    def _save_search_history(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.search_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Could not save search history: %s", e)

    def _record_search(self, query: str, count: int):
        q = (query or "").strip()
        if not q:
            return

        entry = {
            "query": q,
            "mode": self.search_mode,
            "count": int(count),
            "time": datetime.now().strftime("%H:%M:%S  %d-%b"),
        }

        if self.search_history and self.search_history[0].get("query") == q and self.search_history[0].get("mode") == self.search_mode:
            self.search_history[0] = entry
        else:
            self.search_history.insert(0, entry)

        self.search_history = self.search_history[:200]
        self._save_search_history()
        self.after(0, self._refresh_history_page)

    def _clear_search_history(self):
        if not self.search_history:
            return
        if not messagebox.askyesno("Clear History", "Remove all saved search history?"):
            return
        self.search_history = []
        try:
            if self.history_path.exists():
                self.history_path.unlink()
        except Exception as e:
            logger.warning("Could not remove history file: %s", e)
        self._refresh_history_page()

    def _create_nav_rail(self):
        nav = ctk.CTkFrame(
            self,
            fg_color=("#0f172b", "#0f172b"),
            corner_radius=12,
            border_width=1,
            border_color=("#243457", "#243457"),
        )
        nav.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        logo_wrap = ctk.CTkFrame(nav, fg_color="transparent")
        logo_wrap.pack(fill="x", padx=14, pady=(18, 18))

        ctk.CTkLabel(
            logo_wrap,
            text="DocSearch",
            font=ctk.CTkFont(family="Segoe UI", size=34, weight="bold"),
            text_color=("#14b6ff", "#14b6ff"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            logo_wrap,
            text="Multilingual Semantic Search",
            font=ctk.CTkFont(size=12),
            text_color=("#7d8fb9", "#7d8fb9"),
            anchor="w",
        ).pack(fill="x")

        menu_items = [
            ("dashboard", "Dashboard", "\u25a3"),
            ("files", "Files", "\U0001f5ce"),
            ("history", "History", "\u21bb"),
            ("settings", "Settings", "\u2699"),
        ]
        for key, label, icon in menu_items:
            row = ctk.CTkFrame(nav, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            btn = ctk.CTkButton(
                row,
                text=f"{icon}  {label}",
                height=38,
                corner_radius=8,
                anchor="w",
                fg_color="transparent",
                hover_color=("#22365f", "#22365f"),
                text_color=("#97a7cf", "#97a7cf"),
                font=ctk.CTkFont(size=15),
                command=lambda p=key: self._switch_page(p),
            )
            btn.pack(fill="x")
            self.nav_buttons[key] = btn

    def _try_load_existing_index(self):
        """Load existing index from disk if available."""
        if (
            INVERTED_INDEX_PATH.exists()
            and TFIDF_INDEX_PATH.exists()
            and DOCUMENT_STORE_PATH.exists()
        ):
            try:
                self.search_frame.set_loading("Loading saved index...")
                self._load_index()
                self.search_frame.status_label.configure(
                    text=f"{len(self.documents)} documents indexed  |  Type to search instantly",
                    text_color=("#2b7a0b", "#4caf50"),
                )
                self.settings_frame.update_stats(
                    len(self.documents), len(self.inv_index.vocabulary)
                )
                self.search_frame._show_placeholder()
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self._show_first_run()
        else:
            self._show_first_run()

    def _show_first_run(self):
        """Show first-run dialog to select directories."""
        self.search_frame._show_placeholder()
        if not self.settings.get("indexed_directories"):
            self.after(500, self._prompt_directory_selection)

    def _prompt_directory_selection(self):
        """Open directory selection dialog."""
        dialog = DirectoryDialog(
            self, self.settings.get("indexed_directories", [])
        )
        result = dialog.get_result()

        if result is not None and result:
            self.settings["indexed_directories"] = result
            save_settings(self.settings)
            self.settings_frame.update_directories(result)
            self._index_directories(result)

    def _add_directory(self):
        """Add a new directory via file dialog."""
        directory = filedialog.askdirectory(title="Select a folder to index")
        if directory:
            dirs = self.settings.get("indexed_directories", [])
            if directory not in dirs:
                dirs.append(directory)
                self.settings["indexed_directories"] = dirs
                save_settings(self.settings)
                self.settings_frame.update_directories(dirs)
                self._index_directories(dirs)

    def _reindex(self):
        """Re-index all configured directories."""
        dirs = self.settings.get("indexed_directories", [])
        if dirs:
            self._index_directories(dirs)
        else:
            self._prompt_directory_selection()

    def _change_search_mode(self, mode: str):
        """Change the active search mode."""
        self.search_mode = mode
        self.settings["search_mode"] = mode
        save_settings(self.settings)

    def _index_directories(self, directories):
        """Start indexing in a background thread."""
        self._indexing_in_progress = True
        self.progress_frame.grid(
            row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8)
        )
        self.search_frame.set_loading("Indexing documents... This may take a moment.")

        thread = threading.Thread(
            target=self._index_worker, args=(directories,), daemon=True
        )
        thread.start()

    def _index_worker(self, directories):
        """Background worker for indexing."""
        try:
            self._update_progress("Scanning files...", 0.05)
            router = FileRouter()
            all_documents = []

            for directory in directories:
                docs = router.process_directory(
                    directory,
                    progress_callback=lambda cur, total, name: self._update_progress(
                        f"Parsing: {name}", cur / total * 0.3
                    ),
                )
                all_documents.extend(docs)

            for i, doc in enumerate(all_documents):
                doc["doc_id"] = i

            self.documents = all_documents

            self._update_progress("Building inverted index...", 0.35)
            self.inv_index = InvertedIndex()
            self.inv_index.build(all_documents)

            self._update_progress("Computing TF-IDF weights...", 0.45)
            self.tfidf_engine = TFIDFEngine(self.inv_index)
            self.tfidf_engine.compute()

            self._update_progress("Building semantic vectors...", 0.5)
            self.vector_index = VectorIndex()
            self.vector_index.build(
                all_documents,
                progress_callback=lambda cur, total, status: self._update_progress(
                    status, 0.5 + (cur / total) * 0.35
                ),
            )

            self._update_progress("Initializing search engines...", 0.9)
            self.keyword_search = KeywordSearch(self.inv_index, self.tfidf_engine)
            self.semantic_search = SemanticSearch(self.vector_index)
            self.cross_encoder_reranker = None
            if ENABLE_CROSS_ENCODER_RERANK:
                model_name, model_local_dir, rerank_english_only = (
                    self._resolve_reranker_config()
                )
                self.cross_encoder_reranker = CrossEncoderReranker(
                    all_documents,
                    model_name=model_name,
                    model_local_dir=model_local_dir,
                    top_candidates=CROSS_ENCODER_CANDIDATES,
                )
            else:
                rerank_english_only = True
            self.hybrid_search = HybridSearch(
                self.keyword_search,
                self.semantic_search,
                reranker=self.cross_encoder_reranker,
                rerank_candidates=CROSS_ENCODER_CANDIDATES,
                rerank_english_only=rerank_english_only,
            )
            self.wildcard_search = WildcardSearch(self.inv_index, self.tfidf_engine)
            self.wildcard_search.build()
            self.ranker = Ranker(all_documents)
            self._index_loaded = True

            self._update_progress("Saving index to disk...", 0.95)
            self._save_index()

            self._update_progress("Done!", 1.0)
            self.after(800, self._indexing_complete)

        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            self.after(0, lambda: self._indexing_error(str(e)))

    def _update_progress(self, message: str, progress: float):
        self.after(
            0,
            lambda: (
                self.progress_label.configure(text=message),
                self.progress_bar.set(min(progress, 1.0)),
            ),
        )

    def _indexing_complete(self):
        self._indexing_in_progress = False
        self.progress_frame.grid_forget()
        self.search_frame.status_label.configure(
            text=f"{len(self.documents)} documents indexed  |  Type to search instantly",
            text_color=("#2b7a0b", "#4caf50"),
        )
        self.settings_frame.update_stats(
            len(self.documents), len(self.inv_index.vocabulary)
        )
        self.search_frame._show_placeholder()
        self._refresh_files_page()
        self._refresh_settings_page_info()

    def _indexing_error(self, error: str):
        self._indexing_in_progress = False
        self.progress_frame.grid_forget()
        self.search_frame.status_label.configure(
            text=f"Indexing error: {error}", text_color="#e74c3c"
        )

    def _reset_application(self):
        """Reset application to a clean state by deleting persisted index artifacts."""
        if self._indexing_in_progress:
            messagebox.showwarning(
                "Indexing in progress",
                "Please wait for indexing to complete before resetting.",
            )
            return

        confirmed = messagebox.askyesno(
            "Reset Application",
            "This will delete saved index files (.pkl/.faiss) and clear indexed folders.\n\nContinue?",
            icon="warning",
        )
        if not confirmed:
            return

        paths_to_remove = [
            INVERTED_INDEX_PATH,
            TFIDF_INDEX_PATH,
            DOCUMENT_STORE_PATH,
            FAISS_INDEX_PATH,
            DATA_DIR / "vector_meta.pkl",
            METADATA_PATH,
        ]

        removed = 0
        for path in paths_to_remove:
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning("Could not delete %s: %s", path, e)

        # Reset runtime state
        self.documents = []
        self.inv_index = InvertedIndex()
        self.tfidf_engine = None
        self.vector_index = VectorIndex()
        self.keyword_search = None
        self.semantic_search = None
        self.hybrid_search = None
        self.cross_encoder_reranker = None
        self.wildcard_search = None
        self.ranker = None
        self._index_loaded = False

        # Reset persisted app settings for a true fresh start
        self.settings["indexed_directories"] = []
        self.settings["search_mode"] = "hybrid"
        save_settings(self.settings)
        self.search_mode = "hybrid"

        # Refresh UI state
        self.search_frame._clear_search()
        self.search_frame.status_label.configure(
            text=f"Reset complete  |  Removed {removed} saved index files",
            text_color=("#2b7a0b", "#4caf50"),
        )
        self.settings_frame.mode_var.set("hybrid")
        self.settings_frame.update_directories([])
        self.settings_frame.update_stats(0, 0)
        self.search_history = []
        try:
            if self.history_path.exists():
                self.history_path.unlink()
        except Exception as e:
            logger.warning("Could not remove history file: %s", e)
        self._refresh_files_page()
        self._refresh_history_page()
        self._refresh_settings_page_info()

        messagebox.showinfo(
            "Reset complete",
            "Application reset to fresh state. Add a folder and click Re-Index.",
        )

    def _handle_search(self, query: str) -> list:
        """Handle a search query from the UI."""
        if not self._index_loaded:
            return []

        parsed = self.query_processor.parse(query)

        if parsed["type"] == "wildcard":
            raw_results = self.wildcard_search.search(parsed["raw"], DEFAULT_TOP_K)
        elif self.search_mode == "keyword":
            raw_results = self.keyword_search.search(query, DEFAULT_TOP_K)
        elif self.search_mode == "semantic":
            raw_results = self.semantic_search.search(query, DEFAULT_TOP_K)
        else:
            raw_results = self.hybrid_search.search(query, CROSS_ENCODER_TOP_K)

        results = self.ranker.format_results(raw_results, query)
        self._record_search(query, len(results))
        return results

    def _save_index(self):
        self.inv_index.save(INVERTED_INDEX_PATH)
        self.tfidf_engine.save(TFIDF_INDEX_PATH)
        faiss_path = DATA_DIR / "vectors.faiss"
        meta_path = DATA_DIR / "vector_meta.pkl"
        self.vector_index.save(faiss_path, meta_path)
        with open(DOCUMENT_STORE_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def _load_index(self):
        with open(DOCUMENT_STORE_PATH, "rb") as f:
            self.documents = pickle.load(f)

        self.inv_index = InvertedIndex()
        self.inv_index.load(INVERTED_INDEX_PATH)

        self.tfidf_engine = TFIDFEngine(self.inv_index)
        self.tfidf_engine.load(TFIDF_INDEX_PATH)

        self.vector_index = VectorIndex()
        faiss_path = DATA_DIR / "vectors.faiss"
        meta_path = DATA_DIR / "vector_meta.pkl"
        if faiss_path.exists() and meta_path.exists():
            self.vector_index.load(faiss_path, meta_path)

        self.keyword_search = KeywordSearch(self.inv_index, self.tfidf_engine)
        self.semantic_search = SemanticSearch(self.vector_index)
        self.cross_encoder_reranker = None
        if ENABLE_CROSS_ENCODER_RERANK:
            model_name, model_local_dir, rerank_english_only = (
                self._resolve_reranker_config()
            )
            self.cross_encoder_reranker = CrossEncoderReranker(
                self.documents,
                model_name=model_name,
                model_local_dir=model_local_dir,
                top_candidates=CROSS_ENCODER_CANDIDATES,
            )
        else:
            rerank_english_only = True

        self.hybrid_search = HybridSearch(
            self.keyword_search,
            self.semantic_search,
            reranker=self.cross_encoder_reranker,
            rerank_candidates=CROSS_ENCODER_CANDIDATES,
            rerank_english_only=rerank_english_only,
        )
        self.wildcard_search = WildcardSearch(self.inv_index, self.tfidf_engine)
        self.wildcard_search.build()
        self.ranker = Ranker(self.documents)
        self._index_loaded = True


def run_app():
    """Launch the DocSearch application."""
    app = DocSearchApp()
    app.mainloop()
