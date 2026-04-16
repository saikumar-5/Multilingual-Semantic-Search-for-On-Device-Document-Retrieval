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
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
import logging

from src.config import (
    APP_NAME,
    APP_VERSION,
    INVERTED_INDEX_PATH,
    TFIDF_INDEX_PATH,
    FAISS_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    DATA_DIR,
    DEFAULT_TOP_K,
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
        self.wildcard_search = None
        self.ranker = None
        self.query_processor = QueryProcessor()
        self.search_mode = self.settings.get("search_mode", "hybrid")
        self._index_loaded = False

        self._create_layout()
        self._try_load_existing_index()

    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        self.grid_columnconfigure(0, weight=4)
        self.grid_columnconfigure(1, weight=0, minsize=280)
        self.grid_rowconfigure(0, weight=1)

        # Search panel (left side - takes most space)
        self.search_frame = SearchFrame(self, on_search=self._handle_search)
        self.search_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        # Settings panel (right sidebar)
        self.settings_frame = SettingsFrame(
            self,
            indexed_dirs=self.settings.get("indexed_directories", []),
            on_add_directory=self._add_directory,
            on_reindex=self._reindex,
            on_mode_change=self._change_search_mode,
            current_mode=self.search_mode,
        )
        self.settings_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

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
        self.progress_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
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
            self.hybrid_search = HybridSearch(
                self.keyword_search, self.semantic_search
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
        self.progress_frame.grid_forget()
        self.search_frame.status_label.configure(
            text=f"{len(self.documents)} documents indexed  |  Type to search instantly",
            text_color=("#2b7a0b", "#4caf50"),
        )
        self.settings_frame.update_stats(
            len(self.documents), len(self.inv_index.vocabulary)
        )
        self.search_frame._show_placeholder()

    def _indexing_error(self, error: str):
        self.progress_frame.grid_forget()
        self.search_frame.status_label.configure(
            text=f"Indexing error: {error}", text_color="#e74c3c"
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
            raw_results = self.hybrid_search.search(query, DEFAULT_TOP_K)

        return self.ranker.format_results(raw_results, query)

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
        self.hybrid_search = HybridSearch(self.keyword_search, self.semantic_search)
        self.wildcard_search = WildcardSearch(self.inv_index, self.tfidf_engine)
        self.wildcard_search.build()
        self.ranker = Ranker(self.documents)
        self._index_loaded = True


def run_app():
    """Launch the DocSearch application."""
    app = DocSearchApp()
    app.mainloop()
