"""Streamlit app integrating preprocessing, LLM handling, classification and explainability.

REDESIGNED UI: Modern dashboard with card-based layout.

Usage:
    streamlit run app.py

The app lets the user upload a `.txt` file, runs the preprocessing
pipeline, calls the LLM (or a mock), classifies and explains
requirements, and provides a human-in-the-loop validation UI with
downloadable results.
"""

from typing import List, Dict, Any, Optional
import io
import json
import csv
import streamlit as st

from src import preprocess, llm_handler, classifier, explainability


# ============================================================================
# CUSTOM CSS FOR DASHBOARD STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for modern dashboard appearance."""
    css = """
    <style>
        /* General page styling */
        body {
            background-color: #0f1419;
            color: #e8eaed;
        }

        .main {
            background-color: #0f1419;
        }

        /* Left panel container */
        .left-panel {
            background: linear-gradient(135deg, #1a1f2e 0%, #16212d 100%);
            border-radius: 12px;
            padding: 2rem;
            border: 1px solid #2a3f5f;
            height: 100%;
        }

        /* App header */
        .app-header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #2a3f5f;
        }

        .app-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #00d9ff;
            margin: 0.5rem 0;
            letter-spacing: 0.3px;
        }

        .app-subtitle {
            font-size: 0.85rem;
            color: #8892b0;
            margin-top: 0.5rem;
            line-height: 1.4;
        }

        /* Input section styling */
        .input-section {
            background: rgba(42, 63, 95, 0.3);
            border: 1px solid #2a3f5f;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .input-label {
            font-size: 0.9rem;
            color: #00d9ff;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Card styling */
        .requirement-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #141920 100%);
            border: 1px solid #2a3f5f;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }

        .requirement-card:hover {
            border-color: #00d9ff;
            box-shadow: 0 8px 12px rgba(0, 217, 255, 0.15);
        }

        /* Card badges */
        .badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .badge-functional {
            background-color: rgba(76, 175, 80, 0.2);
            color: #4cb050;
            border: 1px solid #4cb050;
        }

        .badge-nonfunctional {
            background-color: rgba(255, 152, 0, 0.2);
            color: #ff9800;
            border: 1px solid #ff9800;
        }

        .badge-subtype {
            background-color: rgba(156, 39, 176, 0.2);
            color: #9c27b0;
            border: 1px solid #9c27b0;
        }

        /* Requirement ID */
        .requirement-id {
            font-size: 0.9rem;
            font-weight: 700;
            color: #00d9ff;
            margin: 0.75rem 0;
            font-family: 'Courier New', monospace;
        }

        /* Requirement text */
        .requirement-text {
            font-size: 0.95rem;
            color: #e8eaed;
            line-height: 1.6;
            margin: 0.75rem 0;
            padding: 0.75rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 6px;
            border-left: 3px solid #00d9ff;
        }

        /* Action buttons */
        .action-buttons {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .action-btn {
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .btn-accept {
            background: rgba(76, 175, 80, 0.2);
            color: #4cb050;
            border: 1px solid #4cb050;
        }

        .btn-accept:hover {
            background: rgba(76, 175, 80, 0.4);
        }

        .btn-edit {
            background: rgba(33, 150, 243, 0.2);
            color: #2196f3;
            border: 1px solid #2196f3;
        }

        .btn-edit:hover {
            background: rgba(33, 150, 243, 0.4);
        }

        .btn-reject {
            background: rgba(244, 67, 54, 0.2);
            color: #f44336;
            border: 1px solid #f44336;
        }

        .btn-reject:hover {
            background: rgba(244, 67, 54, 0.4);
        }

        /* Status tags */
        .status-pending {
            color: #ffc107;
        }

        .status-accepted {
            color: #4cb050;
        }

        .status-edited {
            color: #2196f3;
        }

        .status-rejected {
            color: #f44336;
        }

        /* Results header */
        .results-header {
            margin-bottom: 2rem;
        }

        .results-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e8eaed;
            margin-bottom: 0.5rem;
        }

        .results-subtitle {
            font-size: 0.9rem;
            color: #8892b0;
        }

        /* Summary section */
        .summary-section {
            background: rgba(42, 63, 95, 0.3);
            border: 1px solid #2a3f5f;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 2rem 0;
        }

        .summary-stat {
            display: inline-block;
            margin-right: 2rem;
            margin-bottom: 1rem;
        }

        .summary-stat-label {
            color: #8892b0;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }

        .summary-stat-value {
            color: #00d9ff;
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* Main button styling */
        .stButton > button {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            color: #0f1419;
            border: none;
            font-weight: 700;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            box-shadow: 0 8px 16px rgba(0, 217, 255, 0.4);
            transform: translateY(-2px);
        }

        /* Download buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #2196f3 0%, #0d47a1 100%);
            color: white;
            border: none;
            font-weight: 600;
            border-radius: 8px;
        }

        .stDownloadButton > button:hover {
            box-shadow: 0 8px 16px rgba(33, 150, 243, 0.4);
        }

        /* Text input and area styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: #16212d;
            color: #e8eaed;
            border: 1px solid #2a3f5f !important;
            border-radius: 8px;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #00d9ff !important;
            box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.15);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(42, 63, 95, 0.3);
            border-radius: 8px;
        }

        /* Hide default streamlit elements */
        #MainMenu {
            visibility: hidden;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# BACKEND FUNCTIONS (UNCHANGED)
# ============================================================================

def process_uploaded_text(text: str, *, remove_headers: bool = True) -> List[str]:
    """Clean and split uploaded plain text into sentences."""
    cleaned = preprocess.clean_text(text, remove_headers=remove_headers)
    sentences = preprocess.split_sentences(cleaned)
    sentences = preprocess.remove_empty_or_meaningless(sentences)
    return sentences


def run_pipeline(
    sentences: List[str],
    prompt_template: str,
    mock: bool = False,
    detect_subtypes: bool = True,
) -> List[Dict[str, Any]]:
    """Call the LLM handler to extract requirements, then classify and explain."""
    raw_items = llm_handler.generate_requirements_from_llm(
        sentences, prompt_template, mock=mock
    )

    texts = [itm.get("text") or "" for itm in raw_items]
    classified = classifier.classify_requirements(texts, detect_subtypes)
    explained = explainability.explain_requirements(classified)

    final: List[Dict[str, Any]] = []
    for i, item in enumerate(explained):
        merged = dict(item)
        llm_item = raw_items[i] if i < len(raw_items) and isinstance(raw_items[i], dict) else {}
        if llm_item.get("explanation"):
            merged["explanation"] = llm_item.get("explanation")
        merged.update({k: v for k, v in llm_item.items() if k not in merged})
        final.append(merged)

    return final


def to_csv_bytes(items: List[Dict[str, Any]]) -> bytes:
    """Return CSV bytes for `items`."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "classification", "subtype", "explanation"])
    for it in items:
        writer.writerow([
            it.get("text", ""),
            it.get("classification", ""),
            it.get("subtype", ""),
            it.get("explanation", ""),
        ])
    return output.getvalue().encode("utf-8")


def to_json_bytes(items: List[Dict[str, Any]]) -> bytes:
    """Return JSON bytes for `items`."""
    return json.dumps(items, indent=2).encode("utf-8")


# ============================================================================
# UI COMPONENT FUNCTIONS (NEW)
# ============================================================================

def generate_requirement_id(index: int, classification: str) -> str:
    """Generate professional requirement ID (FR-001, NFR-001, etc)."""
    prefix = "FR" if classification == "Functional" else "NFR"
    return f"{prefix}-{index + 1:03d}"


def render_requirement_card(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    """Render a single requirement card with modern design.
    
    Returns the possibly edited item with updated action status.
    """
    classification = item.get("classification", "Unknown")
    subtype = item.get("subtype")
    text = item.get("text", "")
    explanation = item.get("explanation", "")
    current_status = item.get("status", "pending")

    req_id = generate_requirement_id(idx, classification)

    # Create card container
    with st.container():
        col_badge, col_status = st.columns([3, 1])

        with col_badge:
            # Badges row
            badge_html = f'<div style="margin-bottom: 0.5rem;">'
            
            # Classification badge
            if classification == "Functional":
                badge_html += '<span class="badge badge-functional">FUNCTIONAL</span>'
            else:
                badge_html += '<span class="badge badge-nonfunctional">NON-FUNCTIONAL</span>'
            
            # Subtype badge
            if subtype:
                badge_html += f'<span class="badge badge-subtype">{subtype.upper()}</span>'
            
            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)

        with col_status:
            # Status indicator
            status_colors = {
                "pending": "🔘 Pending",
                "accepted": "✓ Accepted",
                "edited": "✎ Edited",
                "rejected": "✗ Rejected",
            }
            status_text = status_colors.get(current_status, "🔘 Pending")
            st.markdown(f"<p style='text-align: right; color: #8892b0; font-size: 0.85rem;'>{status_text}</p>", 
                       unsafe_allow_html=True)

        # Requirement ID
        st.markdown(f'<div class="requirement-id">ID: {req_id}</div>', unsafe_allow_html=True)

        # Requirement text
        st.markdown(f'<div class="requirement-text">{text}</div>', unsafe_allow_html=True)

        # Explanation in collapsible section
        if explanation:
            with st.expander("👁️ View AI Reasoning", expanded=False):
                st.markdown(f"<p style='color: #8892b0; line-height: 1.6; font-size: 0.9rem;'>{explanation}</p>", 
                           unsafe_allow_html=True)

        # Action buttons
        st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
        
        col_accept, col_edit, col_reject = st.columns(3)
        
        with col_accept:
            if st.button("✓ Accept", key=f"accept_{idx}", use_container_width=True):
                item["status"] = "accepted"
                item["action"] = "accept"
        
        with col_edit:
            if st.button("✎ Edit", key=f"edit_{idx}", use_container_width=True):
                st.session_state[f"edit_modal_{idx}"] = True
        
        with col_reject:
            if st.button("✗ Reject", key=f"reject_{idx}", use_container_width=True):
                item["status"] = "rejected"
                item["action"] = "reject"
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Edit modal (collapsible edit form)
        if st.session_state.get(f"edit_modal_{idx}", False):
            with st.expander("Edit Requirement", expanded=True):
                edited_text = st.text_area(f"Requirement text", 
                                          value=item.get("text", ""), 
                                          key=f"text_edit_{idx}",
                                          height=100)
                
                edited_classification = st.selectbox(
                    "Classification",
                    options=["Functional", "Non-Functional"],
                    index=0 if classification == "Functional" else 1,
                    key=f"class_edit_{idx}",
                )
                
                edited_subtype = None
                if edited_classification == "Non-Functional":
                    edited_subtype = st.selectbox(
                        "Subtype",
                        options=["", "Performance", "Security", "Usability", "Reliability"],
                        index=(0 if not subtype else 
                              ["", "Performance", "Security", "Usability", "Reliability"].index(subtype) 
                              if subtype in ["", "Performance", "Security", "Usability", "Reliability"] else 0),
                        key=f"sub_edit_{idx}",
                    )
                    edited_subtype = edited_subtype if edited_subtype else None
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Save Changes", key=f"save_edit_{idx}", use_container_width=True):
                        item["text"] = edited_text
                        item["classification"] = edited_classification
                        item["subtype"] = edited_subtype
                        item["status"] = "edited"
                        item["action"] = "edit"
                        st.session_state[f"edit_modal_{idx}"] = False
                        st.rerun()
                
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_edit_{idx}", use_container_width=True):
                        st.session_state[f"edit_modal_{idx}"] = False
                        st.rerun()

        st.markdown("---")

    return item


# ============================================================================
# MAIN APP
# ============================================================================

def main() -> None:
    """Main app entry point with dashboard layout."""
    st.set_page_config(
        page_title="LLM Requirements Analysis Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    apply_custom_css()

    # Create two-column layout
    left_col, right_col = st.columns([0.3, 0.7], gap="large")

    # ====================================================================
    # LEFT PANEL: INPUT SECTION
    # ====================================================================
    with left_col:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)

        # App header
        st.markdown("""
            <div class="app-header">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
                <div class="app-title">LLM-Requirements<br/>Analysis Assistant</div>
                <div class="app-subtitle">Offline AI-assisted Requirements Engineering Tool</div>
            </div>
        """, unsafe_allow_html=True)

        # Input section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">📄 Upload or Paste Requirements</div>', unsafe_allow_html=True)

        # File uploader
        uploaded = st.file_uploader("Upload requirements (.txt)", type=["txt"], label_visibility="collapsed")

        st.markdown("<div style='text-align: center; color: #8892b0; font-size: 0.8rem; margin: 0.5rem 0;'>— or —</div>", 
                   unsafe_allow_html=True)

        # Text area for pasting
        free_text = st.text_area(
            "Paste requirements text",
            height=250,
            placeholder="Paste meeting notes, stakeholder feedback, or requirement documents here...",
            label_visibility="collapsed"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Analyze button
        analyze_clicked = st.button("🚀 Analyze Requirements", use_container_width=True)

        # Settings section (collapsed)
        with st.expander("⚙️ Advanced Settings", expanded=False):
            mock = st.checkbox("Mock LLM (no API key)", value=True)
            detect_subtypes = st.checkbox("Detect NFR subtypes", value=True)
            remove_headers = st.checkbox("Remove document headers", value=True)

        # Store settings in session state
        if "mock" not in st.session_state:
            st.session_state["mock"] = True
            st.session_state["detect_subtypes"] = True
            st.session_state["remove_headers"] = True

        st.markdown('</div>', unsafe_allow_html=True)

    # ====================================================================
    # RIGHT PANEL: RESULTS SECTION
    # ====================================================================
    with right_col:
        st.markdown('<div class="results-header">', unsafe_allow_html=True)
        st.markdown('<div class="results-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="results-subtitle">Upload or paste requirements and click Analyze</div>', 
                   unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Handle analysis
        source_text = ""
        if analyze_clicked:
            if free_text and free_text.strip():
                source_text = free_text
            elif uploaded is not None:
                raw_bytes = uploaded.read()
                try:
                    source_text = raw_bytes.decode("utf-8")
                except Exception:
                    source_text = raw_bytes.decode("latin-1")

            if source_text:
                sentences = process_uploaded_text(source_text, remove_headers=st.session_state["remove_headers"])
                
                if len(sentences) == 0:
                    st.warning("No usable sentences found after preprocessing.")
                else:
                    st.info(f"✓ Preprocessing produced {len(sentences)} sentences. Running analysis...")

                    with st.spinner("Analyzing requirements..."):
                        try:
                            prompt_template = ""
                            prompt_path = "prompts/requirement_prompt.txt"
                            try:
                                prompt_template = llm_handler.load_prompt(prompt_path)
                            except FileNotFoundError:
                                prompt_template = (
                                    "Classify the following requirements into Functional or Non-Functional.\n\n{text}\n\n"
                                    "Return a JSON array of objects with fields: text, classification, subtype (optional), explanation (optional)."
                                )

                            items = run_pipeline(
                                sentences,
                                prompt_template,
                                mock=st.session_state["mock"],
                                detect_subtypes=st.session_state["detect_subtypes"]
                            )

                            # Initialize status and action for each item
                            for item in items:
                                if "status" not in item:
                                    item["status"] = "pending"
                                if "action" not in item:
                                    item["action"] = None

                            st.session_state["pipeline_items"] = items
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")
            else:
                st.warning("Please upload a file or paste text to analyze.")

        # Display requirement cards
        items: List[Dict[str, Any]] = st.session_state.get("pipeline_items", [])

        if items:
            st.markdown("---")

            # Process cards and collect actions
            validated_items = []
            for i, item in enumerate(items):
                updated_item = render_requirement_card(i, item)
                items[i] = updated_item
                if updated_item.get("action") == "accept":
                    validated_items.append(updated_item)

            st.session_state["pipeline_items"] = items

            # Summary and report section
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e8eaed; margin-bottom: 1rem;">Generate Final Report</h3>', 
                       unsafe_allow_html=True)

            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

            total = len(items)
            accepted = sum(1 for i in items if i.get("action") == "accept")
            edited = sum(1 for i in items if i.get("action") == "edit")
            rejected = sum(1 for i in items if i.get("action") == "reject")

            with col_stats1:
                st.markdown(f"""
                    <div class="summary-stat">
                        <div class="summary-stat-label">TOTAL</div>
                        <div class="summary-stat-value">{total}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_stats2:
                st.markdown(f"""
                    <div class="summary-stat">
                        <div class="summary-stat-label">ACCEPTED</div>
                        <div class="summary-stat-value" style="color: #4cb050;">{accepted}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_stats3:
                st.markdown(f"""
                    <div class="summary-stat">
                        <div class="summary-stat-label">EDITED</div>
                        <div class="summary-stat-value" style="color: #2196f3;">{edited}</div>
                    </div>
                """, unsafe_allow_html=True)

            with col_stats4:
                st.markdown(f"""
                    <div class="summary-stat">
                        <div class="summary-stat-label">REJECTED</div>
                        <div class="summary-stat-value" style="color: #f44336;">{rejected}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Download section
            if validated_items:
                st.markdown("---")
                st.markdown("<h4 style='color: #e8eaed; margin: 1.5rem 0 1rem 0;'>Export Results</h4>", 
                           unsafe_allow_html=True)

                col_csv, col_json = st.columns(2)

                with col_csv:
                    csv_bytes = to_csv_bytes(validated_items)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_bytes,
                        file_name="validated_requirements.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_json:
                    json_bytes = to_json_bytes(validated_items)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_bytes,
                        file_name="validated_requirements.json",
                        mime="application/json",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()
