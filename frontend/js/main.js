/**
 * Main application client script
 * Handles:
 * 1. User authentication state & logout
 * 2. Document library loading, selecting, and deleting
 * 3. PDF upload, extraction, and chunking display
 * 4. RAG Chat Q&A with Gemini and persistent conversation history
 */

const API_BASE_URL = window.API_BASE_URL !== undefined 
    ? window.API_BASE_URL 
    : (window.location.protocol.startsWith("http") ? "" : "http://127.0.0.1:5000");

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. User Authentication & Header Setup
    // -------------------------------------------------------------
    const userDisplayName = document.getElementById("user-display-name");
    const logoutBtn = document.getElementById("logout-btn");
    const loginNavBtn = document.getElementById("login-nav-btn");

    let currentUser = null;
    let currentDocumentId = null;
    let currentDocumentName = null;

    async function checkAuthSession() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
                method: "GET",
                credentials: "include"
            });
            if (res.ok) {
                const data = await res.json();
                if (data.success && data.user) {
                    currentUser = data.user;
                    if (userDisplayName) {
                        userDisplayName.textContent = `👤 ${currentUser.username}`;
                        userDisplayName.classList.remove("hidden");
                    }
                    if (logoutBtn) logoutBtn.classList.remove("hidden");
                    if (loginNavBtn) loginNavBtn.classList.add("hidden");
                    
                    // Load user documents
                    loadUserDocuments();
                    return;
                }
            }
        } catch (err) {
            console.log("Not logged in or backend offline", err);
        }

        // Unauthenticated state
        currentUser = null;
        if (userDisplayName) userDisplayName.classList.add("hidden");
        if (logoutBtn) logoutBtn.classList.add("hidden");
        if (loginNavBtn) loginNavBtn.classList.remove("hidden");
    }

    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            try {
                await fetch(`${API_BASE_URL}/api/auth/logout`, {
                    method: "POST",
                    credentials: "include"
                });
            } catch (err) {}
            localStorage.removeItem("study_assistant_user");
            window.location.href = "login.html";
        });
    }

    // -------------------------------------------------------------
    // 2. Document Library Management
    // -------------------------------------------------------------
    const docLibraryList = document.getElementById("doc-library-list");
    const docLibraryCount = document.getElementById("doc-library-count");

    async function loadUserDocuments() {
        if (!docLibraryList) return;

        try {
            const res = await fetch(`${API_BASE_URL}/api/documents`, {
                method: "GET",
                credentials: "include"
            });
            const data = await res.json();

            if (res.ok && data.success) {
                const docs = data.documents || [];
                if (docLibraryCount) docLibraryCount.textContent = `${docs.length} Docs`;

                if (docs.length === 0) {
                    docLibraryList.innerHTML = '<p class="text-muted" style="font-size: 0.85rem; padding: 0.5rem 0;">No documents uploaded yet.</p>';
                    return;
                }

                docLibraryList.innerHTML = "";
                docs.forEach((doc) => {
                    const item = document.createElement("div");
                    item.className = `doc-item ${doc.id === currentDocumentId ? "active" : ""}`;
                    item.id = `doc-item-${doc.id}`;
                    item.innerHTML = `
                        <div>
                            <div class="doc-item-title">
                                <span>📖</span>
                                <span title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                            </div>
                            <div class="doc-item-meta">
                                <span>${doc.chunk_count || 0} excerpts</span> • 
                                <span>${
                                    doc.status === "indexed" ? '<span style="color: var(--accent-sage); font-weight: 600;">Indexed</span>' :
                                    doc.status === "ocr_required" ? '<span style="color: var(--accent-ochre); font-weight: 600;" title="Scanned or image-only PDF. OCR required to extract text.">OCR Required</span>' :
                                    doc.status === "failed" ? `<span style="color: var(--accent-rust); font-weight: 600;" title="${escapeHtml(doc.error_message || 'Processing failed')}">Failed</span>` :
                                    doc.status === "processing" ? '<span style="color: var(--accent-terracotta); font-weight: 600;">Cataloging...</span>' :
                                    '<span style="color: var(--accent-navy); font-weight: 600;">Deposited</span>'
                                }</span>
                            </div>
                        </div>
                        <div class="doc-item-actions">
                            <button class="btn btn-sm btn-secondary select-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}">
                                ${doc.id === currentDocumentId ? "Reading" : "Open"}
                            </button>
                            <button class="btn btn-sm btn-secondary summary-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}" title="Generate AI Study Summary">
                                📝 Summary
                            </button>
                            <button class="btn btn-sm btn-accent mcq-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}" title="Generate Practice Quiz">
                                🎯 Quiz
                            </button>
                            <button class="btn btn-sm btn-warning questions-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}" title="Generate Important Exam Questions">
                                ⭐ Exam Qs
                            </button>
                            <button class="btn btn-sm btn-danger delete-doc-btn" data-id="${doc.id}" data-name="${escapeHtml(doc.filename)}" title="Delete volume">
                                ✕
                            </button>
                        </div>
                    `;
                    docLibraryList.appendChild(item);
                });

                // Attach select handlers
                docLibraryList.querySelectorAll(".select-doc-btn").forEach((btn) => {
                    btn.addEventListener("click", () => {
                        const docId = btn.getAttribute("data-id");
                        const docName = btn.getAttribute("data-name");
                        selectDocument(docId, docName);
                    });
                });

                // Attach summary handlers
                docLibraryList.querySelectorAll(".summary-doc-btn").forEach((btn) => {
                    btn.addEventListener("click", () => {
                        const docId = btn.getAttribute("data-id");
                        const docName = btn.getAttribute("data-name");
                        generateDocumentSummary(docId, docName);
                    });
                });

                // Attach MCQ handlers
                docLibraryList.querySelectorAll(".mcq-doc-btn").forEach((btn) => {
                    btn.addEventListener("click", () => {
                        const docId = btn.getAttribute("data-id");
                        const docName = btn.getAttribute("data-name");
                        generateDocumentMCQs(docId, docName);
                    });
                });

                // Attach Important Questions handlers
                docLibraryList.querySelectorAll(".questions-doc-btn").forEach((btn) => {
                    btn.addEventListener("click", () => {
                        const docId = btn.getAttribute("data-id");
                        const docName = btn.getAttribute("data-name");
                        generateImportantQuestions(docId, docName);
                    });
                });

                // Attach delete handlers
                docLibraryList.querySelectorAll(".delete-doc-btn").forEach((btn) => {
                    btn.addEventListener("click", () => {
                        const docId = btn.getAttribute("data-id");
                        const docName = btn.getAttribute("data-name");
                        deleteDocument(docId, docName);
                    });
                });
            }
        } catch (err) {
            console.error("Failed to load documents", err);
        }
    }

    async function selectDocument(docId, docName) {
        currentDocumentId = docId;
        currentDocumentName = docName;

        // Update active UI state in document library
        if (docLibraryList) {
            docLibraryList.querySelectorAll(".doc-item").forEach((item) => {
                item.classList.remove("active");
            });
            const activeItem = document.getElementById(`doc-item-${docId}`);
            if (activeItem) activeItem.classList.add("active");
        }

        // Update Metadata Header
        if (metaFilename) metaFilename.textContent = docName;
        if (metaDocId) metaDocId.textContent = `Document ID: ${docId}`;
        if (metaStatusMessage) metaStatusMessage.textContent = "Selected active document for study and Q&A.";

        // Show search and Q&A sections
        if (emptyResultsState) emptyResultsState.classList.add("hidden");
        if (resultsWrapper) resultsWrapper.classList.remove("hidden");
        if (searchSection) searchSection.classList.remove("hidden");

        // Fetch chat history for this document
        loadChatHistory(docId);
    }

    async function deleteDocument(docId, docName) {
        if (!confirm(`Are you sure you want to delete "${docName}"? This will remove its vector index and chat history.`)) {
            return;
        }

        try {
            const res = await fetch(`${API_BASE_URL}/api/documents/${docId}`, {
                method: "DELETE",
                credentials: "include"
            });
            const data = await res.json();

            if (res.ok && data.success) {
                if (currentDocumentId === docId) {
                    currentDocumentId = null;
                    currentDocumentName = null;
                    if (resultsWrapper) resultsWrapper.classList.add("hidden");
                    if (searchSection) searchSection.classList.add("hidden");
                    if (emptyResultsState) emptyResultsState.classList.remove("hidden");
                }
                loadUserDocuments();
            } else {
                alert(data.error || "Failed to delete document.");
            }
        } catch (err) {
            alert("Error connecting to server to delete document.");
        }
    }

    // -------------------------------------------------------------
    // 3. Chat History Loader
    // -------------------------------------------------------------
    const chatHistorySection = document.getElementById("chat-history-section");
    const chatHistoryWrapper = document.getElementById("chat-history-wrapper");
    const chatHistoryCount = document.getElementById("chat-history-count");

    async function loadChatHistory(docId) {
        if (!chatHistoryWrapper) return;

        try {
            const res = await fetch(`${API_BASE_URL}/api/chat/history?document_id=${docId}`, {
                method: "GET",
                credentials: "include"
            });
            const data = await res.json();

            if (res.ok && data.success) {
                const chats = data.chats || [];
                let allMessages = [];
                chats.forEach((c) => {
                    if (c.messages) allMessages = allMessages.concat(c.messages);
                });

                if (chatHistoryCount) chatHistoryCount.textContent = `${allMessages.length} Messages`;

                if (allMessages.length > 0) {
                    chatHistoryWrapper.innerHTML = "";
                    allMessages.forEach((msg) => {
                        appendMessageToHistory(msg.role, msg.content);
                    });
                    if (chatHistorySection) chatHistorySection.classList.remove("hidden");
                } else {
                    chatHistoryWrapper.innerHTML = '<p class="text-muted" style="font-size: 0.85rem; padding: 0.5rem 0;">No previous questions asked for this document.</p>';
                    if (chatHistorySection) chatHistorySection.classList.remove("hidden");
                }
            }
        } catch (err) {
            console.error("Failed to load chat history", err);
        }
    }

    function appendMessageToHistory(role, content) {
        if (!chatHistoryWrapper) return;
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-history-msg ${role}`;
        msgDiv.innerHTML = `
            <span class="msg-sender">${role === "user" ? "🧑 Student Question" : "🤖 AI Study Assistant"}</span>
            <div>${escapeHtml(content)}</div>
        `;
        chatHistoryWrapper.appendChild(msgDiv);
        chatHistoryWrapper.scrollTop = chatHistoryWrapper.scrollHeight;
    }

    // -------------------------------------------------------------
    // 4. PDF Upload, Text Extraction & FAISS Vector Index Handler
    // -------------------------------------------------------------
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("pdf-file-input");
    const uploadBtn = document.getElementById("upload-btn");
    const selectedFileInfo = document.getElementById("selected-file-info");
    const selectedFileName = document.getElementById("selected-file-name");
    const selectedFileSize = document.getElementById("selected-file-size");
    const removeFileBtn = document.getElementById("remove-file-btn");

    const statusBox = document.getElementById("status-box");
    const statusIcon = document.getElementById("status-icon");
    const statusTitle = document.getElementById("status-title");
    const statusDetails = document.getElementById("status-details");

    const emptyResultsState = document.getElementById("empty-results-state");
    const resultsWrapper = document.getElementById("results-wrapper");
    const statsBadges = document.getElementById("stats-badges");
    const badgePages = document.getElementById("badge-pages");
    const badgeChunks = document.getElementById("badge-chunks");
    const badgeEmbeddings = document.getElementById("badge-embeddings");
    const badgeFaiss = document.getElementById("badge-faiss");

    const metaFilename = document.getElementById("meta-filename");
    const metaDocId = document.getElementById("meta-doc-id");
    const metaStatusMessage = document.getElementById("meta-status-message");
    const chunksContainer = document.getElementById("chunks-container");
    const pagesContainer = document.getElementById("pages-container");
    const fullTextArea = document.getElementById("full-text-area");
    const copyTextBtn = document.getElementById("copy-text-btn");

    // Summary Section Elements
    const generateSummaryBtn = document.getElementById("generate-summary-btn");
    const generateSummaryBtnInner = document.getElementById("generate-summary-btn-inner");
    const summaryEmptyPrompt = document.getElementById("summary-empty-prompt");
    const summaryLoading = document.getElementById("summary-loading");
    const summaryContentWrapper = document.getElementById("summary-content-wrapper");
    const summaryTextBody = document.getElementById("summary-text-body");
    const summaryStrategyBadge = document.getElementById("summary-strategy-badge");
    const summaryPagesBadge = document.getElementById("summary-pages-badge");
    const summarySourcesContainer = document.getElementById("summary-sources-container");
    const summarySourcesList = document.getElementById("summary-sources-list");
    const copySummaryBtn = document.getElementById("copy-summary-btn");
    let currentSummaryText = "";

    // Search / Chat Section Elements
    const searchSection = document.getElementById("search-section");
    const searchQueryInput = document.getElementById("search-query-input");
    const searchBtn = document.getElementById("search-btn");
    const searchStatus = document.getElementById("search-status");
    const searchStatusIcon = document.getElementById("search-status-icon");
    const searchStatusText = document.getElementById("search-status-text");
    const aiAnswerWrapper = document.getElementById("ai-answer-wrapper");
    const aiAnswerText = document.getElementById("ai-answer-text");
    const aiSourcesContainer = document.getElementById("ai-sources-container");
    const aiSourcesList = document.getElementById("ai-sources-list");
    const searchResultsWrapper = document.getElementById("search-results-wrapper");
    const searchResultCount = document.getElementById("search-result-count");
    const searchResultsContainer = document.getElementById("search-results-container");

    let selectedFile = null;

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    function showStatus(type, title, details = "") {
        if (!statusBox) return;
        statusBox.className = `status-box ${type}`;
        statusTitle.textContent = title;
        statusDetails.textContent = details;

        if (type === "loading") {
            statusIcon.textContent = "⏳";
        } else if (type === "success") {
            statusIcon.textContent = "✅";
        } else if (type === "error") {
            statusIcon.textContent = "⚠️";
        }
    }

    function resetFileSelection() {
        selectedFile = null;
        if (fileInput) fileInput.value = "";
        if (selectedFileInfo) selectedFileInfo.classList.add("hidden");
        if (uploadBtn) uploadBtn.disabled = true;
    }

    function handleFile(file) {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showStatus("error", "Invalid File Format", "Please select a valid .pdf document.");
            resetFileSelection();
            return;
        }

        selectedFile = file;
        if (selectedFileName) selectedFileName.textContent = file.name;
        if (selectedFileSize) selectedFileSize.textContent = `(${formatBytes(file.size)})`;
        if (selectedFileInfo) selectedFileInfo.classList.remove("hidden");
        if (uploadBtn) uploadBtn.disabled = false;

        if (statusBox) statusBox.classList.add("hidden");
    }

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (removeFileBtn) {
        removeFileBtn.addEventListener("click", resetFileSelection);
    }

    // Upload Action
    if (uploadBtn) {
        uploadBtn.addEventListener("click", async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append("file", selectedFile);

            uploadBtn.disabled = true;
            showStatus("loading", "Processing Document...", `Extracting, chunking, and indexing "${selectedFile.name}".`);

            try {
                const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
                    method: "POST",
                    credentials: "include",
                    body: formData
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    showStatus("success", "Processing Complete", result.message);

                    currentDocumentId = result.document_id;
                    currentDocumentName = result.filename;

                    if (metaFilename) metaFilename.textContent = result.filename;
                    if (metaDocId) metaDocId.textContent = `Document ID: ${result.document_id}`;
                    if (metaStatusMessage) metaStatusMessage.textContent = result.message;

                    if (badgePages) badgePages.textContent = `${result.pages} Page(s)`;
                    if (badgeChunks) badgeChunks.textContent = `${result.total_chunks} Chunk(s)`;

                    if (badgeEmbeddings) {
                        if (result.embedding && result.embedding.success) {
                            badgeEmbeddings.textContent = `⚡ ${result.embedding.dimension}-dim Vectors`;
                            badgeEmbeddings.className = "badge badge-purple";
                            badgeEmbeddings.classList.remove("hidden");
                        } else {
                            badgeEmbeddings.classList.add("hidden");
                        }
                    }

                    if (badgeFaiss) {
                        if (result.vector_store && result.vector_store.created) {
                            badgeFaiss.textContent = `📦 FAISS Indexed (${result.vector_store.vectors} vectors)`;
                            badgeFaiss.className = "badge badge-accent";
                            badgeFaiss.classList.remove("hidden");
                        } else {
                            badgeFaiss.textContent = "FAISS Pending (.env)";
                            badgeFaiss.className = "badge badge-warning";
                            badgeFaiss.classList.remove("hidden");
                        }
                    }

                    if (statsBadges) statsBadges.classList.remove("hidden");

                    // Populate chunks
                    if (chunksContainer) {
                        chunksContainer.innerHTML = "";
                        if (result.chunks && result.chunks.length > 0) {
                            result.chunks.forEach((chunk) => {
                                const chunkCard = document.createElement("div");
                                chunkCard.className = "chunk-card";
                                chunkCard.innerHTML = `
                                    <div class="chunk-card-header">
                                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                                            <span class="chunk-badge">Chunk #${chunk.chunk_index}</span>
                                            <span>Page ${chunk.page}</span>
                                        </div>
                                        <span class="page-meta">${chunk.word_count} words • ${chunk.char_count} chars</span>
                                    </div>
                                    <div class="chunk-card-body">${escapeHtml(chunk.text)}</div>
                                `;
                                chunksContainer.appendChild(chunkCard);
                            });
                        }
                    }

                    // Populate full text
                    if (fullTextArea) {
                        fullTextArea.value = result.extracted_text || "(No extractable text)";
                    }

                    // Reveal Results & Search Area
                    if (emptyResultsState) emptyResultsState.classList.add("hidden");
                    if (resultsWrapper) resultsWrapper.classList.remove("hidden");
                    if (searchSection) searchSection.classList.remove("hidden");

                    // Refresh document library & chat history
                    loadUserDocuments();
                    loadChatHistory(currentDocumentId);

                } else {
                    if (response.status === 401) {
                        showStatus("error", "Authentication Required", "Please sign in to upload documents.");
                        setTimeout(() => { window.location.href = "login.html"; }, 1500);
                    } else {
                        showStatus("error", "Upload Failed", result.error || "Failed to process PDF file.");
                    }
                }
            } catch (err) {
                showStatus(
                    "error",
                    "Connection Error",
                    "Could not connect to Flask API. Make sure the backend server is running on http://127.0.0.1:5000"
                );
            } finally {
                uploadBtn.disabled = false;
            }
        });
    }

    // -------------------------------------------------------------
    // 5. AI RAG Chat Handler
    // -------------------------------------------------------------
    async function handleSemanticSearch() {
        if (!currentDocumentId) {
            alert("Please select or upload a document first.");
            return;
        }

        const query = searchQueryInput ? searchQueryInput.value.trim() : "";
        if (!query) {
            alert("Please enter a question.");
            return;
        }

        if (searchBtn) searchBtn.disabled = true;
        if (searchStatus) {
            searchStatus.className = "status-box loading";
            searchStatusText.textContent = `Retrieving context from FAISS & generating answer with Gemini...`;
            searchStatus.classList.remove("hidden");
        }
        if (aiAnswerWrapper) aiAnswerWrapper.classList.add("hidden");
        if (searchResultsWrapper) searchResultsWrapper.classList.add("hidden");

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                credentials: "include",
                body: JSON.stringify({
                    question: query,
                    document_id: currentDocumentId,
                    k: 4
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                if (searchStatus) searchStatus.classList.add("hidden");

                // Render latest AI Answer Card
                if (aiAnswerText) {
                    aiAnswerText.textContent = result.answer;
                }

                if (aiSourcesContainer && aiSourcesList) {
                    aiSourcesList.innerHTML = "";
                    if (result.sources && result.sources.length > 0) {
                        result.sources.forEach((src) => {
                            const tag = document.createElement("span");
                            tag.className = "source-tag";
                            tag.innerHTML = `📄 <strong>${escapeHtml(src.document)}</strong> (Page ${src.page})`;
                            aiSourcesList.appendChild(tag);
                        });
                        aiSourcesContainer.classList.remove("hidden");
                    } else {
                        aiSourcesContainer.classList.add("hidden");
                    }
                }

                if (aiAnswerWrapper) aiAnswerWrapper.classList.remove("hidden");

                // Append to Conversation History Thread
                appendMessageToHistory("user", query);
                appendMessageToHistory("assistant", result.answer);
                if (chatHistorySection) chatHistorySection.classList.remove("hidden");

                // Clear question input
                if (searchQueryInput) searchQueryInput.value = "";

                // Fetch context chunks for transparency
                try {
                    const searchRes = await fetch(`${API_BASE_URL}/api/documents/${currentDocumentId}/search`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        credentials: "include",
                        body: JSON.stringify({ query: query, k: 4 })
                    });
                    const searchData = await searchRes.json();
                    if (searchRes.ok && searchData.success && searchResultsContainer) {
                        searchResultsContainer.innerHTML = "";
                        if (searchResultCount) searchResultCount.textContent = `${searchData.results.length} Chunks`;
                        searchData.results.forEach((item, idx) => {
                            const card = document.createElement("div");
                            card.className = "retrieved-chunk-card";
                            card.innerHTML = `
                                <div class="retrieved-header">
                                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                                        <span class="badge badge-accent">Rank #${idx + 1}</span>
                                        <span>Page ${item.page} • Chunk #${item.chunk_index}</span>
                                    </div>
                                    <span class="score-badge">Distance: ${item.score}</span>
                                </div>
                                <div class="chunk-card-body">${escapeHtml(item.text)}</div>
                            `;
                            searchResultsContainer.appendChild(card);
                        });
                        if (searchResultsWrapper) searchResultsWrapper.classList.remove("hidden");
                    }
                } catch (chunkErr) {}

            } else {
                if (searchStatus) {
                    searchStatus.className = "status-box error";
                    searchStatusText.textContent = result.error || "Failed to generate answer.";
                }
            }
        } catch (err) {
            if (searchStatus) {
                searchStatus.className = "status-box error";
                searchStatusText.textContent = "Failed to connect to AI chat endpoint. Make sure the backend server is running.";
            }
        } finally {
            if (searchBtn) searchBtn.disabled = false;
        }
    }

    if (searchBtn) {
        searchBtn.addEventListener("click", handleSemanticSearch);
    }

    if (searchQueryInput) {
        searchQueryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                handleSemanticSearch();
            }
        });
    }

    function escapeHtml(text) {
        if (!text) return "";
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // Tab Switching
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetTabId = button.getAttribute("data-tab");

            tabButtons.forEach((btn) => btn.classList.remove("active"));
            button.classList.add("active");

            document.querySelectorAll(".tab-content").forEach((content) => {
                content.classList.remove("active");
            });

            const activeContent = document.getElementById(targetTabId);
            if (activeContent) {
                activeContent.classList.add("active");
            }
        });
    });

    // -------------------------------------------------------------
    // 5. AI Study Summary Generation & Rendering
    // -------------------------------------------------------------
    function renderMarkdown(md) {
        if (!md) return "";
        let html = escapeHtml(md);

        // Headers
        html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
        html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
        html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

        // Bold text **text**
        html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");

        // Unordered list items
        html = html.replace(/^\- (.*$)/gim, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");
        // Fix nested duplicate <ul> tags
        html = html.replace(/<\/ul>\s*<ul>/gim, "");

        // Convert double line breaks into paragraphs
        const paragraphs = html.split(/\n\n+/);
        html = paragraphs.map((p) => {
            p = p.trim();
            if (!p) return "";
            if (p.startsWith("<h") || p.startsWith("<ul>") || p.startsWith("<li>")) {
                return p;
            }
            return `<p>${p.replace(/\n/g, "<br>")}</p>`;
        }).join("\n");

        return html;
    }

    async function generateDocumentSummary(docId, docName) {
        const targetDocId = docId || currentDocumentId;
        const targetDocName = docName || currentDocumentName;

        if (!targetDocId) {
            alert("Please select or upload a document first.");
            return;
        }

        // Auto select if not current
        if (currentDocumentId !== targetDocId) {
            selectDocument(targetDocId, targetDocName);
        }

        // Switch to Summary Tab
        const summaryTabBtn = document.querySelector('.tab-btn[data-tab="summary-view"]');
        if (summaryTabBtn) summaryTabBtn.click();

        // UI Loading State
        if (summaryEmptyPrompt) summaryEmptyPrompt.classList.add("hidden");
        if (summaryContentWrapper) summaryContentWrapper.classList.add("hidden");
        if (summaryLoading) summaryLoading.classList.remove("hidden");
        if (generateSummaryBtn) generateSummaryBtn.disabled = true;
        if (generateSummaryBtnInner) generateSummaryBtnInner.disabled = true;

        try {
            const res = await fetch(`${API_BASE_URL}/api/documents/${targetDocId}/summary`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include"
            });
            const data = await res.json();

            if (res.ok && data.success) {
                currentSummaryText = data.summary || "";

                if (summaryTextBody) {
                    summaryTextBody.innerHTML = renderMarkdown(currentSummaryText);
                }

                if (summaryStrategyBadge) {
                    summaryStrategyBadge.textContent = data.strategy === "hierarchical" 
                        ? "Hierarchical Multi-Section" 
                        : "Direct Single-Pass";
                }

                if (summaryPagesBadge) {
                    summaryPagesBadge.textContent = `${data.total_pages || 0} Pages`;
                }

                if (summarySourcesContainer && summarySourcesList) {
                    summarySourcesList.innerHTML = "";
                    if (data.sources && data.sources.length > 0) {
                        data.sources.forEach((src) => {
                            const chip = document.createElement("span");
                            chip.className = "source-tag";
                            chip.innerHTML = `📄 <strong>${escapeHtml(src.document)}</strong> (Page ${src.page})`;
                            summarySourcesList.appendChild(chip);
                        });
                        summarySourcesContainer.classList.remove("hidden");
                    } else {
                        summarySourcesContainer.classList.add("hidden");
                    }
                }

                if (summaryLoading) summaryLoading.classList.add("hidden");
                if (summaryContentWrapper) summaryContentWrapper.classList.remove("hidden");
            } else {
                if (summaryLoading) summaryLoading.classList.add("hidden");
                if (summaryEmptyPrompt) summaryEmptyPrompt.classList.remove("hidden");
                alert(data.error || "Failed to generate study summary.");
            }
        } catch (err) {
            if (summaryLoading) summaryLoading.classList.add("hidden");
            if (summaryEmptyPrompt) summaryEmptyPrompt.classList.remove("hidden");
            alert("Error connecting to server for summary generation. Ensure the backend is running.");
        } finally {
            if (generateSummaryBtn) generateSummaryBtn.disabled = false;
            if (generateSummaryBtnInner) generateSummaryBtnInner.disabled = false;
        }
    }

    if (generateSummaryBtn) {
        generateSummaryBtn.addEventListener("click", () => {
            generateDocumentSummary(currentDocumentId, currentDocumentName);
        });
    }

    if (generateSummaryBtnInner) {
        generateSummaryBtnInner.addEventListener("click", () => {
            generateDocumentSummary(currentDocumentId, currentDocumentName);
        });
    }

    if (copySummaryBtn) {
        copySummaryBtn.addEventListener("click", async () => {
            if (!currentSummaryText) return;
            try {
                await navigator.clipboard.writeText(currentSummaryText);
                const originalText = copySummaryBtn.textContent;
                copySummaryBtn.textContent = "✅ Copied!";
                setTimeout(() => {
                    copySummaryBtn.textContent = originalText;
                }, 2000);
            } catch (err) {
                console.error("Failed to copy summary", err);
            }
        });
    }

    // -------------------------------------------------------------
    // 6. AI MCQ Generator & Interactive Quiz Component
    // -------------------------------------------------------------
    const generateMcqsBtnTop = document.getElementById("generate-mcqs-btn-top");
    const generateMcqsBtn = document.getElementById("generate-mcqs-btn");
    const mcqCountSelect = document.getElementById("mcq-count-select");
    const mcqDifficultySelect = document.getElementById("mcq-difficulty-select");
    const mcqsEmptyPrompt = document.getElementById("mcqs-empty-prompt");
    const mcqsLoading = document.getElementById("mcqs-loading");
    const quizWrapper = document.getElementById("quiz-wrapper");
    const quizProgressText = document.getElementById("quiz-progress-text");
    const quizDifficultyBadge = document.getElementById("quiz-difficulty-badge");
    const quizScoreDisplay = document.getElementById("quiz-score-display");
    const quizQuestionTitle = document.getElementById("quiz-question-title");
    const quizOptionsContainer = document.getElementById("quiz-options-container");
    const quizFeedbackBox = document.getElementById("quiz-feedback-box");
    const feedbackIcon = document.getElementById("feedback-icon");
    const feedbackVerdict = document.getElementById("feedback-verdict");
    const feedbackExplanation = document.getElementById("feedback-explanation");
    const feedbackSourceTag = document.getElementById("feedback-source-tag");
    const quizPrevBtn = document.getElementById("quiz-prev-btn");
    const quizNextBtn = document.getElementById("quiz-next-btn");
    const quizRestartBtn = document.getElementById("quiz-restart-btn");

    let currentQuizQuestions = [];
    let currentQuestionIndex = 0;
    let userAnswers = {}; // { questionIndex: selectedOptionKey }

    async function generateDocumentMCQs(docId, docName) {
        const targetDocId = docId || currentDocumentId;
        const targetDocName = docName || currentDocumentName;

        if (!targetDocId) {
            alert("Please select or upload a document first.");
            return;
        }

        if (currentDocumentId !== targetDocId) {
            selectDocument(targetDocId, targetDocName);
        }

        // Switch to MCQs tab
        const mcqsTabBtn = document.querySelector('.tab-btn[data-tab="mcqs-view"]');
        if (mcqsTabBtn) mcqsTabBtn.click();

        const count = mcqCountSelect ? parseInt(mcqCountSelect.value, 10) || 5 : 5;
        const difficulty = mcqDifficultySelect ? mcqDifficultySelect.value || "medium" : "medium";

        // Loading state
        if (mcqsEmptyPrompt) mcqsEmptyPrompt.classList.add("hidden");
        if (quizWrapper) quizWrapper.classList.add("hidden");
        if (mcqsLoading) mcqsLoading.classList.remove("hidden");
        if (generateMcqsBtn) generateMcqsBtn.disabled = true;
        if (generateMcqsBtnTop) generateMcqsBtnTop.disabled = true;

        try {
            const res = await fetch(`${API_BASE_URL}/api/documents/${targetDocId}/mcqs`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ count: count, difficulty: difficulty })
            });
            const data = await res.json();

            if (res.ok && data.success && data.questions && data.questions.length > 0) {
                currentQuizQuestions = data.questions;
                userAnswers = {};
                currentQuestionIndex = 0;

                if (quizDifficultyBadge) {
                    quizDifficultyBadge.textContent = (data.difficulty || "medium").toUpperCase();
                }

                if (mcqsLoading) mcqsLoading.classList.add("hidden");
                if (quizWrapper) quizWrapper.classList.remove("hidden");

                renderQuizQuestion(0);
            } else {
                if (mcqsLoading) mcqsLoading.classList.add("hidden");
                if (mcqsEmptyPrompt) mcqsEmptyPrompt.classList.remove("hidden");
                alert(data.error || "Failed to generate multiple-choice questions.");
            }
        } catch (err) {
            if (mcqsLoading) mcqsLoading.classList.add("hidden");
            if (mcqsEmptyPrompt) mcqsEmptyPrompt.classList.remove("hidden");
            alert("Error connecting to server for MCQ generation. Ensure the backend is running.");
        } finally {
            if (generateMcqsBtn) generateMcqsBtn.disabled = false;
            if (generateMcqsBtnTop) generateMcqsBtnTop.disabled = false;
        }
    }

    function calculateScore() {
        let correctCount = 0;
        let answeredCount = 0;
        currentQuizQuestions.forEach((q, idx) => {
            const ans = userAnswers[idx];
            if (ans !== undefined) {
                answeredCount++;
                if (ans === q.correct_answer) {
                    correctCount++;
                }
            }
        });
        return { correctCount, answeredCount, total: currentQuizQuestions.length };
    }

    function updateQuizScoreUI() {
        if (!quizScoreDisplay) return;
        const { correctCount, answeredCount, total } = calculateScore();
        quizScoreDisplay.textContent = `Score: ${correctCount} / ${total} (Answered ${answeredCount}/${total})`;
    }

    function renderQuizQuestion(index) {
        if (!currentQuizQuestions || currentQuizQuestions.length === 0) return;
        if (index < 0 || index >= currentQuizQuestions.length) return;

        currentQuestionIndex = index;
        const q = currentQuizQuestions[index];

        if (quizProgressText) {
            quizProgressText.textContent = `Question ${index + 1} of ${currentQuizQuestions.length}`;
        }

        updateQuizScoreUI();

        if (quizQuestionTitle) {
            quizQuestionTitle.textContent = `${index + 1}. ${q.question}`;
        }

        if (quizOptionsContainer) {
            quizOptionsContainer.innerHTML = "";
            const isAnswered = userAnswers[index] !== undefined;
            const selectedKey = userAnswers[index];

            const optionKeys = ["A", "B", "C", "D"];
            optionKeys.forEach((key) => {
                const optText = q.options[key] || "";
                if (!optText) return;

                const btn = document.createElement("button");
                btn.className = "mcq-option-btn";
                if (isAnswered) btn.classList.add("disabled");

                if (isAnswered) {
                    if (key === q.correct_answer) {
                        btn.classList.add("correct");
                    } else if (key === selectedKey) {
                        btn.classList.add("wrong");
                    }
                }

                btn.innerHTML = `
                    <span class="option-key-badge">${key}</span>
                    <span>${escapeHtml(optText)}</span>
                `;

                if (!isAnswered) {
                    btn.addEventListener("click", () => {
                        handleOptionSelect(key);
                    });
                }

                quizOptionsContainer.appendChild(btn);
            });
        }

        // Handle feedback banner
        if (userAnswers[index] !== undefined) {
            showFeedbackBanner(q, userAnswers[index]);
        } else {
            if (quizFeedbackBox) quizFeedbackBox.classList.add("hidden");
        }

        // Update nav buttons
        if (quizPrevBtn) quizPrevBtn.disabled = index === 0;
        if (quizNextBtn) {
            quizNextBtn.textContent = index === currentQuizQuestions.length - 1 ? "Finish / Summary" : "Next ▶";
        }
    }

    function handleOptionSelect(selectedKey) {
        if (userAnswers[currentQuestionIndex] !== undefined) return;

        userAnswers[currentQuestionIndex] = selectedKey;
        const q = currentQuizQuestions[currentQuestionIndex];

        renderQuizQuestion(currentQuestionIndex);
    }

    function showFeedbackBanner(q, selectedKey) {
        if (!quizFeedbackBox) return;

        const isCorrect = selectedKey === q.correct_answer;
        quizFeedbackBox.className = `quiz-feedback-box ${isCorrect ? "correct-verdict" : "wrong-verdict"}`;

        if (feedbackIcon) feedbackIcon.textContent = isCorrect ? "✅" : "❌";
        if (feedbackVerdict) {
            feedbackVerdict.textContent = isCorrect 
                ? "Correct!" 
                : `Incorrect. (Correct Answer: ${q.correct_answer})`;
        }

        if (feedbackExplanation) {
            feedbackExplanation.textContent = q.explanation || "No explanation provided.";
        }

        if (feedbackSourceTag && q.source) {
            feedbackSourceTag.innerHTML = `📄 <strong>${escapeHtml(q.source.document)}</strong> (Page ${q.source.page || 1})`;
            if (feedbackSourceTag.parentElement) {
                feedbackSourceTag.parentElement.classList.remove("hidden");
            }
        }

        quizFeedbackBox.classList.remove("hidden");
    }

    if (generateMcqsBtn) {
        generateMcqsBtn.addEventListener("click", () => {
            generateDocumentMCQs(currentDocumentId, currentDocumentName);
        });
    }

    if (generateMcqsBtnTop) {
        generateMcqsBtnTop.addEventListener("click", () => {
            generateDocumentMCQs(currentDocumentId, currentDocumentName);
        });
    }

    if (quizPrevBtn) {
        quizPrevBtn.addEventListener("click", () => {
            if (currentQuestionIndex > 0) {
                renderQuizQuestion(currentQuestionIndex - 1);
            }
        });
    }

    if (quizNextBtn) {
        quizNextBtn.addEventListener("click", () => {
            if (currentQuestionIndex < currentQuizQuestions.length - 1) {
                renderQuizQuestion(currentQuestionIndex + 1);
            } else {
                const { correctCount, total } = calculateScore();
                alert(`🎉 Quiz Completed!\nYour Final Score: ${correctCount} / ${total} (${Math.round((correctCount / total) * 100)}%)`);
            }
        });
    }

    if (quizRestartBtn) {
        quizRestartBtn.addEventListener("click", () => {
            userAnswers = {};
            renderQuizQuestion(0);
        });
    }

    // -------------------------------------------------------------
    // 7. AI Important Exam Questions Generator & Revision Workspace
    // -------------------------------------------------------------
    const generateQuestionsBtnTop = document.getElementById("generate-questions-btn-top");
    const generateQuestionsBtn = document.getElementById("generate-questions-btn");
    const questionCountSelect = document.getElementById("question-count-select");
    const questionTypeSelect = document.getElementById("question-type-select");
    const questionsEmptyPrompt = document.getElementById("questions-empty-prompt");
    const questionsLoading = document.getElementById("questions-loading");
    const questionsContentWrapper = document.getElementById("questions-content-wrapper");
    const questionsCountBadge = document.getElementById("questions-count-badge");
    const questionsTypeBadge = document.getElementById("questions-type-badge");
    const questionsListContainer = document.getElementById("questions-list-container");
    const copyQuestionsBtn = document.getElementById("copy-questions-btn");

    let currentQuestionsList = [];

    async function generateImportantQuestions(docId, docName) {
        const targetDocId = docId || currentDocumentId;
        const targetDocName = docName || currentDocumentName;

        if (!targetDocId) {
            alert("Please select or upload a document first.");
            return;
        }

        if (currentDocumentId !== targetDocId) {
            selectDocument(targetDocId, targetDocName);
        }

        // Switch to Important Questions Tab
        const questionsTabBtn = document.querySelector('.tab-btn[data-tab="questions-view"]');
        if (questionsTabBtn) questionsTabBtn.click();

        const count = questionCountSelect ? parseInt(questionCountSelect.value, 10) || 10 : 10;
        const qType = questionTypeSelect ? questionTypeSelect.value || "mixed" : "mixed";

        // Loading state
        if (questionsEmptyPrompt) questionsEmptyPrompt.classList.add("hidden");
        if (questionsContentWrapper) questionsContentWrapper.classList.add("hidden");
        if (questionsLoading) questionsLoading.classList.remove("hidden");
        if (generateQuestionsBtn) generateQuestionsBtn.disabled = true;
        if (generateQuestionsBtnTop) generateQuestionsBtnTop.disabled = true;

        try {
            const res = await fetch(`${API_BASE_URL}/api/documents/${targetDocId}/questions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ count: count, type: qType })
            });
            const data = await res.json();

            if (res.ok && data.success && data.questions && data.questions.length > 0) {
                currentQuestionsList = data.questions;

                if (questionsCountBadge) {
                    questionsCountBadge.textContent = `${data.total_questions || data.questions.length} Questions`;
                }
                if (questionsTypeBadge) {
                    questionsTypeBadge.textContent = (data.type || "mixed").toUpperCase();
                }

                renderQuestionsList(currentQuestionsList);

                if (questionsLoading) questionsLoading.classList.add("hidden");
                if (questionsContentWrapper) questionsContentWrapper.classList.remove("hidden");
            } else {
                if (questionsLoading) questionsLoading.classList.add("hidden");
                if (questionsEmptyPrompt) questionsEmptyPrompt.classList.remove("hidden");
                alert(data.error || "Failed to generate important questions.");
            }
        } catch (err) {
            if (questionsLoading) questionsLoading.classList.add("hidden");
            if (questionsEmptyPrompt) questionsEmptyPrompt.classList.remove("hidden");
            alert("Error connecting to server for questions generation. Ensure the backend is running.");
        } finally {
            if (generateQuestionsBtn) generateQuestionsBtn.disabled = false;
            if (generateQuestionsBtnTop) generateQuestionsBtnTop.disabled = false;
        }
    }

    function renderQuestionsList(questions) {
        if (!questionsListContainer) return;
        questionsListContainer.innerHTML = "";

        // Group by priority
        const priorityOrder = ["high", "medium", "low"];
        const priorityLabels = {
            high: { title: "High-Priority Revision Questions", icon: "🔴", badgeClass: "priority-high", desc: "Core exam topics and heavily tested principles" },
            medium: { title: "Medium-Priority Concepts", icon: "🟡", badgeClass: "priority-medium", desc: "Key supporting concepts, mechanisms, and processes" },
            low: { title: "Supplementary & Definition Points", icon: "🟢", badgeClass: "priority-low", desc: "Definitions, terminology, and secondary details" }
        };

        const grouped = { high: [], medium: [], low: [] };
        questions.forEach((q, idx) => {
            const imp = (q.importance || "medium").toLowerCase();
            if (grouped[imp]) {
                grouped[imp].push({ ...q, originalIndex: idx + 1 });
            } else {
                grouped["medium"].push({ ...q, originalIndex: idx + 1 });
            }
        });

        priorityOrder.forEach((pKey) => {
            const items = grouped[pKey];
            if (items && items.length > 0) {
                const groupConfig = priorityLabels[pKey];
                const section = document.createElement("div");
                section.className = `priority-group ${groupConfig.badgeClass}-group`;
                section.style.marginBottom = "1.5rem";

                section.innerHTML = `
                    <div class="priority-group-header" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.4rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.1rem;">${groupConfig.icon}</span>
                            <h4 style="margin: 0; font-size: 1rem; font-weight: 600;">${groupConfig.title}</h4>
                            <span class="badge ${pKey === 'high' ? 'badge-danger' : pKey === 'medium' ? 'badge-warning' : 'badge-info'}" style="font-size: 0.75rem;">${items.length}</span>
                        </div>
                        <span class="text-muted" style="font-size: 0.8rem;">${groupConfig.desc}</span>
                    </div>
                    <div class="priority-group-items" style="display: flex; flex-direction: column; gap: 0.75rem;">
                    </div>
                `;

                const itemsContainer = section.querySelector(".priority-group-items");

                items.forEach((q) => {
                    const card = document.createElement("div");
                    card.className = "question-card";
                    card.style.background = "var(--bg-primary)";
                    card.style.border = "1px solid var(--border-color)";
                    card.style.borderRadius = "var(--radius-md)";
                    card.style.padding = "1rem";

                    const typeLabels = {
                        short: "Short Answer",
                        long: "Long / Descriptive",
                        conceptual: "Conceptual & Analytical",
                        definition: "Definition & Terminology"
                    };
                    const typeDisplay = typeLabels[q.type] || q.type || "Short Answer";

                    const sourceDoc = q.source && q.source.document ? q.source.document : (currentDocumentName || "Document");
                    const sourcePage = q.source && q.source.page ? q.source.page : 1;

                    card.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem; gap: 0.5rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                                <span class="badge badge-purple" style="font-weight: 700; font-size: 0.8rem;">Q${q.originalIndex}</span>
                                <span class="badge badge-secondary" style="font-size: 0.75rem;">${typeDisplay}</span>
                                <span class="badge ${pKey === 'high' ? 'badge-danger' : pKey === 'medium' ? 'badge-warning' : 'badge-info'}" style="font-size: 0.75rem; text-transform: capitalize;">${pKey} Priority</span>
                            </div>
                            <span class="source-tag" style="font-size: 0.75rem;">📄 ${escapeHtml(sourceDoc)} (Page ${sourcePage})</span>
                        </div>
                        <p style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--text-primary); line-height: 1.4;">
                            ${escapeHtml(q.question)}
                        </p>
                        ${q.answer ? `
                            <div style="background: var(--bg-secondary); padding: 0.6rem 0.75rem; border-radius: var(--radius-sm); font-size: 0.85rem; color: var(--text-secondary); border-left: 3px solid var(--primary);">
                                <strong style="color: var(--text-primary); display: block; margin-bottom: 0.2rem;">💡 Answer Key / Key Points:</strong>
                                ${escapeHtml(q.answer)}
                            </div>
                        ` : ""}
                    `;

                    itemsContainer.appendChild(card);
                });

                questionsListContainer.appendChild(section);
            }
        });
    }

    if (generateQuestionsBtn) {
        generateQuestionsBtn.addEventListener("click", () => {
            generateImportantQuestions(currentDocumentId, currentDocumentName);
        });
    }

    if (generateQuestionsBtnTop) {
        generateQuestionsBtnTop.addEventListener("click", () => {
            generateImportantQuestions(currentDocumentId, currentDocumentName);
        });
    }

    if (copyQuestionsBtn) {
        copyQuestionsBtn.addEventListener("click", async () => {
            if (!currentQuestionsList || currentQuestionsList.length === 0) return;
            try {
                let formatted = `# Important Exam Questions - ${currentDocumentName || "Study Material"}\n\n`;
                const priorityOrder = ["high", "medium", "low"];
                const priorityTitles = {
                    high: "HIGH-PRIORITY REVISION QUESTIONS",
                    medium: "MEDIUM-PRIORITY CONCEPTS",
                    low: "SUPPLEMENTARY & DEFINITIONS"
                };

                priorityOrder.forEach((pKey) => {
                    const subset = currentQuestionsList.filter((q) => (q.importance || "medium").toLowerCase() === pKey);
                    if (subset.length > 0) {
                        formatted += `## ${priorityTitles[pKey]}\n\n`;
                        subset.forEach((q, idx) => {
                            formatted += `### Q${idx + 1}. ${q.question}\n`;
                            formatted += `- **Type**: ${q.type || "short"}\n`;
                            formatted += `- **Priority**: ${q.importance || "medium"}\n`;
                            if (q.source) {
                                formatted += `- **Source**: ${q.source.document || "Document"} (Page ${q.source.page || 1})\n`;
                            }
                            if (q.answer) {
                                formatted += `- **Answer Key / Key Points**: ${q.answer}\n`;
                            }
                            formatted += `\n`;
                        });
                    }
                });

                await navigator.clipboard.writeText(formatted);
                const originalText = copyQuestionsBtn.textContent;
                copyQuestionsBtn.textContent = "✅ Copied!";
                setTimeout(() => {
                    copyQuestionsBtn.textContent = originalText;
                }, 2000);
            } catch (err) {
                console.error("Failed to copy questions", err);
            }
        });
    }

    // -------------------------------------------------------------
    // Landing Page (index.html) Backend Health Check
    // -------------------------------------------------------------
    const checkApiBtn = document.getElementById("check-api-btn");
    const apiStatusCard = document.getElementById("api-status-card");
    const apiStatusText = document.getElementById("api-status-text");

    async function checkBackendHealth() {
        if (!apiStatusCard || !apiStatusText) return;
        apiStatusCard.classList.remove("active", "error");
        apiStatusText.textContent = "Connecting to backend API...";
        try {
            const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET" });
            if (res.ok) {
                const data = await res.json();
                apiStatusCard.classList.add("active");
                apiStatusText.textContent = `Backend Online: ${data.message || "Ready"}`;
            } else {
                apiStatusCard.classList.add("error");
                apiStatusText.textContent = `Backend returned HTTP ${res.status}`;
            }
        } catch (err) {
            apiStatusCard.classList.add("error");
            apiStatusText.textContent = "Backend Offline. Ensure Flask server is running.";
        }
    }

    if (checkApiBtn) {
        checkApiBtn.addEventListener("click", (e) => {
            e.preventDefault();
            checkBackendHealth();
        });
    }

    if (apiStatusCard) {
        checkBackendHealth();
    }

    // Initialize Auth Session Check on load
    checkAuthSession();
});

