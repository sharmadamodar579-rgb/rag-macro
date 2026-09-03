/**
 * Chat and Document Interaction script placeholder
 * Will handle sending queries to RAG service, rendering answers, summaries, and MCQs
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const uploadDropzone = document.getElementById("upload-dropzone");
    const pdfFileInput = document.getElementById("pdf-file-input");

    // Placeholder upload click handler
    if (uploadDropzone && pdfFileInput) {
        uploadDropzone.addEventListener("click", () => {
            pdfFileInput.click();
        });

        pdfFileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                const fileName = e.target.files[0].name;
                alert(`Selected file: "${fileName}". Document processing and RAG pipeline will be implemented in future steps.`);
            }
        });
    }

    // Placeholder send message handler
    function handleSend() {
        const text = chatInput.value.trim();
        if (!text) return;

        alert(`Query submitted: "${text}". RAG & Gemini response pipeline will be connected in future steps.`);
        chatInput.value = "";
    }

    if (sendBtn) {
        sendBtn.addEventListener("click", handleSend);
    }

    if (chatInput) {
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                handleSend();
            }
        });
    }
});
