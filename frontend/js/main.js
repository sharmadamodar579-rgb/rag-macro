/**
 * Main application client script
 * Handles API health-check and global utility events
 */

const API_BASE_URL = "http://127.0.0.1:5000";

document.addEventListener("DOMContentLoaded", () => {
    const statusCard = document.getElementById("api-status-card");
    const statusText = document.getElementById("api-status-text");
    const checkBtn = document.getElementById("check-api-btn");
    const statusDot = statusCard ? statusCard.querySelector(".status-dot") : null;

    async function checkBackendHealth() {
        if (!statusText) return;
        statusText.textContent = "Connecting to backend...";
        if (statusDot) {
            statusDot.className = "status-dot";
        }

        try {
            const response = await fetch(`${API_BASE_URL}/`, {
                method: "GET",
                headers: {
                    "Content-Type": "application/json"
                }
            });

            if (response.ok) {
                const data = await response.json();
                statusText.textContent = `Backend Online: "${data.message}"`;
                if (statusDot) {
                    statusDot.className = "status-dot online";
                }
            } else {
                throw new Error(`Server returned HTTP ${response.status}`);
            }
        } catch (error) {
            statusText.textContent = "Backend Offline (Ensure Flask is running on port 5000)";
            if (statusDot) {
                statusDot.className = "status-dot offline";
            }
        }
    }

    if (checkBtn) {
        checkBtn.addEventListener("click", checkBackendHealth);
    }

    // Run initial health check on page load if status elements exist
    if (statusCard) {
        checkBackendHealth();
    }
});
