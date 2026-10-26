/**
 * Authentication management script
 * Handles user login, registration, validation, and session redirection
 */

const API_BASE_URL = window.API_BASE_URL !== undefined 
    ? window.API_BASE_URL 
    : (window.location.protocol.startsWith("http") ? "" : "http://127.0.0.1:5000");

document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");
    const toggleAuthBtn = document.getElementById("toggle-auth-btn");
    const toggleAuthText = document.getElementById("toggle-auth-text");
    const authTitle = document.getElementById("auth-title");
    const authSubtitle = document.getElementById("auth-subtitle");
    const authStatus = document.getElementById("auth-status");
    const authStatusText = document.getElementById("auth-status-text");
    const authStatusIcon = document.getElementById("auth-status-icon");

    let isRegisterMode = false;

    function showStatus(type, message) {
        if (!authStatus) return;
        authStatus.className = `status-box ${type}`;
        authStatusText.textContent = message;
        if (type === "error") {
            authStatusIcon.textContent = "⚠️";
        } else if (type === "success") {
            authStatusIcon.textContent = "✅";
        } else {
            authStatusIcon.textContent = "⏳";
        }
        authStatus.classList.remove("hidden");
    }

    function hideStatus() {
        if (authStatus) authStatus.classList.add("hidden");
    }

    // Toggle between Login and Registration views
    function toggleMode(e) {
        if (e) e.preventDefault();
        hideStatus();
        isRegisterMode = !isRegisterMode;

        if (isRegisterMode) {
            authTitle.textContent = "Create an Account";
            authSubtitle.textContent = "Join to start indexing and studying your lecture materials";
            loginForm.classList.add("hidden");
            registerForm.classList.remove("hidden");
            toggleAuthText.innerHTML = 'Already have an account? <a href="#" id="toggle-auth-btn">Sign In</a>';
        } else {
            authTitle.textContent = "Welcome Back";
            authSubtitle.textContent = "Sign in to access your study materials";
            registerForm.classList.add("hidden");
            loginForm.classList.remove("hidden");
            toggleAuthText.innerHTML = 'Don\'t have an account? <a href="#" id="toggle-auth-btn">Register</a>';
        }

        // Rebind listener to newly injected anchor
        const newToggleBtn = document.getElementById("toggle-auth-btn");
        if (newToggleBtn) {
            newToggleBtn.addEventListener("click", toggleMode);
        }
    }

    if (toggleAuthBtn) {
        toggleAuthBtn.addEventListener("click", toggleMode);
    }

    // Handle Login Submission
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideStatus();

            const identifier = document.getElementById("login-email").value.trim();
            const password = document.getElementById("login-password").value;

            if (!identifier || !password) {
                showStatus("error", "Please provide both email/username and password.");
                return;
            }

            showStatus("info", "Signing in...");

            try {
                const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        username_or_email: identifier,
                        password: password
                    }),
                    credentials: "include"
                });

                const data = await res.json();
                if (res.ok && data.success) {
                    showStatus("success", `Login successful! Redirecting...`);
                    setTimeout(() => {
                        window.location.href = "dashboard.html";
                    }, 800);
                } else {
                    showStatus("error", data.error || "Login failed. Please check your credentials.");
                }
            } catch (err) {
                showStatus("error", "Unable to connect to server. Ensure the backend is running.");
            }
        });
    }

    // Handle Register Submission
    if (registerForm) {
        registerForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideStatus();

            const username = document.getElementById("reg-username").value.trim();
            const email = document.getElementById("reg-email").value.trim();
            const password = document.getElementById("reg-password").value;

            if (!username || !email || !password) {
                showStatus("error", "All fields are required.");
                return;
            }

            if (username.length < 3) {
                showStatus("error", "Username must be at least 3 characters.");
                return;
            }

            if (password.length < 6) {
                showStatus("error", "Password must be at least 6 characters.");
                return;
            }

            showStatus("info", "Creating account...");

            try {
                const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        username: username,
                        email: email,
                        password: password
                    }),
                    credentials: "include"
                });

                const data = await res.json();
                if (res.ok && data.success) {
                    showStatus("success", "Account created successfully! Redirecting...");
                    setTimeout(() => {
                        window.location.href = "dashboard.html";
                    }, 800);
                } else {
                    showStatus("error", data.error || "Registration failed.");
                }
            } catch (err) {
                showStatus("error", "Unable to connect to server. Ensure the backend is running.");
            }
        });
    }
});
