/**
 * Interactive Neural Knowledge Live-Wallpaper
 * Cinematic ambient background canvas with floating knowledge nodes,
 * connecting constellation filaments, and cursor responsiveness.
 * Optimized for 60fps performance and respects prefers-reduced-motion.
 */

(function () {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let animationFrameId = null;
    let isVisible = true;

    // Cursor position tracking with smooth dampening
    const mouse = {
        x: -1000,
        y: -1000,
        targetX: -1000,
        targetY: -1000,
        radius: 180
    };

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    window.addEventListener("resize", () => {
        resize();
        initParticles();
    });

    window.addEventListener("mousemove", (e) => {
        mouse.targetX = e.clientX;
        mouse.targetY = e.clientY;
    });

    window.addEventListener("mouseleave", () => {
        mouse.targetX = -1000;
        mouse.targetY = -1000;
    });

    // Particle Node System
    const particles = [];
    const PARTICLE_COUNT = Math.min(Math.floor(window.innerWidth / 22), 65);
    const CONNECT_DIST = 140;

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.45;
            this.vy = (Math.random() - 0.5) * 0.45;
            this.baseRadius = Math.random() * 1.8 + 0.8;
            this.radius = this.baseRadius;
            this.alpha = Math.random() * 0.4 + 0.2;
            // Palette: electric indigo, cyber cyan, soft lavender
            const colors = [
                "99, 102, 241",  // Indigo
                "6, 182, 212",   // Cyan
                "168, 85, 247",  // Purple
                "148, 163, 184"  // Slate
            ];
            this.color = colors[Math.floor(Math.random() * colors.length)];
            this.pulseSpeed = Math.random() * 0.02 + 0.008;
            this.pulseVal = Math.random() * Math.PI * 2;
        }

        update() {
            this.pulseVal += this.pulseSpeed;
            this.radius = this.baseRadius + Math.sin(this.pulseVal) * 0.5;

            // Natural movement
            this.x += this.vx;
            this.y += this.vy;

            // Bounce off edges
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            // Gentle cursor repulsion
            const dx = this.x - mouse.x;
            const dy = this.y - mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < mouse.radius && dist > 0) {
                const force = (1 - dist / mouse.radius) * 0.8;
                this.x += (dx / dist) * force;
                this.y += (dy / dist) * force;
            }
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${this.color}, ${this.alpha})`;
            ctx.shadowBlur = 10;
            ctx.shadowColor = `rgba(${this.color}, 0.6)`;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    function initParticles() {
        particles.length = 0;
        const count = Math.min(Math.floor(width / 24), 70);
        for (let i = 0; i < count; i++) {
            particles.push(new Particle());
        }
    }

    function drawFilaments() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < CONNECT_DIST) {
                    const alpha = (1 - dist / CONNECT_DIST) * 0.16;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(129, 140, 248, ${alpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        if (!isVisible) return;

        // Smooth mouse dampening
        mouse.x += (mouse.targetX - mouse.x) * 0.08;
        mouse.y += (mouse.targetY - mouse.y) * 0.08;

        ctx.clearRect(0, 0, width, height);

        // Draw connections
        drawFilaments();

        // Update and draw particles
        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();
        }

        animationFrameId = requestAnimationFrame(animate);
    }

    // Initialize
    resize();
    initParticles();

    if (!prefersReducedMotion) {
        animate();
    } else {
        // Static frame for reduced motion preference
        ctx.clearRect(0, 0, width, height);
        drawFilaments();
        for (const p of particles) p.draw();
    }

    // Visibility management
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            isVisible = false;
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
        } else {
            isVisible = true;
            if (!prefersReducedMotion) animate();
        }
    });
})();
