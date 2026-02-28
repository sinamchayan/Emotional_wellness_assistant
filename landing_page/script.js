document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.classList.add('js-enabled');
    // 3D Glass Card Tilt Effect for the hero text
    const card = document.querySelector('.glass-card');

    // Feature check to avoid mobile jittering
    const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

    if (!isTouchDevice && card) {
        // Tilt effect when mouse moves over the whole window
        window.addEventListener('mousemove', (e) => {
            const xAxis = (window.innerWidth / 2 - e.pageX) / 40;
            const yAxis = (window.innerHeight / 2 - e.pageY) / 40;
            card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
        });
    }

    // Interactive Image Tracking
    const images = document.querySelectorAll('.image-wrapper');

    if (!isTouchDevice) {
        window.addEventListener('mousemove', (e) => {
            images.forEach((img, index) => {
                const speed = (index + 1) * 20; // different speed for each image
                const x = (window.innerWidth - e.pageX * speed) / 1000;
                const y = (window.innerHeight - e.pageY * speed) / 1000;

                // Add a subtle translation based on mouse movement on top of CSS floating
                img.style.transform = `translateX(${x}px) translateY(${y}px)`;
            });
        });
    }

    // Magnetic Buttons Interaction
    const magneticBtns = document.querySelectorAll('.magnetic');

    magneticBtns.forEach(btn => {
        if (!isTouchDevice) {
            btn.addEventListener('mousemove', (e) => {
                const position = btn.getBoundingClientRect();
                const x = e.pageX - position.left - position.width / 2;
                // scroll adjustment 
                const y = e.pageY - window.scrollY - position.top - position.height / 2;

                const strength = btn.dataset.strength || 20;

                btn.style.transform = `translate(${x / strength}px, ${y / strength}px)`;
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'translate(0px, 0px)';
            });
        }
    });

    // Parallax effect on scroll for navbar blur dynamically
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(3, 3, 5, 0.7)';
            navbar.style.borderBottom = '1px solid rgba(255, 255, 255, 0.1)';
        } else {
            navbar.style.background = 'rgba(3, 3, 5, 0.5)';
            navbar.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
        }
    });

    // ── Scroll Reveal ──────────────────────────────────────────────
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Honour any animation-delay set inline on the element
                const delay = entry.target.style.animationDelay || '0s';
                const ms = parseFloat(delay) * 1000;
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, ms);
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });

    document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

    // ── Smooth scroll for anchor nav links ────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                const offset = 90; // account for fixed navbar height
                const top = target.getBoundingClientRect().top + window.scrollY - offset;
                window.scrollTo({ top, behavior: 'smooth' });
            }
        });
    });
});

