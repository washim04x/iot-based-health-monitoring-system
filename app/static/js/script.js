// Create animated background particles
function createParticles() {
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 15 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        document.body.appendChild(particle);
    }
}

// Form validation and user experience enhancements
function setupFormHandling() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const submitBtn = document.querySelector('.submit-btn');
            submitBtn.innerHTML = '⏳ Analyzing...';
            submitBtn.style.opacity = '0.7';
        });
    }
}

// Auto-scroll to result after prediction
function scrollToResult() {
    const resultCard = document.querySelector('.result-card');
    if (resultCard && window.hasPrediction) {
        setTimeout(() => {
            window.scrollTo({
                top: resultCard.offsetTop - 100,
                behavior: 'smooth'
            });
        }, 100);
    }
}

// Initialize all functions when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    createParticles();
    setupFormHandling();
    scrollToResult();
});
