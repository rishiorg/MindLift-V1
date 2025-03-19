// Main JavaScript file for global functionality

document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    
    if (burger) {
        burger.addEventListener('click', () => {
            nav.classList.toggle('nav-active');
            
            // Burger Animation
            burger.classList.toggle('toggle');
        });
    }
    
    // Testimonial Slider
    const testimonials = document.querySelectorAll('.testimonial-card');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    
    if (testimonials.length > 0) {
        let currentTestimonial = 0;
        
        // Show first testimonial
        testimonials[currentTestimonial].classList.add('active');
        
        // Next button click
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                testimonials[currentTestimonial].classList.remove('active');
                currentTestimonial = (currentTestimonial + 1) % testimonials.length;
                testimonials[currentTestimonial].classList.add('active');
            });
        }
        
        // Previous button click
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                testimonials[currentTestimonial].classList.remove('active');
                currentTestimonial = (currentTestimonial - 1 + testimonials.length) % testimonials.length;
                testimonials[currentTestimonial].classList.add('active');
            });
        }
        
        // Auto rotate testimonials
        setInterval(() => {
            if (document.visibilityState === 'visible' && testimonials.length > 1) {
                testimonials[currentTestimonial].classList.remove('active');
                currentTestimonial = (currentTestimonial + 1) % testimonials.length;
                testimonials[currentTestimonial].classList.add('active');
            }
        }, 8000);
    }
    
    // Theme Toggle
    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
        // Check for saved theme preference or use device preference
        const savedTheme = localStorage.getItem('theme') || 
            (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
            themeSwitch.checked = true;
        }
        
        themeSwitch.addEventListener('change', () => {
            if (themeSwitch.checked) {
                document.body.classList.add('dark-theme');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-theme');
                localStorage.setItem('theme', 'light');
            }
        });
    }
});