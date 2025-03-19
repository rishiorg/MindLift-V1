document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles.js background
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: "#8b5cf6" },
                shape: { type: "circle" },
                opacity: { value: 0.2, random: true },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: "#8b5cf6",
                    opacity: 0.1,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 1,
                    direction: "none",
                    random: true,
                    out_mode: "out"
                }
            },
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: { enable: true, mode: "grab" },
                    onclick: { enable: false },
                    resize: true
                },
                modes: {
                    grab: { distance: 140, line_linked: { opacity: 0.3 } }
                }
            }
        });
    }

    // Initialize Three.js background
    initThreeJSBackground();

    // Initialize 3D Brain Model
    initBrainModel();

    // Initialize Navigation
    initNavigation();

    // Initialize mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Animate sections on scroll
    animateSectionsOnScroll();
});

function initThreeJSBackground() {
    const canvas = document.getElementById('bg-canvas');
    
    if (!canvas || typeof THREE === 'undefined') return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        alpha: true,
        antialias: true
    });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Create gradient background
    const vertexShader = `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

    const fragmentShader = `
        varying vec2 vUv;
        uniform float time;
        
        void main() {
            vec2 uv = vUv;
            
            float dist = length(uv - vec2(0.5, 0.5));
            
            vec3 color1 = vec3(0.1, 0.0, 0.2);
            vec3 color2 = vec3(0.05, 0.0, 0.1);
            
            float noise = sin(uv.x * 10.0 + time * 0.2) * sin(uv.y * 10.0 + time * 0.3) * 0.05;
            
            float mixFactor = smoothstep(0.3, 0.7, dist + noise);
            vec3 color = mix(color1, color2, mixFactor);
            
            gl_FragColor = vec4(color, 1.0);
        }
    `;

    const uniforms = {
        time: { value: 0 }
    };

    const material = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    const plane = new THREE.Mesh(geometry, material);
    scene.add(plane);

    // Add floating particles
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 200;
    
    const positions = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    
    for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = (Math.random() - 0.5) * 10;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 10 - 5;
        
        sizes[i] = Math.random() * 0.05 + 0.01;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    
    const particleMaterial = new THREE.PointsMaterial({
        size: 0.1,
        color: 0x8b5cf6,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    camera.position.z = 1;

    // Animation loop
    function animate(time) {
        requestAnimationFrame(animate);
        
        uniforms.time.value = time * 0.001;
        
        // Rotate particles
        particles.rotation.x += 0.0003;
        particles.rotation.y += 0.0005;
        
        renderer.render(scene, camera);
    }
    
    animate();

    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

function initBrainModel() {
    const brainContainer = document.getElementById('brain-model');
    
    if (!brainContainer || typeof THREE === 'undefined') return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, brainContainer.clientWidth / brainContainer.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true
    });
    
    renderer.setSize(brainContainer.clientWidth, brainContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    brainContainer.appendChild(renderer.domElement);

    // Create a spherical brain representation
    const geometry = new THREE.SphereGeometry(5, 32, 32);
    
    // Create a shader material for the brain
    const brainVertexShader = `
        varying vec3 vNormal;
        varying vec3 vPosition;
        
        void main() {
            vNormal = normalize(normalMatrix * normal);
            vPosition = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

    const brainFragmentShader = `
        varying vec3 vNormal;
        varying vec3 vPosition;
        uniform float time;
        
        float noise(vec3 p) {
            return fract(sin(dot(p, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
        }
        
        void main() {
            // Base color - purple to blue gradient
            vec3 color1 = vec3(0.545, 0.361, 0.965); // Purple
            vec3 color2 = vec3(0.231, 0.51, 0.965); // Blue
            
            // Create a pattern based on position and time
            float pattern = noise(vPosition * 0.5 + time * 0.05);
            
            // Add pulsing effect
            float pulse = 0.5 + 0.5 * sin(time * 0.5);
            
            // Mix colors based on pattern and pulse
            vec3 finalColor = mix(color1, color2, pattern * pulse);
            
            // Add rim lighting
            float rimLight = 1.0 - max(0.0, dot(vNormal, vec3(0.0, 0.0, 1.0)));
            rimLight = pow(rimLight, 3.0) * 1.5;
            
            finalColor += vec3(0.6, 0.7, 1.0) * rimLight;
            
            // Add subtle pulsing glow
            finalColor += vec3(0.4, 0.3, 0.8) * 0.2 * pulse;
            
            // Set final color with transparency
            gl_FragColor = vec4(finalColor, 0.9);
        }
    `;

    const uniforms = {
        time: { value: 0 }
    };

    const material = new THREE.ShaderMaterial({
        vertexShader: brainVertexShader,
        fragmentShader: brainFragmentShader,
        uniforms: uniforms,
        transparent: true
    });

    const brain = new THREE.Mesh(geometry, material);
    scene.add(brain);

    // Add neural connections (lines)
    const neuronCount = 100;
    const neurons = [];
    const neuronMaterial = new THREE.LineBasicMaterial({ 
        color: 0x8b5cf6,
        transparent: true,
        opacity: 0.3
    });

    for (let i = 0; i < neuronCount; i++) {
        const points = [];
        const startPoint = new THREE.Vector3(
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 8
        );
        
        points.push(startPoint);
        
        // Create a random endpoint
        const endPoint = new THREE.Vector3(
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 8,
            (Math.random() - 0.5) * 8
        );
        
        points.push(endPoint);
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, neuronMaterial);
        scene.add(line);
        neurons.push({
            line: line,
            life: Math.random() * 2 + 1
        });
    }

    // Add synapse points
    const synapseGeometry = new THREE.SphereGeometry(0.1, 8, 8);
    const synapseMaterial = new THREE.MeshBasicMaterial({ 
        color: 0xffffff,
        transparent: true,
        opacity: 0.8
    });

    for (let i = 0; i < 30; i++) {
        const synapse = new THREE.Mesh(synapseGeometry, synapseMaterial);
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        synapse.position.x = 5 * Math.sin(phi) * Math.cos(theta);
        synapse.position.y = 5 * Math.sin(phi) * Math.sin(theta);
        synapse.position.z = 5 * Math.cos(phi);
        
        scene.add(synapse);
    }

    camera.position.z = 15;

    // Animation loop
    function animate(time) {
        requestAnimationFrame(animate);
        
        const t = time * 0.001;
        uniforms.time.value = t;
        
        // Rotate brain
        brain.rotation.y = t * 0.2;
        brain.rotation.z = Math.sin(t * 0.1) * 0.1;
        
        // Update neural connections
        neurons.forEach((neuron, index) => {
            neuron.life -= 0.01;
            
            if (neuron.life <= 0) {
                // Reset neuron
                const points = [];
                const startPoint = new THREE.Vector3(
                    (Math.random() - 0.5) * 8,
                    (Math.random() - 0.5) * 8,
                    (Math.random() - 0.5) * 8
                );
                
                points.push(startPoint);
                
                const endPoint = new THREE.Vector3(
                    (Math.random() - 0.5) * 8,
                    (Math.random() - 0.5) * 8,
                    (Math.random() - 0.5) * 8
                );
                
                points.push(endPoint);
                
                neuron.line.geometry.setFromPoints(points);
                neuron.life = Math.random() * 2 + 1;
            }
            
            // Pulse opacity
            neuron.line.material.opacity = 0.3 * (0.5 + 0.5 * Math.sin(t * 2 + index));
        });
        
        renderer.render(scene, camera);
    }
    
    animate();

    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = brainContainer.clientWidth / brainContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(brainContainer.clientWidth, brainContainer.clientHeight);
    });
}

function initNavigation() {
    // Smooth scroll for navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('data-section');
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                window.scrollTo({
                    top: targetSection.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Set active link
                document.querySelectorAll('.nav-link').forEach(navLink => {
                    navLink.classList.remove('active');
                });
                this.classList.add('active');
            }
        });
    });
}

function animateSectionsOnScroll() {
    // Use GSAP for scroll animations if available
    if (typeof gsap !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
        
        // Animate feature cards
        gsap.utils.toArray('.feature-card').forEach((card, i) => {
            gsap.from(card, {
                y: 50,
                opacity: 0,
                duration: 0.8,
                ease: "power2.out",
                scrollTrigger: {
                    trigger: card,
                    start: "top 80%",
                    toggleActions: "play none none none"
                },
                delay: i * 0.1
            });
        });
        
        // Animate team cards
        gsap.utils.toArray('.team-card').forEach((card, i) => {
            gsap.from(card, {
                y: 50,
                opacity: 0,
                duration: 0.8,
                ease: "power2.out",
                scrollTrigger: {
                    trigger: card,
                    start: "top 80%",
                    toggleActions: "play none none none"
                },
                delay: i * 0.1
            });
        });
    }
}