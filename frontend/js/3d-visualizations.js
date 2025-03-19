document.addEventListener('DOMContentLoaded', function() {
    // Initialize brain visualization modal
    initBrainModal();
    
    // Setup brain visualization
    setupBrainVisualization();
});

function initBrainModal() {
    const brainModal = document.getElementById('brain-modal');
    const closeBrainModal = document.getElementById('close-brain-modal');
    
    if (brainModal && closeBrainModal) {
        // Close modal when clicking the close button
        closeBrainModal.addEventListener('click', function() {
            brainModal.style.display = 'none';
        });
        
        // Close modal when clicking outside the content
        brainModal.addEventListener('click', function(e) {
            if (e.target === brainModal) {
                brainModal.style.display = 'none';
            }
        });
    }
}

function setupBrainVisualization() {
    const brainVisualization = document.getElementById('brain-visualization');
    
    if (!brainVisualization || typeof THREE === 'undefined') return;
    
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, brainVisualization.clientWidth / brainVisualization.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });
    
    renderer.setSize(brainVisualization.clientWidth, brainVisualization.clientHeight);
    brainVisualization.appendChild(renderer.domElement);
    
    // Create brain model
    const brainGroup = new THREE.Group();
    scene.add(brainGroup);
    
    // Brain hemispheres
    const leftHemisphere = createHemisphere(-0.5, 0x8b5cf6);
    const rightHemisphere = createHemisphere(0.5, 0x3b82f6);
    
    brainGroup.add(leftHemisphere);
    brainGroup.add(rightHemisphere);
    
    // Add neural connections
    addNeuralConnections(brainGroup);
    
    // Add brain regions
    addBrainRegions(brainGroup);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    camera.position.z = 8;
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate brain
        brainGroup.rotation.y += 0.005;
        
        renderer.render(scene, camera);
    }
    
    animate();
    
    // Handle window resize
    window.addEventListener('resize', function() {
        camera.aspect = brainVisualization.clientWidth / brainVisualization.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(brainVisualization.clientWidth, brainVisualization.clientHeight);
    });
    
    // Function to create hemisphere
    function createHemisphere(xOffset, color) {
        const geometry = new THREE.SphereGeometry(2, 32, 32, 0, Math.PI, 0, Math.PI);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.7,
            shininess: 30
        });
        
        const hemisphere = new THREE.Mesh(geometry, material);
        hemisphere.position.x = xOffset;
        
        return hemisphere;
    }
    
    // Function to add neural connections
    function addNeuralConnections(group) {
        const connectionCount = 50;
        const connectionMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.2
        });
        
        for (let i = 0; i < connectionCount; i++) {
            const points = [];
            
            // Random start point in left hemisphere
            const startPoint = new THREE.Vector3(
                -0.5 + (Math.random() - 0.5) * 1.5,
                (Math.random() - 0.5) * 1.5,
                (Math.random() - 0.5) * 1.5
            );
            
            // Random end point in right hemisphere
            const endPoint = new THREE.Vector3(
                0.5 + (Math.random() - 0.5) * 1.5,
                (Math.random() - 0.5) * 1.5,
                (Math.random() - 0.5) * 1.5
            );
            
            points.push(startPoint);
            points.push(endPoint);
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, connectionMaterial);
            
            group.add(line);
        }
    }
    
    // Function to add brain regions
    function addBrainRegions(group) {
        // Prefrontal cortex
        addBrainRegion(group, 0, 1.5, 0, 0.8, 0xf472b6); // Pink
        
        // Amygdala
        addBrainRegion(group, -0.8, -0.5, 0, 0.4, 0xef4444); // Red
        addBrainRegion(group, 0.8, -0.5, 0, 0.4, 0xef4444); // Red
        
        // Hippocampus
        addBrainRegion(group, -0.7, -1, 0, 0.5, 0x22c55e); // Green
        addBrainRegion(group, 0.7, -1, 0, 0.5, 0x22c55e); // Green
    }
    
    // Function to add a brain region
    function addBrainRegion(group, x, y, z, radius, color) {
        const geometry = new THREE.SphereGeometry(radius, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: 0.6,
            shininess: 30
        });
        
        const region = new THREE.Mesh(geometry, material);
        region.position.set(x, y, z);
        
        group.add(region);
    }
}

// Function to show brain visualization with specific emotion data
function showBrainVisualization(emotionData) {
    const brainModal = document.getElementById('brain-modal');
    
    if (brainModal) {
        // Update visualization based on emotion data
        updateBrainVisualization(emotionData);
        
        // Show modal
        brainModal.style.display = 'flex';
    }
}

// Function to update brain visualization based on emotion data
function updateBrainVisualization(emotionData) {
    // This would be implemented to update the colors and activity
    // of different brain regions based on the provided emotion data
    console.log('Updating brain visualization with:', emotionData);
    
    // For demo purposes, this is a placeholder
    // In a real implementation, this would modify the 3D brain model
    // to highlight different regions based on the emotional state
}