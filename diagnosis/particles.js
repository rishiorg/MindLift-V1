// This is the background animation script that creates the floating particles
document.addEventListener('DOMContentLoaded', function() {
    // Create a scene for the particles
    const scene = new THREE.Scene();
    
    // Setup camera
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 20;
    
    // Create renderer and set its properties
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Add the renderer to our background animation container
    const container = document.getElementById('background-animation');
    container.appendChild(renderer.domElement);
    
    // Create particles
    const particlesCount = 100;
    const particles = new THREE.Group();
    scene.add(particles);
    
    // Create different particle geometries
    const geometries = [
      new THREE.SphereGeometry(0.5, 12, 12),
      new THREE.IcosahedronGeometry(0.5, 0),
      new THREE.ConeGeometry(0.3, 0.7, 8)
    ];
    
    // Create materials with different colors
    const materials = [
      new THREE.MeshBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.7 }),
      new THREE.MeshBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.5 }),
      new THREE.MeshBasicMaterial({ color: 0x4f46e5, transparent: true, opacity: 0.6 })
    ];
    
    // Create individual particles
    for (let i = 0; i < particlesCount; i++) {
      // Randomly select geometry and material
      const geomIndex = Math.floor(Math.random() * geometries.length);
      const matIndex = Math.floor(Math.random() * materials.length);
      
      const particle = new THREE.Mesh(geometries[geomIndex], materials[matIndex]);
      
      // Set random position
      const x = (Math.random() - 0.5) * 40;
      const y = (Math.random() - 0.5) * 40;
      const z = (Math.random() - 0.5) * 20;
      particle.position.set(x, y, z);
      
      // Set random rotation
      particle.rotation.x = Math.random() * Math.PI;
      particle.rotation.y = Math.random() * Math.PI;
      
      // Set random scale
      const scale = Math.random() * 0.5 + 0.5;
      particle.scale.set(scale, scale, scale);
      
      // Store animation properties
      particle.userData = {
        speed: Math.random() * 0.02 + 0.01,
        rotationSpeed: Math.random() * 0.01 - 0.005,
        direction: new THREE.Vector3(
          Math.random() * 0.01 - 0.005,
          Math.random() * 0.01 - 0.005,
          Math.random() * 0.01 - 0.005
        )
      };
      
      particles.add(particle);
    }
    
    // Resize handler
    function onWindowResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    window.addEventListener('resize', onWindowResize);
    
    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      
      // Update particle positions and rotations
      particles.children.forEach(particle => {
        // Move in the stored direction
        particle.position.x += particle.userData.direction.x;
        particle.position.y += particle.userData.direction.y;
        particle.position.z += particle.userData.direction.z;
        
        // Rotate particle
        particle.rotation.x += particle.userData.rotationSpeed;
        particle.rotation.y += particle.userData.rotationSpeed;
        
        // Boundaries check - if particle goes too far, reverse direction
        if (Math.abs(particle.position.x) > 20) particle.userData.direction.x *= -1;
        if (Math.abs(particle.position.y) > 20) particle.userData.direction.y *= -1;
        if (Math.abs(particle.position.z) > 10) particle.userData.direction.z *= -1;
      });
      
      // Rotate the entire particle group slowly
      particles.rotation.y += 0.0005;
      
      // Render scene
      renderer.render(scene, camera);
    }
    
    // Start animation
    animate();
  });