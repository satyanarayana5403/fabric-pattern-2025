console.log("3D scene loaded");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    2000
);
camera.position.z = 1000;

const renderer = new THREE.CSS3DRenderer(); // ✅ USE THREE. HERE
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement); // ✅ Also THREE.

controls.minDistance = 500;
controls.maxDistance = 1500;

const style = document.createElement('style');
style.textContent = `
.label-panel {
    width: 800px;
    height: 600px;
    border: 2px solid #ccc;
    border-radius: 12px;
    box-shadow: 0 0 30px rgba(255,255,255,0.5);
    background: white;
}
body {
    margin: 0;
    overflow: hidden;
    background: #111;
}
`;
document.head.appendChild(style);

function createPanel(url, x, y, z) {
    const element = document.createElement('iframe');
    element.src = url;
    element.className = 'label-panel';
    element.allow = "camera; microphone";
    element.loading = "lazy";

    const object = new THREE.CSS3DObject(element); // ✅ USE THREE.
    object.position.set(x, y, z);
    scene.add(object);
}

createPanel('/home', -1000, 0, 0);
createPanel('/about', 1000, 0, 0);
createPanel('/contact', 0, -800, 0);
createPanel('/predict', 0, 800, 0);

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
