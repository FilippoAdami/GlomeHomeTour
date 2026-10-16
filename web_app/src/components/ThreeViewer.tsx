// src/components/ThreeViewer.tsx
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { API_BASE, SemanticCategory, ObjectInstance } from '../lib/api';

interface Props {
  meshUrl?: string;
  splatUrl?: string;
  initialMode?: 'mesh' | 'splat';
  categories?: SemanticCategory[];
  selectedCategory?: string | null;
  focusedInstance?: ObjectInstance | null;
  onCameraUpdate?: (pos: { x: number; y: number; z: number; rotation: number }) => void;
}

export function ThreeViewer({
  meshUrl,
  splatUrl,
  initialMode = 'mesh',
  categories = [],
  selectedCategory,
  focusedInstance,
  onCameraUpdate,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const splatContainerRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<'mesh' | 'splat'>(initialMode);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingProgress, setLoadingProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshObjectRef = useRef<THREE.Mesh | null>(null);
  const pointsObjectRef = useRef<THREE.Points | null>(null);
  const markersGroupRef = useRef<THREE.Group | null>(null);

  // Setup Three.js scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.05,
      1000
    );
    camera.position.set(0, 1.6, 3);
    cameraRef.current = camera;

    // Markers group for bounding highlights
    const markersGroup = new THREE.Group();
    scene.add(markersGroup);
    markersGroupRef.current = markersGroup;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.1;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI - 0.05;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(5, 10, 7);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // Floor grid
    const gridHelper = new THREE.GridHelper(20, 40, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -1.2;
    scene.add(gridHelper);

    // Resize handler
    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Render loop
    let reqId: number;
    const animate = () => {
      reqId = requestAnimationFrame(animate);
      controls.update();

      if (onCameraUpdate) {
        onCameraUpdate({
          x: camera.position.x,
          y: camera.position.y,
          z: camera.position.z,
          rotation: camera.rotation.y,
        });
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(reqId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Load Model based on selected mode
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    setLoading(true);
    setLoadingProgress(0);
    setError(null);

    // Cleanup previous models
    if (meshObjectRef.current) {
      scene.remove(meshObjectRef.current);
      meshObjectRef.current.geometry.dispose();
      meshObjectRef.current = null;
    }
    if (pointsObjectRef.current) {
      scene.remove(pointsObjectRef.current);
      pointsObjectRef.current.geometry.dispose();
      pointsObjectRef.current = null;
    }

    const targetUrl = mode === 'splat' ? splatUrl : meshUrl;

    if (!targetUrl) {
      setError('No 3D asset URL provided');
      setLoading(false);
      return;
    }

    const resolvedUrl = targetUrl.startsWith('http')
      ? targetUrl
      : `${API_BASE}${targetUrl}`;

    if (mode === 'splat') {
      let viewer: import('@mkkellogg/gaussian-splats-3d').Viewer | undefined;
      let cancelled = false;
      import('@mkkellogg/gaussian-splats-3d').then(async ({ Viewer }) => {
        if (!splatContainerRef.current || cancelled) return;
        const assetDirectory = resolvedUrl.slice(0, resolvedUrl.lastIndexOf('/'));
        const transformResponse = await fetch(`${assetDirectory}/splat_transform.json`);
        const transform = transformResponse.ok ? await transformResponse.json() : {};
        if (cancelled) return;
        viewer = new Viewer({
          rootElement: splatContainerRef.current,
          sharedMemoryForWorkers: false,
          ignoreDevicePixelRatio: window.matchMedia('(pointer: coarse)').matches,
        });
        try {
          await viewer.addSplatScene(resolvedUrl, {
            showLoadingUI: false,
            position: transform.position,
            rotation: transform.rotation,
          });
          if (!cancelled) {
            viewer.start();
            setLoading(false);
          }
        } catch (err) {
          if (!cancelled) {
            console.error('Error loading Gaussian splat:', err);
            setError('Failed to load Gaussian splat.');
            setLoading(false);
          }
        }
      }).catch((err: unknown) => {
        if (!cancelled) {
          console.error('Error initializing Gaussian splat viewer:', err);
          setError('Failed to load Gaussian splat viewer.');
          setLoading(false);
        }
      });
      return () => {
        cancelled = true;
        void viewer?.dispose();
      };
    }

    const isObj = resolvedUrl.endsWith('.obj');

    const onGeometryReady = (geometry: THREE.BufferGeometry) => {
      geometry.computeVertexNormals();
      geometry.center();

      const hasColors = Boolean(geometry.attributes.color);

      if (!geometry.index) {
        const material = new THREE.PointsMaterial({
          size: 0.035,
          vertexColors: hasColors,
          color: hasColors ? 0xffffff : 0x38bdf8,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.95,
        });
        const points = new THREE.Points(geometry, material);
        points.rotation.x = -Math.PI / 2;
        scene.add(points);
        pointsObjectRef.current = points;
      } else {
        const material = new THREE.MeshStandardMaterial({
          vertexColors: hasColors,
          color: hasColors ? 0xffffff : 0xcbd5e1,
          roughness: 0.6,
          metalness: 0.1,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        scene.add(mesh);
        meshObjectRef.current = mesh;
      }

      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }

      setLoading(false);
    };

    if (isObj) {
      import('three-stdlib').then(({ OBJLoader }) => {
        const objLoader = new OBJLoader();
        objLoader.load(
          resolvedUrl,
          (group) => {
            let foundMesh: THREE.Mesh | null = null;
            group.traverse((child) => {
              if ('isMesh' in child && (child as THREE.Mesh).isMesh && !foundMesh) {
                foundMesh = child as THREE.Mesh;
              }
            });
            if (foundMesh && (foundMesh as THREE.Mesh).geometry) {
              onGeometryReady((foundMesh as THREE.Mesh).geometry);
            } else {
              setError('No valid geometry in OBJ file');
              setLoading(false);
            }
          },
          (xhr) => {
            if (xhr.lengthComputable && xhr.total > 0) {
              setLoadingProgress(Math.round((xhr.loaded / xhr.total) * 100));
            }
          },
          (err) => {
            console.error('Error loading OBJ:', err);
            setError('Failed to load OBJ model.');
            setLoading(false);
          }
        );
      });
    } else {
      const loader = new PLYLoader();
      loader.load(
        resolvedUrl,
        onGeometryReady,
        (xhr) => {
          if (xhr.lengthComputable && xhr.total > 0) {
            setLoadingProgress(Math.round((xhr.loaded / xhr.total) * 100));
          }
        },
        (err) => {
          console.error('Error loading PLY:', err);
          setError('Failed to load 3D model.');
          setLoading(false);
        }
      );
    }
  }, [mode, meshUrl, splatUrl]);

  // Update 3D Semantic Markers when category is selected
  useEffect(() => {
    const group = markersGroupRef.current;
    if (!group) return;

    // Clear previous markers
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if ('geometry' in child) (child as THREE.Mesh).geometry.dispose();
    }

    if (!selectedCategory) return;

    const cat = categories.find((c) => c.id === selectedCategory);
    if (!cat) return;

    const color = new THREE.Color(cat.color);

    cat.instances.forEach((inst) => {
      // 3D bounding box indicator
      const boxGeo = new THREE.BoxGeometry(0.8, 0.7, 0.8);
      const wireGeo = new THREE.WireframeGeometry(boxGeo);
      const line = new THREE.LineSegments(
        wireGeo,
        new THREE.LineBasicMaterial({ color, linewidth: 2 })
      );
      line.position.set(inst.centroid[0], inst.centroid[1], inst.centroid[2]);
      group.add(line);

      // Semi-transparent center volume
      const meshMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.25,
      });
      const boxMesh = new THREE.Mesh(boxGeo, meshMat);
      boxMesh.position.set(inst.centroid[0], inst.centroid[1], inst.centroid[2]);
      group.add(boxMesh);

      // Top beacon sphere
      const sphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({ color });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.set(inst.centroid[0], inst.centroid[1] + 0.5, inst.centroid[2]);
      group.add(sphere);
    });
  }, [selectedCategory, categories]);

  // Focus Camera smoothly on instance when clicked
  useEffect(() => {
    if (!focusedInstance || !controlsRef.current || !cameraRef.current) return;

    const targetPos = new THREE.Vector3(...focusedInstance.centroid);
    controlsRef.current.target.copy(targetPos);
    cameraRef.current.position.set(
      targetPos.x + 1.2,
      targetPos.y + 1.0,
      targetPos.z + 1.5
    );
    controlsRef.current.update();
  }, [focusedInstance]);

  return (
    <div className="relative w-full h-full select-none overflow-hidden bg-slate-900">
      <div ref={containerRef} className={`w-full h-full ${mode === 'splat' ? 'hidden' : ''}`} />
      <div ref={splatContainerRef} className={`absolute inset-0 ${mode === 'mesh' ? 'hidden' : ''}`} />

      {/* Floating Control Toolbar */}
      <div className="absolute top-5 left-5 z-20 flex items-center gap-2 bg-slate-900/80 backdrop-blur-md border border-slate-700/60 p-1.5 rounded-xl shadow-xl">
        <button
          onClick={() => setMode('mesh')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            mode === 'mesh'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-300 hover:text-white hover:bg-slate-800'
          }`}
        >
          📐 Metric Mesh
        </button>
        <button
          onClick={() => setMode('splat')}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            mode === 'splat'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-300 hover:text-white hover:bg-slate-800'
          }`}
        >
          ✨ Gaussian Splat
        </button>
      </div>

      {/* Instructions / Controls Overlay */}
      <div className="absolute bottom-5 left-5 z-20 bg-slate-900/80 backdrop-blur-md border border-slate-700/60 px-3.5 py-2 rounded-xl text-xs text-slate-300 shadow-xl flex items-center gap-3">
        <span>🖱️ <b>Left click + drag</b>: Orbit</span>
        <span className="text-slate-600">•</span>
        <span><b>Right click + drag</b>: Pan</span>
        <span className="text-slate-600">•</span>
        <span><b>Scroll</b>: Zoom</span>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-950/70 backdrop-blur-sm text-white">
          <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mb-4" />
          <p className="text-sm font-semibold tracking-wide">
            Loading {mode === 'mesh' ? '3D Metric Mesh' : 'Gaussian Splat'}…
          </p>
          {loadingProgress > 0 && (
            <p className="text-xs text-slate-400 mt-1">{loadingProgress}% downloaded</p>
          )}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-slate-950/80 text-white p-6">
          <div className="text-4xl mb-3">⚠️</div>
          <p className="text-red-400 font-semibold mb-2">{error}</p>
          <button
            onClick={() => setMode((m) => (m === 'mesh' ? 'splat' : 'mesh'))}
            className="btn-secondary text-xs mt-3"
          >
            Try alternate representation
          </button>
        </div>
      )}
    </div>
  );
}
