import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls";
import Inspector from "three-inspect";
import "./colors.js";

export default class Sketch {
  constructor(name, buildFn) {
    this._init(name);
    //this._initInspector();
    this._initOrbitControls();
    buildFn(this);
  }

  _init(name) {
    this.container = document.getElementById(name);

    this.updatable = [];
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.camera = new THREE.PerspectiveCamera(
      75,
      this.container.clientWidth / (window.innerHeight * 0.9),
      0.1,
      1000
    );

    this.scene.add(this.camera);
    this.renderer.setClearColor("#87ceeb");

    this.renderer.setSize(this.container.clientWidth, window.innerHeight * 0.9);
    this.container.appendChild(this.renderer.domElement);

    window.addEventListener("resize", (ev) => {
      this.renderer.setSize(
        this.container.clientWidth,
        window.innerHeight * 0.9
      );
      this.camera.aspect =
        this.container.clientWidth / (window.innerHeight * 0.9);
      this.camera.updateProjectionMatrix();
    });

    const tick = () => {
      requestAnimationFrame(tick);
      for (let update of this.updatable) update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  _initInspector() {
    new Inspector({
      scene: this.scene,
      camera: this.camera,
      renderer: this.renderer,
      options: {
        location: "overlay",
      },
    });

    let inspectorRoot = document.getElementsByClassName(
      "z-[100] absolute top-0 right-0 w-[350px] h-screen flex"
    )[0];
    inspectorRoot.parentNode.removeChild(inspectorRoot);
    this.container.appendChild(inspectorRoot);
    inspectorRoot.classList = "z-[100] absolute top-0 right-0 flex";
    inspectorRoot.style.width = "280px";
    inspectorRoot.style.height = "100%";
    inspectorRoot.style.left = "0";
    inspectorRoot.style.right = "unset";
    inspectorRoot.style.display = "none";

    window.addEventListener("keypress", (ev) => {
      if (ev.key === "i") {
        if (inspectorRoot.style.display === "none") {
          inspectorRoot.style.display = "flex";
        } else {
          inspectorRoot.style.display = "none";
        }
      }
    });
  }

  _initOrbitControls() {
    this.orbitControls = new OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    this.orbitControls.enableDamping = true;
    this.orbitControls.minPolarAngle = Math.PI / 10;
    this.orbitControls.maxPolarAngle = Math.PI / 2.25;
    this.updatable.push(() => {
      if (this.orbitControls.enabled) {
        this.orbitControls.update();
      }
    });
  }
}
