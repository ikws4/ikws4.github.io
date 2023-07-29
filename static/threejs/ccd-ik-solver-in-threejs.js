import * as THREE from "three";
import Sketch from "./sketch.js";
import colors from "./colors.js";
import { DragControls } from "three/addons/controls/DragControls";

function buildLight(scene) {
  let light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(0, 25, 100);

  //const helper = new THREE.DirectionalLightHelper(light, 5);
  //scene.add(helper);

  let lightAngle = 0;
  // updates.push(() => {
  //   let r = 50;
  //   light.position.x = Math.cos(lightAngle) * r;
  //   light.position.z = Math.sin(lightAngle) * r;
  //   lightAngle += Math.PI / 760;
  //   light.lookAt(0, 0, 0);
  // });

  let ambientLight = new THREE.AmbientLight("#a7ceeb", 0.5);
  scene.add(ambientLight);

  scene.add(light);
}

function buildCheckBoard(scene) {
  let geometry = new THREE.PlaneGeometry(100, 100, 10, 10);
  let texture = new THREE.TextureLoader().load(
    "/threejs/textures/checkerboard.png"
  );
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(25, 25);
  texture.minFilter = THREE.LinearMipmapNearestFilter;
  texture.magFilter = THREE.NearestFilter;
  let material = new THREE.MeshStandardMaterial({ map: texture });
  let plane = new THREE.Mesh(geometry, material);
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = -0.5;
  scene.add(plane);
}

function buildCube(scene) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
  const cube = new THREE.Mesh(geometry, material);
  scene.add(cube);
}

function buildCCDIK(t) {
  let scene = t.scene;
  let camera = t.camera;
  let renderer = t.renderer;
  let orbitControls = t.orbitControls;

  const numBones = 10;

  // build bones
  function createBone(pos, col) {
    const geometry = new THREE.BoxGeometry(1, 1.5, 1);
    geometry.translate(0, 1.5 / 2, 0);
    const material = new THREE.MeshStandardMaterial({ color: col });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(pos[0], pos[1], pos[2]);
    cube.name = "bone";
    return cube;
  }

  let bones = [createBone([0, 0, 0], colors.all[0])];
  for (let i = 1; i < numBones; i++) {
    let bone = createBone([0, 1.5, 0], colors.all[i % colors.all.length]);
    bones.push(bone);
    bones[i - 1].add(bone);
  }
  scene.add(bones[0]);

  function worldPos(bone) {
    let pos = new THREE.Vector3();
    bone.getWorldPosition(pos);
    return pos;
  }

  function solve(bones, target) {
    let effector = bones[bones.length - 1];

    for (let j = 2; j < bones.length; j++) {
      for (let i = bones.length - 2; i >= bones.length - j; i--) {
        let bone = bones[i];
        let bonePos = worldPos(bone);

        let v1 = worldPos(effector).sub(bonePos);
        let v2 = worldPos(target).sub(bonePos);
        let angle = v1.angleTo(v2);

        let axis = new THREE.Vector3();
        axis.crossVectors(v1, v2).normalize();
        bone.rotateOnAxis(axis, angle);
      }
    }
  }

  let target = new THREE.Mesh(
    new THREE.SphereGeometry(0.2),
    new THREE.MeshStandardMaterial({ color: colors.foam })
  );
  target.position.set(10, 10, 0);
  scene.add(target);

  const dragControls = new DragControls([target], camera, renderer.domElement);
  dragControls.addEventListener("dragstart", (event) => {
    orbitControls.enabled = false;
  });

  dragControls.addEventListener("dragend", (event) => {
    orbitControls.enabled = true;
  });

  t.updatable.push(() => solve(bones, target));

  let angle = 0;
  t.updatable.push(() => {
    target.position.x = Math.cos(angle) * 10;
    angle += 0.025;
  });
}

new Sketch("ccd-ik-solver-in-threejs", (t) => {
  t.camera.position.set(0, 12, -20);
  buildLight(t.scene);
  buildCheckBoard(t.scene);
  // buildCube(t.scene);
  buildCCDIK(t);
});
