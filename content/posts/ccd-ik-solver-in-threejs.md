+++
title = "CCD Ik Solver in Three.js (Not Complete)"
date = "2023-07-28T00:56:16+08:00"
author = "ikws4"
cover = ""
tags = []
keywords = []
readingTime = false
threejs = true
Toc = false
+++

<!--more-->

{{< threejs name="ccd-ik-solver-in-threejs" >}}

```js
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
```

# Resources

1. ik rig: https://youtu.be/KLjTU0yKS00
1. ccd ik: https://youtu.be/MA1nT9RAF3k
1. Skeletal Animation: https://youtu.be/ZzMnu3v_MOw
1. Cyclic Coordonate Descent Inverse Kynematic (CCD IK): http://rodolphe-vaillant.fr/entry/114/cyclic-coordonate-descent-inverse-kynematic-ccd-ik