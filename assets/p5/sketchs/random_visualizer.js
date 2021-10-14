// Gui setup {{{
let gui;
let sidebarWidth;
let sidebarHeight;
let sketchWidth;
let sketchHeight;
let isStart = false;

function createGUI() {
  gui = createGui();
  gui.loadStyle("Seafoam");

  sidebarWidth = width * 0.4;
  sidebarHeight = height;
  sketchWidth = width - sidebarWidth;
  sketchHeight = height;

  // define control button
  const buttonStyle = {
    textSize: 12,
    rounding: 6,
  };
  const buttons = [
    {
      lable: "PLAY",
      onPressed: onPlayButtonPressed,
    },
    {
      lable: "STOP",
      onPressed: onStopButtonPressed,
    },
    {
      lable: "RESET",
      onPressed: onResetButtonPressed,
    },
  ];

  const buttonGap = 16;
  const buttonWidth =
    (sidebarWidth - buttonGap * (buttons.length + 1)) / buttons.length;

  buttons.forEach((item, index) => {
    let x = index * buttonWidth + buttonGap * (index + 1);
    let button = createButton(item.lable, x, 16, buttonWidth, 32);
    button.setStyle(buttonStyle);
    button.onPress = item.onPressed;
  });

  sidebarGui(0, 64);
}

function drawGUI() {
  drawGui();

  stroke(p_overlay);
  strokeWeight(2);
  line(sidebarWidth, 0, sidebarWidth, height);
  line(0, 64, sidebarWidth, 64);
}
// }}}

// Callback functions {{{
function onPlayButtonPressed() {
  isStart = true;
}

function onStopButtonPressed() {
  isStart = false;
}

function onResetButtonPressed() {
  isStart = false;
  background(p_base);
  resetSketch();
}
// }}}

// Internal {{{
function setup() {
  const id = "p5-sketch-random-visualizer";
  const container = document.getElementById(id);
  const canvas = createCanvas(container.offsetWidth, 400);
  canvas.parent(id);

  createGUI();
  background(p_base);
}

function draw() {
  if (isStart) {
    background(p_base);
  }

  push();
  {
    translate(sidebarWidth, 0);
    if (isStart) {
      drawSketch();
    } else {
      drawPausedOverlay();
    }
  }
  pop();

  drawGUI();
}

function drawPausedOverlay() {
  push();
  {
    translate(sketchWidth / 2, sketchHeight / 2);
    rotate(PI / 2);
    noStroke();
    fill(p_overlay);
    ellipse(0, 0, 64, 64);
    fill(p_subtle);
    triangle(-15, 10, 0, -20, 15, 10);
  }
  pop();
}
// }}}

let data = [
  {
    name: "gragon",
    probability: 1,
    count: 0,
  },
  {
    name: "fish",
    probability: 10,
    count: 0,
  },
  {
    name: "tiger",
    probability: 5,
    count: 0,
  },
  {
    name: "cat",
    probability: 10,
    count: 0,
  },
  {
    name: "elephant",
    probability: 14,
    count: 0,
  },
  {
    name: "dog",
    probability: 20,
    count: 0,
  },
];

function resetSketch() {
  data.forEach((item) => {
    item.count = 0;
  });
}

function sidebarGui(originX, originY) {
  let sliderStyle = {
    rounding: 6,
  };
  data.forEach((item, index) => {
    let slider = createSlider(
      item.name,
      originX + 16,
      originY + 16 + index * 48,
      sidebarWidth - 32,
      32,
      0,
      20
    );
    slider.setStyle(sliderStyle);
    slider.val = item.probability;
    slider.onChange = function () {
      item.probability = int(slider.val);
    };
  });
}

function drawSketch() {
  noStroke();
  fill(p_foam);
  pick();
  drawHistogram();
}

function drawHistogram() {
  let gap = 4;
  let w = (sketchWidth - gap) / data.length - gap;

  data.forEach((item, index) => {
    let x = index * (w + gap) + gap;
    let y = height - item.count;

    rect(x, y, w, item.count);

    textSize(12);
    textAlign(CENTER);
    text(`${item.name}(${item.probability})`, x + w / 2, y - 4);
  });
}

function pick() {
  let total = data.reduce((sum, e) => sum + e.probability, 0);
  let i = 0;
  let r = int(random(total + 1));
  let acc = data[i].probability;

  while (acc < r) {
    acc += data[++i].probability;
  }

  data[i].count++;
}
