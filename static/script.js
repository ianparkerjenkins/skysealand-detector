const imageInput = document.getElementById("imageInput");
const dropArea = document.getElementById("dropArea");
const uploadBtn = document.getElementById("uploadBtn");
const canvasContainer = document.getElementById("canvasContainer");

let selectedFiles = [];

// Drag & drop handlers
dropArea.addEventListener("click", () => imageInput.click());

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("hover");
});

dropArea.addEventListener("dragleave", () => dropArea.classList.remove("hover"));

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("hover");
  const files = Array.from(e.dataTransfer.files);
  handleFiles(files);
});

// File input handler
imageInput.addEventListener("change", (event) => handleFiles(Array.from(event.target.files)));

function handleFiles(files) {
  selectedFiles = files;
  canvasContainer.innerHTML = ""; // clear previous canvases

  selectedFiles.forEach(file => {
    const canvas = document.createElement("canvas");
    canvasContainer.appendChild(canvas);
    const ctx = canvas.getContext("2d");

    const reader = new FileReader();
    reader.onload = function (e) {
      const img = new Image();
      img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// Batch upload
uploadBtn.addEventListener("click", async () => {
  if (selectedFiles.length === 0) {
    alert("Please select one or more images first.");
    return;
  }

  const formData = new FormData();
  selectedFiles.forEach(file => formData.append("files", file));

  const response = await fetch("/infer", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const err = await response.json();
    alert("Error: " + err.detail);
    return;
  }

  const data = await response.json();
  drawBatchDetections(data.results, selectedFiles);
});


const CLASS_NAMES = {
  0: "airplane",
  1: "boat",
  2: "car",
  3: "ship"
};


function colorForClass(classId) {
  const hue = (classId * 137.508) % 360; // golden angle
  return `hsl(${hue}, 85%, 50%)`;
}


function drawBatchDetections(results, files) {
  canvasContainer.innerHTML = "";
  results.forEach(item => {
    const file = files.find(f => f.name === item.filename);
    if (!file) return;

    const canvas = document.createElement("canvas");
    canvasContainer.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = new Image();
      img.onload = function () {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        ctx.lineWidth = 2;
        ctx.font = "16px Arial";

        item.inference.forEach(det => {
          const [x1, y1, x2, y2] = det.bbox;

          const classId = det.class_id;
          const className = CLASS_NAMES[classId] ?? `class_${classId}`;
          const color = colorForClass(classId);
          console.log(className, color)

          // box
          ctx.strokeStyle = color;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          // label background
          const label = `${className} ${det.confidence.toFixed(2)}`;
          const textWidth = ctx.measureText(label).width;
          const textHeight = 16;

          ctx.fillStyle = color;
          ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 6, textHeight + 4);

          // label text
          ctx.fillStyle = "white";
          ctx.fillText(label, x1 + 3, y1 - 6);
        });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}
