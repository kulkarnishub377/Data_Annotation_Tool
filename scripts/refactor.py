import os

# Paths
base_dir = r"d:\model_train\Data_annotrator_tool"
index_path = os.path.join(base_dir, "index.html")
js_path = os.path.join(base_dir, "app.js")
css_path = os.path.join(base_dir, "style.css")

if os.path.exists(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "<style>" in content and "app.js" not in content:
        print("[UPGRADE] Splitting index.html into app.js and style.css...")
        
        # 1. Extract CSS
        style_s = content.find("<style>") + 7
        style_e = content.find("</style>")
        css = content[style_s:style_e].strip()
        
        # 2. Extract JS
        script_s = content.find("<script>") + 8
        script_e = content.find("</script>", script_s)
        js = content[script_s:script_e].strip()
        
        # 3. Create new HTML
        html_head = content[:content.find("<style>")].strip()
        
        # NOTE: Using Flask standard routing for static files is safer, but if you want them in the root:
        # We will link them directly.
        html_head += '\n  <link rel="stylesheet" href="/style.css">\n'
        
        html_body = content[style_e+8:content.find("<script>")].strip()
        
        # Inject Export Button
        html_body = html_body.replace('<div class="grid-controls">', 
            '<div class="grid-controls">\n          <button class="btn btn-primary" id="btn-export-dataset" onclick="window.location.href=\'/api/export\'"><i class="fa-solid fa-file-export"></i> Export Dataset</button>')
        
        # Inject Minimap Canvas
        html_body = html_body.replace('<canvas id="annot-canvas"></canvas>',
            '<canvas id="annot-canvas"></canvas>\n        <canvas id="minimap" style="position:absolute; bottom:10px; right:40px; width:120px; border:1px solid #3b82f6; border-radius:4px; display:none; background:#000; pointer-events:none; z-index:20; box-shadow: 0 4px 12px rgba(0,0,0,0.5);"></canvas>')
        
        # Inject Dashboard Health Warnings
        html_body = html_body.replace('<div class="cls-chart">',
            '<div class="cls-chart">\n          <div id="health-warnings" style="margin-bottom:8px;font-size:11px;color:#f87171;font-weight:600;display:none;"></div>')
        
        html_tail = content[script_e+9:].strip()
        html_tail = f'<script src="/app.js"></script>\n{html_tail}'
        
        new_html = f'{html_head}\n{html_body}\n{html_tail}'
        
        # 4. Inject new JS features
        new_js = js + """
\n// ─── V3.0 NEW FEATURES ───
function forceDrawMode(state) {
  if (drawMode !== state) document.getElementById("btn-draw").click();
}

// Right Click Hold to Draw
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {
  if (e.button === 2 && currentIndex >= 0 && !isPanning) {
    e.preventDefault();
    forceDrawMode(true);
    const { x, y } = getPos(e);
    drag = { on: true, type: "draw", bIdx: -1, sx: x, sy: y, ds: { x, y } };
  }
});
canvas.addEventListener("mouseup", e => {
  if (e.button === 2) {
    e.preventDefault();
    setTimeout(() => forceDrawMode(false), 50);
  }
});

// Minimap
const minimap = document.getElementById("minimap");
const mctx = minimap ? minimap.getContext("2d") : null;
function updateMinimap() {
  if (!minimap || !imgEl.src || !imgEl.complete) return;
  if (zoomScale <= 1.0) { minimap.style.display = "none"; return; }
  minimap.style.display = "block";
  const mw = minimap.width = 120;
  const mh = minimap.height = 120 * (imgNatH / imgNatW);
  mctx.clearRect(0,0,mw,mh);
  mctx.drawImage(imgEl, 0, 0, mw, mh);
  
  const { ox, oy, scale } = getTransform();
  const dW = imgNatW * scale, dH = imgNatH * scale;
  
  const vX = Math.max(0, -ox / dW);
  const vY = Math.max(0, -oy / dH);
  const vW = Math.min(1, canvas.width / dW);
  const vH = Math.min(1, canvas.height / dH);
  
  mctx.strokeStyle = "#3b82f6";
  mctx.lineWidth = 2;
  mctx.fillStyle = "rgba(59,130,246,0.2)";
  mctx.fillRect(vX * mw, vY * mh, vW * mw, vH * mh);
  mctx.strokeRect(vX * mw, vY * mh, vW * mw, vH * mh);
}

const oldRenderCanvas = renderCanvas;
renderCanvas = function() {
  oldRenderCanvas();
  updateMinimap();
};

// Health Dashboard Checks
setInterval(() => {
  const hw = document.getElementById('health-warnings');
  if(hw) {
    let warns = [];
    let nullCount = 0;
    if (typeof allImages !== 'undefined') {
       nullCount = allImages.filter(i => !i.annotated).length;
       if(nullCount > 0) warns.push('⚠️ ' + nullCount + ' pending/unannotated images.');
    }
    if(warns.length > 0) {
      hw.innerHTML = warns.join('<br>');
      hw.style.display = 'block';
    } else {
      hw.style.display = 'none';
    }
  }
}, 3000);
"""
        
        # 5. Write everything to files
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_html)
            
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(css)
            
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(new_js)
            
        print("[SUCCESS] Files successfully split and V3.0 features added!")
    else:
        print("[INFO] index.html seems to be already split.")
else:
    print("[ERROR] index.html not found.")
