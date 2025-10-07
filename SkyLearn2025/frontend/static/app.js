// app.js
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const topkEl = document.getElementById("topk");

function appendMsg(text, cls="bot") {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerHTML = text;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}

sendBtn.onclick = async () => {
  const q = inputEl.value.trim();
  if (!q) return;
  appendMsg(`<strong>Vous:</strong> ${q}`, "user");
  inputEl.value = "";
  appendMsg("… recherche et génération en cours …", "bot");
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question:q, top_k: parseInt(topkEl.value)})
    });
    const data = await resp.json();
    // remove the loading message (last bot)
    const nodes = chatEl.querySelectorAll(".msg.bot");
    if (nodes.length) nodes[nodes.length-1].remove();
    appendMsg(`<strong>SkyLearn:</strong> <div>${data.answer.replace(/\n/g,"<br>")}</div>`, "bot");
    // sources
    let srcHtml = "<div class='source'><strong>Sources :</strong><ul>";
    for (const s of data.sources) {
      srcHtml += `<li><em>${s.source}</em> (score: ${s.score.toFixed(3)})<div>${s.text.substring(0,400)}...</div></li>`;
    }
    srcHtml += "</ul></div>";
    appendMsg(srcHtml, "bot");
  } catch (e) {
    appendMsg("Erreur : " + e.toString(), "bot");
  }
};
