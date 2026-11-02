const $ = (selector) => document.querySelector(selector);

const elements = {
  sidebar: $("#sidebar"),
  sidebarClose: $("#sidebarClose"),
  menuButton: $("#menuButton"),
  backdrop: $("#mobileBackdrop"),
  newChatButton: $("#newChatButton"),
  composerForm: $("#composerForm"),
  promptInput: $("#promptInput"),
  sendButton: $("#sendButton"),
  charCount: $("#charCount"),
  welcomeScreen: $("#welcomeScreen"),
  messageList: $("#messageList"),
  chatViewport: $("#chatViewport"),
  modelName: $("#modelName"),
  sidebarStatus: $("#sidebarStatus"),
  detailsButton: $("#detailsButton"),
  inspector: $("#inspector"),
  closeInspector: $("#closeInspector"),
  inspectorMeta: $("#inspectorMeta"),
  inspectorJson: $("#inspectorJson"),
  toast: $("#toast"),
};

const SESSION_KEY = "loop-studio.session";
const TRANSCRIPT_PREFIX = "loop-studio.transcript.";
const MAX_VISIBLE_MESSAGES = 80;
let sessionId = sessionStorage.getItem(SESSION_KEY);
let isSending = false;
let lastRun = null;
let toastTimer = null;

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2600);
}

function setApiState(online, label) {
  const footer = elements.sidebarStatus.parentElement;
  elements.sidebarStatus.textContent = label;
  footer.classList.toggle("offline", !online);
}

function transcriptKey(id = sessionId) {
  return `${TRANSCRIPT_PREFIX}${id}`;
}

function saveTranscript() {
  if (!sessionId) return;
  const messages = [...elements.messageList.querySelectorAll(".message-row[data-role]")]
    .filter((row) => row.dataset.role !== "pending")
    .slice(-MAX_VISIBLE_MESSAGES)
    .map((row) => ({
      role: row.dataset.role,
      text: row.dataset.text || "",
      status: row.dataset.status || "success",
      duration: row.dataset.duration || "",
      time: row.dataset.time || "",
    }));
  try {
    sessionStorage.setItem(transcriptKey(), JSON.stringify(messages));
  } catch (error) {
    console.warn("Could not save the in-tab transcript", error);
  }
}

function loadTranscript() {
  if (!sessionId) return;
  try {
    const raw = sessionStorage.getItem(transcriptKey());
    const messages = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(messages) || messages.length === 0) return;
    elements.welcomeScreen.classList.add("hidden");
    elements.messageList.classList.remove("hidden");
    for (const message of messages.slice(-MAX_VISIBLE_MESSAGES)) {
      appendMessage(message.role, message.text, {
        status: message.status,
        duration: message.duration,
        time: message.time,
        persist: false,
      });
    }
    scrollToBottom();
  } catch (error) {
    console.warn("Could not restore this tab's transcript", error);
  }
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.chatViewport.scrollTop = elements.chatViewport.scrollHeight;
  });
}

function formatTime(value = new Date()) {
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(value);
}

function appendInlineMarkdown(parent, text) {
  // Build DOM nodes instead of injecting HTML so model output stays inert/safe.
  const tokenPattern = /(\*\*|__)(.+?)\1|~~(.+?)~~|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<]+)|(?<!\*)\*([^*\n]+)\*|(?<!_)_([^_\n]+)_/g;
  let cursor = 0;
  let match;

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > cursor) {
      parent.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }

    let node;
    if (match[2] !== undefined) {
      node = document.createElement('strong');
      node.textContent = match[2];
    } else if (match[3] !== undefined) {
      node = document.createElement('del');
      node.textContent = match[3];
    } else if (match[4] !== undefined) {
      node = document.createElement('code');
      node.className = 'inline-code';
      node.textContent = match[4];
    } else if (match[5] !== undefined) {
      node = document.createElement('a');
      node.href = match[6];
      node.textContent = match[5];
      node.target = '_blank';
      node.rel = 'noopener noreferrer';
    } else if (match[7] !== undefined) {
      node = document.createElement('a');
      node.href = match[7];
      node.textContent = match[7];
      node.target = '_blank';
      node.rel = 'noopener noreferrer';
    } else {
      node = document.createElement('em');
      node.textContent = match[8] ?? match[9] ?? '';
    }
    parent.appendChild(node);
    cursor = tokenPattern.lastIndex;
  }

  if (cursor < text.length) {
    parent.appendChild(document.createTextNode(text.slice(cursor)));
  }
}

function appendCodeBlock(container, codeText, language = '') {
  const block = document.createElement('div');
  block.className = 'code-block';
  const head = document.createElement('div');
  head.className = 'code-block-head';
  const languageLabel = document.createElement('span');
  languageLabel.textContent = language || 'CODE';
  const copy = document.createElement('button');
  copy.className = 'code-copy';
  copy.type = 'button';
  copy.textContent = 'Copy';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
    } catch {
      showToast('Clipboard access is unavailable in this browser context.');
    }
  });
  head.append(languageLabel, copy);
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.textContent = codeText.replace(/\n$/, '');
  pre.appendChild(code);
  block.append(head, pre);
  container.appendChild(block);
}

function isMarkdownBlockStart(lines, index) {
  const line = lines[index] ?? '';
  return /^\s*```/.test(line)
    || /^ {0,3}#{1,6}\s+/.test(line)
    || /^ {0,3}(?:---+|___+|\*\s*\*\s*\*)\s*$/.test(line)
    || /^\s*>/.test(line)
    || /^\s*[-*+]\s+/.test(line)
    || /^\s*\d+[.)]\s+/.test(line)
    || (index + 1 < lines.length && line.includes('|') && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1]));
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function renderMarkdownBlocks(container, source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    const fence = line.match(/^ {0,3}```([\w.+-]*)\s*$/);
    if (fence) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !/^ {0,3}```\s*$/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      appendCodeBlock(container, codeLines.join('\n'), fence[1]);
      continue;
    }

    const heading = line.match(/^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      container.appendChild(element);
      index += 1;
      continue;
    }

    if (/^ {0,3}(?:---+|___+|\*\s*\*\s*\*)\s*$/.test(line)) {
      container.appendChild(document.createElement('hr'));
      index += 1;
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*> ?/, ''));
        index += 1;
      }
      const quote = document.createElement('blockquote');
      renderMarkdownBlocks(quote, quoteLines.join('\n'));
      container.appendChild(quote);
      continue;
    }

    // Markdown tables: header, separator, then body rows.
    if (index + 1 < lines.length && line.includes('|')
      && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1])) {
      const headers = splitTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        rows.push(splitTableRow(lines[index]));
        index += 1;
      }
      const wrap = document.createElement('div');
      wrap.className = 'markdown-table-wrap';
      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      headers.forEach((value) => {
        const th = document.createElement('th');
        appendInlineMarkdown(th, value);
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);
      const tbody = document.createElement('tbody');
      rows.forEach((values) => {
        const tr = document.createElement('tr');
        headers.forEach((_, column) => {
          const td = document.createElement('td');
          appendInlineMarkdown(td, values[column] ?? '');
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.append(thead, tbody);
      wrap.appendChild(table);
      container.appendChild(wrap);
      continue;
    }

    const listMatch = line.match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
    if (listMatch) {
      const ordered = /^\d/.test(listMatch[1]);
      const list = document.createElement(ordered ? 'ol' : 'ul');
      while (index < lines.length) {
        const item = lines[index].match(/^\s*([-*+]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        const li = document.createElement('li');
        appendInlineMarkdown(li, item[2]);
        list.appendChild(li);
        index += 1;
      }
      container.appendChild(list);
      continue;
    }

    // Ordinary paragraph. Soft-wrapped source lines become one readable line.
    const paragraphLines = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines, index)) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = document.createElement('p');
    appendInlineMarkdown(paragraph, paragraphLines.join(' '));
    container.appendChild(paragraph);
  }
}

function appendMarkdown(container, text) {
  const markdown = document.createElement('div');
  markdown.className = 'assistant-markdown';
  renderMarkdownBlocks(markdown, text);
  container.appendChild(markdown);
}

function appendUserText(container, text) {
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  container.appendChild(body);
}

function appendMessage(role, text, options = {}) {
  const { status = "success", duration = "", time = formatTime(), persist = true } = options;
  elements.welcomeScreen.classList.add("hidden");
  elements.messageList.classList.remove("hidden");

  const row = document.createElement("article");
  row.className = `message-row ${role}${role === "assistant" && status === "failed" ? " error" : ""}`;
  row.dataset.role = role;
  row.dataset.text = text;
  row.dataset.status = status;
  row.dataset.duration = String(duration || "");
  row.dataset.time = time;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "L";
    avatar.setAttribute("aria-hidden", "true");
    row.appendChild(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";

  if (role === "assistant") {
    const author = document.createElement("div");
    author.className = "message-author";
    const name = document.createElement("span");
    name.textContent = "Loop Agent";
    const timestamp = document.createElement("span");
    timestamp.className = "message-time";
    timestamp.textContent = time;
    author.append(name, timestamp);
    content.appendChild(author);
  }

  if (role === "assistant") appendMarkdown(content, text);
  else appendUserText(content, text);

  if (role === "assistant" && duration !== "") {
    const footer = document.createElement("div");
    footer.className = "message-footer";
    const badge = document.createElement("span");
    badge.className = `run-badge${status === "failed" ? " failed" : ""}`;
    badge.textContent = status === "success" ? "COMPLETED" : "FAILED";
    const elapsed = document.createElement("span");
    elapsed.textContent = `${duration} ms`;
    footer.append(badge, elapsed);
    content.appendChild(footer);
  }

  row.appendChild(content);
  elements.messageList.appendChild(row);
  if (persist) saveTranscript();
  scrollToBottom();
  return row;
}

function addPendingMessage() {
  elements.welcomeScreen.classList.add("hidden");
  elements.messageList.classList.remove("hidden");
  const row = document.createElement("article");
  row.className = "message-row assistant pending-row";
  row.dataset.role = "pending";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "L";
  const content = document.createElement("div");
  content.className = "message-content";
  const author = document.createElement("div");
  author.className = "message-author";
  author.textContent = "Loop Agent";
  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  const label = document.createElement("span");
  label.textContent = "";
  indicator.append(label, document.createElement("span"), document.createElement("span"));
  const caption = document.createElement("span");
  caption.textContent = "Thinking";
  caption.style.marginLeft = "5px";
  indicator.appendChild(caption);
  content.append(author, indicator);
  row.append(avatar, content);
  elements.messageList.appendChild(row);
  scrollToBottom();
  return row;
}

function updateInspector(data) {
  lastRun = data;
  const bits = [
    `Request ${data.request_id || "—"}`,
    `${data.duration_ms ?? "—"} ms`,
    `Status: ${(data.status || "unknown").toUpperCase()}`,
  ];
  elements.inspectorMeta.textContent = bits.join("   ·   ");
  elements.inspectorJson.textContent = JSON.stringify(data, null, 2);
}

async function createSession() {
  const response = await fetch("/api/sessions", { method: "POST" });
  if (!response.ok) throw new Error("The local API could not create a test session.");
  const data = await response.json();
  sessionId = data.session_id;
  sessionStorage.setItem(SESSION_KEY, sessionId);
  elements.messageList.replaceChildren();
  elements.messageList.classList.add("hidden");
  elements.welcomeScreen.classList.remove("hidden");
  elements.inspectorMeta.textContent = "No runs yet";
  elements.inspectorJson.textContent = "Send a message to inspect its structured result.";
  lastRun = null;
  setApiState(true, "Local API ready");
  return sessionId;
}

async function startNewChat() {
  if (isSending) {
    showToast("Let the current run finish before starting a new session.");
    return;
  }
  const previousSession = sessionId;
  elements.newChatButton.disabled = true;
  try {
    if (previousSession) {
      try {
        await fetch(`/api/sessions/${encodeURIComponent(previousSession)}`, { method: "DELETE" });
      } catch (error) {
        console.debug("Previous session cleanup skipped", error);
      }
      sessionStorage.removeItem(transcriptKey(previousSession));
    }
    await createSession();
    elements.promptInput.value = "";
    updateComposerState();
    closeSidebar();
    elements.promptInput.focus();
    showToast("New test session ready.");
  } catch (error) {
    sessionId = null;
    sessionStorage.removeItem(SESSION_KEY);
    showToast(error.message || "Could not start a new session.");
    setApiState(false, "API unavailable");
  } finally {
    elements.newChatButton.disabled = false;
  }
}

function updateComposerState() {
  const value = elements.promptInput.value;
  elements.charCount.textContent = `${value.length.toLocaleString()} / 12,000`;
  elements.sendButton.disabled = !value.trim() || isSending;
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
}

async function sendMessage(message) {
  const trimmed = message.trim();
  if (!trimmed || isSending) return;
  if (!sessionId) {
    try { await createSession(); }
    catch (error) { showToast(error.message); return; }
  }

  isSending = true;
  elements.promptInput.value = "";
  updateComposerState();
  appendMessage("user", trimmed);
  const pending = addPendingMessage();
  elements.sendButton.disabled = true;
  elements.promptInput.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: trimmed }),
    });

    if (response.status === 404) {
      pending.remove();
      appendMessage("assistant", "This test session has expired. Start a new chat to continue.", { status: "failed" });
      setApiState(false, "Session expired");
      return;
    }

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The local API returned an error.");
    pending.remove();
    updateInspector(data);
    const answer = data.status === "success"
      ? (data.response || "The agent completed without a text response.")
      : (data.error || "The agent run failed without an error message.");
    appendMessage("assistant", answer, {
      status: data.status,
      duration: data.duration_ms,
    });
    setApiState(true, "Local API ready");
  } catch (error) {
    pending.remove();
    appendMessage("assistant", `Could not reach the agent runtime. ${error.message || "Check the server terminal and try again."}`, { status: "failed" });
    setApiState(false, "Runtime unavailable");
    showToast("Request failed. Check the FastAPI terminal for details.");
  } finally {
    isSending = false;
    elements.promptInput.disabled = false;
    updateComposerState();
    elements.promptInput.focus();
    scrollToBottom();
  }
}

function openSidebar() {
  elements.sidebar.classList.add("open");
  elements.backdrop.classList.add("visible");
}
function closeSidebar() {
  elements.sidebar.classList.remove("open");
  elements.backdrop.classList.remove("visible");
}
function toggleInspector(force) {
  const shouldOpen = typeof force === "boolean" ? force : elements.inspector.classList.contains("hidden");
  elements.inspector.classList.toggle("hidden", !shouldOpen);
  elements.detailsButton.setAttribute("aria-expanded", String(shouldOpen));
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("API health check failed");
    const data = await response.json();
    elements.modelName.textContent = data.model || "Configured model";
    elements.modelName.title = data.model || "";
    setApiState(true, "Local API ready");
  } catch {
    elements.modelName.textContent = "API unavailable";
    setApiState(false, "API unavailable");
  }
}

// Submit and composer controls.
elements.composerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.promptInput.value);
});
elements.promptInput.addEventListener("input", updateComposerState);
elements.promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage(elements.promptInput.value);
  }
});

// Prompt cards and sidebar shortcuts.
document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.promptInput.value = button.dataset.prompt || "";
    updateComposerState();
    elements.promptInput.focus();
    closeSidebar();
  });
});
elements.newChatButton.addEventListener("click", startNewChat);
elements.menuButton.addEventListener("click", openSidebar);
elements.sidebarClose.addEventListener("click", closeSidebar);
elements.backdrop.addEventListener("click", closeSidebar);
elements.detailsButton.addEventListener("click", () => toggleInspector());
elements.closeInspector.addEventListener("click", () => toggleInspector(false));

// Keyboard shortcut for a fresh session.
document.addEventListener("keydown", (event) => {
  const isMac = navigator.platform.toUpperCase().includes("MAC");
  if ((isMac ? event.metaKey : event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    startNewChat();
  }
  if (event.key === "Escape") {
    closeSidebar();
    toggleInspector(false);
  }
});

// Boot: restore this tab's in-memory conversation when possible.
(async function boot() {
  updateComposerState();
  await loadHealth();
  if (sessionId) {
    try {
      const transcript = sessionStorage.getItem(transcriptKey());
      // Check that the backend still owns this session before restoring it.
      if (transcript !== null) {
        const messages = JSON.parse(transcript);
        if (Array.isArray(messages) && messages.length) loadTranscript();
      }
      const check = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { cache: "no-store" });
      if (!check.ok) throw new Error("The previous session is no longer active");
    } catch {
      sessionId = null;
      sessionStorage.removeItem(SESSION_KEY);
      await createSession();
    }
  } else {
    try { await createSession(); }
    catch { setApiState(false, "API unavailable"); }
  }
})();
