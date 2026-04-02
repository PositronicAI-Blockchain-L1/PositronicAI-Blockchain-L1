/**
 * Positronic AI Assistant — Widget Logic
 *
 * Self-initializing: injects DOM, handles open/close, focus trap,
 * keyword matching, typing indicator, session history, and ARIA.
 *
 * Note on innerHTML usage: All HTML content rendered by this widget comes
 * exclusively from the local assistant-data.js knowledge base (trusted,
 * developer-authored content shipped with the site). No user input is ever
 * rendered as HTML — user messages are always escaped via textContent.
 * This is a static site with no backend; the knowledge base is a JS file
 * served from the same origin.
 *
 * Focus order (when panel is open):
 *   Close button -> Input -> Send -> Suggestion chips -> Answer links -> (cycle)
 */
(function () {
  'use strict';

  /* ── State ─────────────────────────────────────── */
  var isOpen = false;
  var history = [];
  var HISTORY_KEY = 'posi-assistant-history';
  var SHOWN_KEY = 'posi-assistant-shown';
  var TYPING_DELAY = 600; // ms — snappy response feel

  /* ── Data ──────────────────────────────────────── */
  function getData() {
    return window.PositronicAssistantData || { categories: [], entries: [] };
  }

  /* ── Safe DOM helpers ──────────────────────────── */
  // Create a text node (safe — no HTML interpretation)
  function textEl(tag, text, className) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = text;
    return el;
  }

  // Create element with trusted HTML from knowledge base
  // ONLY used for developer-authored KB content, never for user input
  function trustedHtmlEl(tag, trustedHtml, className) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    el.innerHTML = trustedHtml; // eslint-disable-line -- trusted KB content only
    return el;
  }

  function escapeHtml(str) {
    var d = document.createElement('div');
    d.appendChild(document.createTextNode(str));
    return d.innerHTML;
  }

  /* ── DOM injection ─────────────────────────────── */
  function init() {
    if (document.getElementById('posi-assistant-trigger')) return;

    var logoSrc = '/logo.png';

    // Build trigger button
    var trigger = document.createElement('button');
    trigger.id = 'posi-assistant-trigger';
    trigger.className = 'posi-assistant-widget posi-assistant-trigger';
    trigger.setAttribute('aria-label', 'Open Positronic assistant');
    trigger.setAttribute('aria-expanded', 'false');
    var triggerImg = document.createElement('img');
    triggerImg.src = logoSrc;
    triggerImg.alt = '';
    triggerImg.draggable = false;
    trigger.appendChild(triggerImg);

    // Build panel
    var panel = document.createElement('div');
    panel.id = 'posi-assistant-panel';
    panel.className = 'posi-assistant-widget posi-assistant-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'posi-assistant-title');
    panel.setAttribute('data-open', 'false');

    // Header
    var header = document.createElement('div');
    header.className = 'posi-assistant-header';

    var headerLeft = document.createElement('div');
    headerLeft.className = 'posi-assistant-header-left';

    var headerAvatarWrap = document.createElement('div');
    headerAvatarWrap.className = 'posi-assistant-header-avatar';
    var headerAvatar = document.createElement('img');
    headerAvatar.src = logoSrc;
    headerAvatar.alt = '';
    headerAvatar.width = 36;
    headerAvatar.height = 36;
    headerAvatarWrap.appendChild(headerAvatar);
    headerLeft.appendChild(headerAvatarWrap);

    var headerInfo = document.createElement('div');
    headerInfo.className = 'posi-assistant-header-info';
    var title = document.createElement('h3');
    title.id = 'posi-assistant-title';
    title.className = 'posi-assistant-title';
    title.textContent = 'Positronic AI';
    headerInfo.appendChild(title);
    var subtitle = document.createElement('span');
    subtitle.className = 'posi-assistant-subtitle';
    subtitle.textContent = 'Your blockchain knowledge base';
    headerInfo.appendChild(subtitle);
    headerLeft.appendChild(headerInfo);

    var closeBtn = document.createElement('button');
    closeBtn.id = 'posi-assistant-close';
    closeBtn.className = 'posi-assistant-widget posi-assistant-close';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.textContent = '\u00D7';
    header.appendChild(headerLeft);
    header.appendChild(closeBtn);

    // Messages area
    var messages = document.createElement('div');
    messages.id = 'posi-assistant-messages';
    messages.className = 'posi-assistant-messages';
    messages.setAttribute('aria-live', 'polite');
    messages.setAttribute('aria-atomic', 'false');

    // Suggestions area
    var suggestions = document.createElement('div');
    suggestions.id = 'posi-assistant-suggestions';
    suggestions.className = 'posi-assistant-suggestions';
    suggestions.setAttribute('role', 'group');
    suggestions.setAttribute('aria-label', 'Suggested questions');

    // Input area
    var inputArea = document.createElement('div');
    inputArea.className = 'posi-assistant-input-area';
    var input = document.createElement('input');
    input.id = 'posi-assistant-input';
    input.className = 'posi-assistant-widget posi-assistant-input';
    input.type = 'text';
    input.placeholder = 'Ask anything about Positronic\u2026';
    input.setAttribute('aria-label', 'Ask anything about Positronic');
    var sendBtn = document.createElement('button');
    sendBtn.id = 'posi-assistant-send';
    sendBtn.className = 'posi-assistant-widget posi-assistant-send';
    sendBtn.setAttribute('aria-label', 'Send');
    sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    inputArea.appendChild(input);
    inputArea.appendChild(sendBtn);

    // Assemble panel with inner wrapper (for animated gradient border)
    var panelInner = document.createElement('div');
    panelInner.className = 'posi-assistant-panel-inner';
    panelInner.appendChild(header);
    panelInner.appendChild(messages);
    panelInner.appendChild(suggestions);
    panelInner.appendChild(inputArea);
    panel.appendChild(panelInner);

    document.body.appendChild(trigger);
    document.body.appendChild(panel);

    // Build tooltip-style label above trigger (hidden after first click)
    var ASK_LABEL_KEY = 'posi-assistant-ask-dismissed';
    if (!sessionStorage.getItem(ASK_LABEL_KEY)) {
      var askLabel = document.createElement('div');
      askLabel.className = 'posi-assistant-widget posi-assistant-ask-label';
      askLabel.id = 'posi-assistant-ask-label';
      var askLabelText = document.createElement('span');
      askLabelText.textContent = 'AI Assistant';
      askLabel.appendChild(askLabelText);
      var askArrow = document.createElement('div');
      askArrow.className = 'posi-assistant-ask-arrow';
      askLabel.appendChild(askArrow);
      document.body.appendChild(askLabel);
    }

    // Render suggestion chips
    renderSuggestions();

    // If there is stored history, show welcome message first, then restore history
    var hasStoredHistory = false;
    try {
      var storedCheck = sessionStorage.getItem(HISTORY_KEY);
      if (storedCheck) {
        var parsed = JSON.parse(storedCheck);
        if (parsed && parsed.length > 0) hasStoredHistory = true;
      }
    } catch (e) { /* ignore */ }

    if (hasStoredHistory && !sessionStorage.getItem(SHOWN_KEY)) {
      addBotMessage(
        '<p class="posi-assistant-welcome">'
        + '<strong style="color:#00E5FF;font-size:16px">Positronic Neural Assistant</strong></p>'
        + '<p style="margin-top:8px">I\'m your AI guide to the Positronic ecosystem. Ask me about:</p>'
        + '<ul style="margin:6px 0 8px 16px;line-height:1.7">'
        + '<li><strong>Wallet &amp; Explorer</strong> — send ASF, browse chain</li>'
        + '<li><strong>Tokenomics &amp; Staking</strong> — supply, halving, rewards</li>'
        + '<li><strong>AI Consensus</strong> — PoNC, 4 neural models, ZKML</li>'
        + '<li><strong>Features</strong> — PRC-20/721, bridge, DePIN, P2E</li>'
        + '<li><strong>Developers</strong> — RPC, SDKs, smart contracts</li>'
        + '<li><strong>Security</strong> — post-quantum, gasless txs</li>'
        + '</ul>'
        + '<p style="opacity:0.5;font-size:12px">Ctrl+K to toggle &bull; 70+ topics available</p>'
      );
      sessionStorage.setItem(SHOWN_KEY, '1');
    }

    // Restore session history
    restoreHistory();

    // Single scroll after all restored messages are added
    var msgContainer = document.getElementById('posi-assistant-messages');
    if (msgContainer) scrollToBottom(msgContainer);

    // Event listeners
    trigger.addEventListener('click', toggle);
    closeBtn.addEventListener('click', close);
    sendBtn.addEventListener('click', submitInput);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); submitInput(); }
    });

    // Escape + focus trap
    panel.addEventListener('keydown', handlePanelKeydown);

    // Keyboard shortcut: Ctrl+K / Cmd+K
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        toggle();
      }
    });
  }

  /* ── Suggestions renderer (accordion) ─────────── */
  var openCategoryId = null; // tracks which category is expanded

  function renderSuggestions() {
    var container = document.getElementById('posi-assistant-suggestions');
    if (!container) return;

    // Clear existing
    while (container.firstChild) container.removeChild(container.firstChild);

    var data = getData();

    data.categories.forEach(function (cat) {
      var entries = data.entries.filter(function (e) { return e.category === cat.id; });
      if (entries.length === 0) return;

      // Category header (clickable accordion toggle)
      var header = document.createElement('button');
      header.className = 'posi-assistant-widget posi-assistant-cat-header';
      header.setAttribute('aria-expanded', openCategoryId === cat.id ? 'true' : 'false');
      header.setAttribute('data-cat-id', cat.id);

      var headerLabel = document.createElement('span');
      headerLabel.className = 'posi-assistant-cat-header-label';
      headerLabel.textContent = cat.icon + ' ' + cat.label;

      var chevron = document.createElement('span');
      chevron.className = 'posi-assistant-cat-chevron';
      chevron.textContent = openCategoryId === cat.id ? '\u25BE' : '\u25B8'; // ▾ or ▸

      var countBadge = document.createElement('span');
      countBadge.className = 'posi-assistant-cat-count';
      countBadge.textContent = entries.length;

      header.appendChild(headerLabel);
      header.appendChild(countBadge);
      header.appendChild(chevron);
      container.appendChild(header);

      // Chips container (collapsed by default)
      var chips = document.createElement('div');
      chips.className = 'posi-assistant-chips';
      chips.setAttribute('data-cat-chips', cat.id);
      if (openCategoryId !== cat.id) {
        chips.style.display = 'none';
      }

      entries.forEach(function (entry) {
        var chip = document.createElement('button');
        chip.className = 'posi-assistant-widget posi-assistant-chip';
        chip.setAttribute('data-entry-id', entry.id);
        chip.textContent = entry.title;
        chip.addEventListener('click', function () {
          var match = getData().entries.find(function (e) { return e.id === entry.id; });
          if (match) showAnswer(match.title, match);
        });
        chips.appendChild(chip);
      });

      container.appendChild(chips);

      // Accordion toggle handler
      header.addEventListener('click', function () {
        toggleCategory(cat.id);
      });
    });
  }

  function toggleCategory(catId) {
    var container = document.getElementById('posi-assistant-suggestions');
    if (!container) return;

    // If already open, close it
    if (openCategoryId === catId) {
      openCategoryId = null;
    } else {
      openCategoryId = catId;
    }

    // Update all headers and chips
    var headers = container.querySelectorAll('.posi-assistant-cat-header');
    for (var i = 0; i < headers.length; i++) {
      var h = headers[i];
      var id = h.getAttribute('data-cat-id');
      var isExpanded = id === openCategoryId;
      h.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
      // Update chevron
      var chev = h.querySelector('.posi-assistant-cat-chevron');
      if (chev) chev.textContent = isExpanded ? '\u25BE' : '\u25B8';
    }

    var allChips = container.querySelectorAll('[data-cat-chips]');
    for (var j = 0; j < allChips.length; j++) {
      var c = allChips[j];
      var cId = c.getAttribute('data-cat-chips');
      c.style.display = cId === openCategoryId ? 'flex' : 'none';
    }
  }

  /* ── Open / Close ──────────────────────────────── */
  function dismissAskLabel() {
    var label = document.getElementById('posi-assistant-ask-label');
    if (label) {
      label.classList.add('hidden');
      sessionStorage.setItem('posi-assistant-ask-dismissed', '1');
      setTimeout(function () { if (label.parentNode) label.parentNode.removeChild(label); }, 400);
    }
  }

  function toggle() {
    dismissAskLabel();
    if (isOpen) close(); else open();
  }

  function open() {
    var panel = document.getElementById('posi-assistant-panel');
    var trigger = document.getElementById('posi-assistant-trigger');
    if (!panel || !trigger) return;

    isOpen = true;
    panel.setAttribute('data-open', 'true');
    trigger.setAttribute('aria-expanded', 'true');
    trigger.setAttribute('aria-label', 'Close Positronic assistant');

    // Show welcome on first open
    if (!sessionStorage.getItem(SHOWN_KEY) && history.length === 0) {
      addBotMessage(
        '<p class="posi-assistant-welcome">'
        + '<strong style="color:#00E5FF;font-size:16px">Positronic Neural Assistant</strong></p>'
        + '<p style="margin-top:8px">I\'m your AI guide to the Positronic ecosystem. Ask me about:</p>'
        + '<ul style="margin:6px 0 8px 16px;line-height:1.7">'
        + '<li><strong>Wallet &amp; Explorer</strong> — send ASF, browse chain</li>'
        + '<li><strong>Tokenomics &amp; Staking</strong> — supply, halving, rewards</li>'
        + '<li><strong>AI Consensus</strong> — PoNC, 4 neural models, ZKML</li>'
        + '<li><strong>Features</strong> — PRC-20/721, bridge, DePIN, P2E</li>'
        + '<li><strong>Developers</strong> — RPC, SDKs, smart contracts</li>'
        + '<li><strong>Security</strong> — post-quantum, gasless txs</li>'
        + '</ul>'
        + '<p style="opacity:0.5;font-size:12px">Ctrl+K to toggle &bull; 70+ topics available</p>'
      );
      sessionStorage.setItem(SHOWN_KEY, '1');
    }

    // Focus close button (first focusable in tab order)
    var closeBtn = document.getElementById('posi-assistant-close');
    if (closeBtn) closeBtn.focus();
  }

  function close() {
    var panel = document.getElementById('posi-assistant-panel');
    var trigger = document.getElementById('posi-assistant-trigger');
    if (!panel || !trigger) return;

    isOpen = false;
    panel.setAttribute('data-open', 'false');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-label', 'Open Positronic assistant');

    // Return focus to trigger
    trigger.focus();
  }

  /* ── Focus trap ────────────────────────────────── */
  function handlePanelKeydown(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }

    if (e.key !== 'Tab') return;

    var panel = document.getElementById('posi-assistant-panel');
    var focusable = panel.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;

    var first = focusable[0];
    var last = focusable[focusable.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  /* ── Input handling ────────────────────────────── */
  function submitInput() {
    var input = document.getElementById('posi-assistant-input');
    if (!input) return;
    var query = input.value.trim();
    if (!query) return;
    input.value = '';

    addUserMessage(query);

    var match = findBestMatch(query);
    showTypingThenAnswer(query, match);
  }

  function showAnswer(questionText, entry) {
    addUserMessage(questionText);
    showTypingThenAnswer(questionText, entry);
  }

  /* ── Matching algorithm ────────────────────────── */
  // Common stopwords that should not contribute to matching scores
  var STOPWORDS = {
    'a':1,'an':1,'the':1,'is':1,'are':1,'was':1,'were':1,'be':1,'been':1,
    'am':1,'do':1,'does':1,'did':1,'have':1,'has':1,'had':1,'will':1,
    'would':1,'could':1,'should':1,'may':1,'might':1,'can':1,'shall':1,
    'to':1,'of':1,'in':1,'for':1,'on':1,'with':1,'at':1,'by':1,'from':1,
    'as':1,'into':1,'about':1,'it':1,'its':1,'this':1,'that':1,'these':1,
    'those':1,'i':1,'me':1,'my':1,'we':1,'our':1,'you':1,'your':1,
    'he':1,'she':1,'they':1,'them':1,'their':1,'not':1,'no':1,'or':1,
    'and':1,'but':1,'if':1,'so':1,'just':1,'very':1,'also':1,'too':1,
    'what':1,'which':1,'who':1,'whom':1,'where':1,'when':1,'why':1,'how':1,
    'there':1,'here':1,'all':1,'any':1,'each':1,'some':1,'more':1,'much':1,
    'many':1,'most':1,'other':1,'than':1,'then':1,'up':1,'out':1,'only':1
  };

  function isStopword(w) { return STOPWORDS.hasOwnProperty(w); }

  function findBestMatch(query) {
    var data = getData();
    var q = normalize(query);
    var tokens = tokenize(q).filter(function (t) { return !isStopword(t); });
    var best = null;
    var bestScore = 0;

    // If all tokens are stopwords, no meaningful match possible
    if (tokens.length === 0) return null;

    data.entries.forEach(function (entry) {
      var score = 0;

      // Keyword match (high weight) — keywords are domain-specific, skip stopword filter
      entry.keywords.forEach(function (kw) {
        if (q.indexOf(kw.toLowerCase()) !== -1) score += 3;
      });

      // Title word match (medium-high weight) — only non-stopword tokens
      var titleLower = entry.title.toLowerCase();
      tokens.forEach(function (tok) {
        if (tok.length < 2) return;
        if (titleLower.indexOf(tok) !== -1) score += 4;
      });

      // Body text match (low weight) — only non-stopword tokens
      var bodyText = stripHtml(entry.body).toLowerCase();
      tokens.forEach(function (tok) {
        if (tok.length < 3) return;
        if (bodyText.indexOf(tok) !== -1) score += 1;
      });

      if (score > bestScore) {
        bestScore = score;
        best = entry;
      }
    });

    return bestScore >= 4 ? best : null;
  }

  function normalize(str) {
    return str.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
  }

  function tokenize(str) {
    return str.split(' ').filter(function (w) { return w.length > 0; });
  }

  function stripHtml(html) {
    var tmp = document.createElement('div');
    tmp.innerHTML = html; // eslint-disable-line -- stripping trusted KB HTML to text
    return tmp.textContent || tmp.innerText || '';
  }

  /* ── Message rendering ─────────────────────────── */
  function addUserMessage(text) {
    var container = document.getElementById('posi-assistant-messages');
    if (!container) return;

    var div = document.createElement('div');
    div.className = 'posi-assistant-msg posi-assistant-msg-q';
    var span = document.createElement('span');
    span.textContent = text; // Safe: user input rendered as textContent
    div.appendChild(span);
    container.appendChild(div);
    scrollToBottom(container);
  }

  // Renders trusted developer-authored HTML from the knowledge base.
  // This function is ONLY called with content from assistant-data.js
  // (a static file shipped with the site), never with user input.
  function addBotMessage(trustedBodyHtml) {
    var container = document.getElementById('posi-assistant-messages');
    if (!container) return;

    var div = document.createElement('div');
    div.className = 'posi-assistant-msg posi-assistant-msg-a';

    var avatar = document.createElement('img');
    avatar.className = 'posi-assistant-msg-avatar';
    avatar.src = '/logo.png';
    avatar.alt = '';
    div.appendChild(avatar);

    // Trusted KB content only — see file header comment
    var body = trustedHtmlEl('div', trustedBodyHtml, 'posi-assistant-msg-body');
    div.appendChild(body);

    container.appendChild(div);
    scrollToBottom(container);
  }

  function showTypingThenAnswer(query, match) {
    var container = document.getElementById('posi-assistant-messages');
    if (!container) return;

    // Build typing indicator using DOM methods
    var typing = document.createElement('div');
    typing.className = 'posi-assistant-typing';
    var tAvatar = document.createElement('img');
    tAvatar.className = 'posi-assistant-msg-avatar';
    tAvatar.src = '/logo.png';
    tAvatar.alt = '';
    typing.appendChild(tAvatar);

    var dots = document.createElement('div');
    dots.className = 'posi-assistant-dots';
    for (var i = 0; i < 3; i++) {
      var dot = document.createElement('div');
      dot.className = 'posi-assistant-dot';
      dots.appendChild(dot);
    }
    typing.appendChild(dots);
    container.appendChild(typing);
    scrollToBottom(container);

    setTimeout(function () {
      if (typing.parentNode) typing.parentNode.removeChild(typing);

      if (match) {
        addBotMessage(match.body);
        saveHistory(query, match.id);
      } else {
        addBotMessage(
          '<p>No exact match found for that query.</p>'
          + '<p style="margin-top:6px">Try keywords like:</p>'
          + '<ul style="margin:4px 0 8px 16px;line-height:1.7">'
          + '<li><strong>wallet, staking, tokenomics</strong></li>'
          + '<li><strong>PoNC, AI models, ZKML</strong></li>'
          + '<li><strong>PRC-20, bridge, DePIN</strong></li>'
          + '<li><strong>RPC, SDK, smart contracts</strong></li>'
          + '</ul>'
          + '<p style="font-size:12px;opacity:0.5">Or browse the categories below.</p>'
        );
        saveHistory(query, null);
      }
    }, TYPING_DELAY);
  }

  function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
  }

  /* ── Session history ───────────────────────────── */
  function saveHistory(query, entryId) {
    history.push({ q: query, id: entryId, t: Date.now() });
    if (history.length > 20) history = history.slice(-20);
    try {
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) { /* quota exceeded */ }
  }

  function restoreHistory() {
    try {
      var stored = sessionStorage.getItem(HISTORY_KEY);
      if (!stored) return;
      history = JSON.parse(stored);
      var data = getData();

      history.forEach(function (item) {
        addUserMessage(item.q);
        if (item.id) {
          var entry = data.entries.find(function (e) { return e.id === item.id; });
          if (entry) addBotMessage(entry.body);
          else addBotMessage('<p><em>(Answer no longer available)</em></p>');
        } else {
          addBotMessage('<p>I didn\'t find an exact match for that question.</p>');
        }
      });

      if (history.length > 0) sessionStorage.setItem(SHOWN_KEY, '1');
    } catch (e) { /* corrupt data */ }
  }

  /* ── Initialize on DOM ready ───────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
