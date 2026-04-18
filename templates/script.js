// --- Configure Marked.js options ---
marked.setOptions({
    breaks: true, // Render <br> on single line breaks
    gfm: true,    // GitHub Flavored Markdown (Tables, etc.)
    headerIds: false,
    mangle: false
});

// --- Highlight.js code renderer hook ---
// Intercepts marked's code block rendering and pipes it through hljs.
// Handles both old marked (code, lang args) and new marked (token object).
// FIX: Cache highlighted code blocks to prevent re-highlighting during streaming
const _highlightCache = new Map();
marked.use({
    renderer: {
        code(token) {
            const text = typeof token === 'object' ? (token.text || '') : token;
            const lang = (typeof token === 'object' ? token.lang : arguments[1]) || '';
            
            // Create a cache key from the code text + language
            const cacheKey = `${lang}:${text}`;
            
            // Check if we've already highlighted this exact code block
            if (_highlightCache.has(cacheKey)) {
                return _highlightCache.get(cacheKey);
            }
            
            let highlighted;
            try {
                if (lang && typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
                    highlighted = hljs.highlight(text, { language: lang }).value;
                } else if (typeof hljs !== 'undefined') {
                    highlighted = hljs.highlightAuto(text).value;
                } else {
                    highlighted = text;
                }
            } catch(e) {
                highlighted = text;
            }
            
            const result = `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`;
            
            // Cache the result for future renders
            _highlightCache.set(cacheKey, result);
            
            // Prevent cache from growing unbounded - keep last 100 entries
            if (_highlightCache.size > 100) {
                const firstKey = _highlightCache.keys().next().value;
                _highlightCache.delete(firstKey);
            }
            
            return result;
        }
    }
});

// ===== TOAST SYSTEM =====
// Note: #toast-container is declared statically in index.html — no dynamic creation needed.

function showToast(msg, type = 'info', duration = 2800) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const icons = { success: '✓', error: '✕', info: 'ℹ', warn: '⚠' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    // FIX #11: Build the toast DOM manually so the message text is always set via
    // textContent, never innerHTML. Server error strings (data.error) come from
    // the backend and could contain HTML — setting them via innerHTML would be XSS.
    const iconSpan = document.createElement('span');
    iconSpan.className = 'toast-icon';
    iconSpan.textContent = icons[type] || icons.info;
    const msgSpan = document.createElement('span');
    msgSpan.className = 'toast-msg';
    msgSpan.textContent = msg;  // textContent = safe, never parsed as HTML
    toast.appendChild(iconSpan);
    toast.appendChild(msgSpan);

    container.appendChild(toast);

    const remove = () => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    };

    const timer = setTimeout(remove, duration);
    toast.addEventListener('click', () => { clearTimeout(timer); remove(); });
}

// ===== RIPPLE EFFECT =====
function addRipple(btn, e) {
    const rect = btn.getBoundingClientRect();
    const ripple = document.createElement('span');
    ripple.className = 'ripple';
    ripple.style.left = (e.clientX - rect.left) + 'px';
    ripple.style.top  = (e.clientY - rect.top)  + 'px';
    btn.appendChild(ripple);
    ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
}

// Attach ripple to all grid-btns (also covers dynamically added ones via delegation)
document.addEventListener('click', function(e) {
    const btn = e.target.closest('.grid-btn, .shape-pill, .vanilla-mode-btn, .chunk-unit-btn');
    if (btn) addRipple(btn, e);
});
// ===== END TOAST & RIPPLE =====

let isSending = false;
let _streamAbortController = null;
let _generationId = 0;
let currentActivePersonaName = "Rivet"; // Must match backend default in app.py
let attachedFiles = [];  // Multi-file array (max 10)
let pasteToFileLength = 0; // 0 = disabled; loaded from backend on startup

// --- Backend mode cache (module-level so it persists across modal open/close) ---
// Helper — true when OpenRouter mode is active
function _isOpenRouter() {
    return document.getElementById('backendModeOpenRouter')?.checked || false;
}
// Gray out / restore a wrapper div
function _setKoboldOnly(divId, disabled) {
    const el = document.getElementById(divId);
    if (!el) return;
    el.style.opacity = disabled ? '0.35' : '1';
    el.style.pointerEvents = disabled ? 'none' : '';
    el.querySelectorAll('input').forEach(i => i.disabled = disabled);
}
let _koboldCache     = { endpoint: 'http://127.0.0.1:5001/v1/chat/completions', model: '' };
let _openRouterCache = { endpoint: 'https://openrouter.ai/api/v1/chat/completions', model: '', apiKey: '' };
let _backendModeListenerAdded = false;  // guard against stacking listeners on re-open
let pendingConfirmationAction = null;
const PANELS_VISIBLE_KEY = 'chatPanelsVisible';

// --- Search Tool Token Detection ---
let searchToolTrigger = '<tool_search>';  // JS-BUG-1 FIX: match backend default (was '[TOOL_SEARCH:')
let searchToolCloser = '</tool_search>';
let recallToolTrigger = '[RECALL:';
let recallToolCloser = ']';
let searchResultHeader = '[Search Results]:';
let recallResultHeader = '[Recall Results]:';

let currentUserName = "User";          // meta backbone — memory recalls, backend context. Never overwritten by loadout.
let currentLoadoutDisplayName = "User"; // visual only — active loadout's display_name, drives bubble nametags.
let assistantName = "Assistant";
let userAvatar = '❄️';
let assistantAvatar = '🐺';

// Avatar shape class list — defined here so it's available to appendMessage(),
// send(), and _applyAvatarShapeToDOM() without relying on declaration order.
const AVATAR_SHAPE_CLASSES = ['avatar-circle', 'avatar-rounded', 'avatar-square', 'avatar-seamless'];
let thinkingIndicatorText = "🦊";
let thinkingInterval = null; 
let thinkVisibility = 'visible'; // 'visible' | 'hidden'
let catchFailedThinking = true;  // Flag unclosed think blocks as "Failed to think ❌"

// Custom think block tokens — frontend display layer only.
// Backend always uses <think>/<\/think> for KV cache / stripping — do NOT change those.
// These are EXTRA open/close tokens the frontend regex will also match.
// Empty string = disabled (fall back to hardcoded defaults only).
let thinkOpenToken  = '';   // e.g. '<|thinking|>'  or  '[THINK_START]'
let thinkCloseToken = '';   // e.g. '<|/thinking|>' or  '[THINK_END]'

// --- Scroll tracking ---
// Simplified: arrow = manual one-shot, visibility = passive position check only.

let streamingSettings = {
    streamingDelayEnabled: true,   // true = typewriter (casual/roleplay), false = raw TPS (coder)
    charDelay: 10,          // FIX BUG 6: was 15 — matches backend default in streaming_config.json
    punctuationDelay: 40,   // FIX BUG 6: was 180 — matches backend default
    commaDelay: 20          // FIX BUG 6: was 150 — matches backend default
};

// Server-side appearance cache — populated on load, updated on save.
// Used by synchronous hot paths (bubble creation, streaming) that can't await.
let _appearanceCache = {
    assistantAvatarShape: 'circle',
    userAvatarShape: 'circle',
    assistantAvatar: '🐺',
    thinkingIndicator: '🦊',
    avatarSize: 34,
    fontSize: 15,
};

let editingMemoryIndex = null;
let editingVisibleIntroIndex = null;

// --- Persistent thought-box state (survives re-renders during streaming) ---
// Key: the assistant bubble element. Value: array of booleans (expanded per box index).
// Declared here (top-level) so event listeners registered at parse-time can safely reference it.
const thoughtExpandedStates = new WeakMap();

// Top-level avatar helper — shared by appendMessage and send()
function avatarContent(avatarVal) {
    if (!avatarVal) return '❄️';
    if (avatarVal.startsWith('data:image') || avatarVal.startsWith('/') || avatarVal.startsWith('http')) {
        return `<img src="${avatarVal}" alt="avatar" style="width:100%;height:100%;object-fit:cover;">`;
    }
    return avatarVal; // emoji fallback
}

// --- Vanilla Mode ---
// 'off'            = full fancy (default)
// 'pure'           = no avatar, no name, raw text only
// 'nametag-user'   = no avatar/emoji, but show user nametag only
// 'nametag-both'   = no avatar/emoji, show both user + persona nametags
// 'nametag-noicon' = show both nametags as plain text, NO emoji/image avatars
let vanillaMode = 'off';

function applyVanillaMode(mode) {
    vanillaMode = mode || 'off';
    const body = document.body;
    body.classList.remove('vanilla-pure', 'vanilla-nametag-user', 'vanilla-nametag-both', 'vanilla-nametag-noicon');
    if (mode && mode !== 'off') body.classList.add('vanilla-' + mode);
    // Update toggle buttons in modal
    document.querySelectorAll('.vanilla-mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === (mode || 'off'));
    });
    // Update description label
    const desc = document.getElementById('vanillaModeDesc');
    if (desc) {
        const descs = {
            'off':           '✨ <strong>Full Fancy</strong> — avatars, emoji, images, everything on.',
            'nametag-user':  '🏷️ <strong>User Tag Only</strong> — no avatars or icons anywhere. Only the user side shows a plain name label.',
            'nametag-both':  '🏷️🏷️ <strong>Both Tags</strong> — no avatars. Both user and assistant show a plain name label above their message.',
            'nametag-noicon':'🔤 <strong>Tags, No Icons</strong> — name labels for both sides. Avatars replaced with a single initial letter — no emoji or images.',
            'pure':          '⬜ <strong>Pure Vanilla</strong> — completely soulless corporate mode. No names, no icons, no avatars. Just raw text.',
        };
        desc.innerHTML = descs[mode || 'off'] || '';
    }
}

/** Builds the avatar+nametag HTML for a bubble, respecting vanillaMode */
/** Escapes a string for safe injection into innerHTML. Prevents XSS from
 *  user-controlled strings like persona names and loadout display_names.
 *  Called by buildBubbleHeader for every nametag render path. */
function _escHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#39;');
}

function buildBubbleHeader(cls, name) {
    const isUser = cls === 'user';
    const avatarVal = isUser ? userAvatar : assistantAvatar;
    const safeName = _escHtml(name);  // FIX BUG 12: escape before innerHTML injection

    if (vanillaMode === 'pure') return ''; // nothing

    if (vanillaMode === 'nametag-user') {
        // Only user gets a plain text nametag, no avatar
        if (isUser) return `<span class="bubble-name user-name">${safeName}</span>`;
        return '';
    }

    if (vanillaMode === 'nametag-both') {
        // Both get plain text nametags, no avatar/icon
        return `<span class="bubble-name ${isUser ? 'user-name' : 'assist-name'}">${safeName}</span>`;
    }

    if (vanillaMode === 'nametag-noicon') {
        // Both get nametags, avatars are text-initial only (no emoji, no image)
        const initial = _escHtml((name || '?')[0].toUpperCase());
        return `<span class="avatar avatar-initial">${initial}</span><span class="bubble-name ${isUser ? 'user-name' : 'assist-name'}">${safeName}</span>`;
    }

    // Default: full fancy avatar
    return `<span class="avatar">${avatarContent(avatarVal)}</span>`;
}

// --- Scroll to Bottom Logic ---
const chatlog = document.getElementById("chatlog");
const scrollBtn = document.getElementById("scrollToBottomBtn");

// Event delegation for thought container toggle — DOMPurify-safe, no inline onclick needed
if (chatlog) {
    // FIX: Thought-box toggle uses a movement-threshold gesture detector instead of
    // raw 'pointerdown'. The original problem: 'click' gets cancelled during streaming
    // because the typewriter replaces innerHTML every ~15ms, destroying the element
    // between mousedown and mouseup. Using raw 'pointerdown' fixed that but was too
    // aggressive — any graze while scrolling would toggle the box.
    //
    // Solution: delegate pointerdown + pointermove + pointerup all on chatlog (which
    // is never replaced by streaming). Record start position on pointerdown; if the
    // pointer moves more than SCROLL_THRESHOLD px before pointerup, treat it as a
    // scroll and suppress the toggle. Single tap still works naturally.
    const THOUGHT_SCROLL_THRESHOLD = 10; // px — below this = tap, above = scroll
    let _thoughtGesture = null; // { x, y, scrolled } — live while finger is down

    chatlog.addEventListener("pointerdown", (e) => {
        const container = e.target.closest(".thought-container");
        if (!container) return;
        if (container.classList.contains('thought-failed')) return;
        _thoughtGesture = { x: e.clientX, y: e.clientY, scrolled: false };
    });

    chatlog.addEventListener("pointermove", (e) => {
        if (!_thoughtGesture || _thoughtGesture.scrolled) return;
        const dx = Math.abs(e.clientX - _thoughtGesture.x);
        const dy = Math.abs(e.clientY - _thoughtGesture.y);
        if (dx > THOUGHT_SCROLL_THRESHOLD || dy > THOUGHT_SCROLL_THRESHOLD) {
            _thoughtGesture.scrolled = true;
        }
    });

    chatlog.addEventListener("pointerup", (e) => {
        const gesture = _thoughtGesture;
        _thoughtGesture = null;
        if (!gesture || gesture.scrolled) return; // was a scroll, not a tap
        if (window.getSelection().toString().length > 0) return;

        const container = e.target.closest(".thought-container");
        if (!container) return;
        if (container.classList.contains('thought-failed')) return;

        container.classList.toggle("expanded");

        // Write to the persistent Map so streaming re-renders can restore this state
        const bubble = container.closest(".chat-bubble");
        if (bubble) {
            const allBoxes = [...bubble.querySelectorAll(".thought-container")];
            const states = allBoxes.map(el => el.classList.contains("expanded"));
            thoughtExpandedStates.set(bubble, states);
        }
    });

    chatlog.addEventListener("pointercancel", () => { _thoughtGesture = null; });

    // Copy button stays on 'click' — it doesn't race the typewriter and needs
    // the full click gesture to avoid accidental copies on scroll/drag.
    chatlog.addEventListener("click", (e) => {
        const copyBtn = e.target.closest('[data-action="copy-code"]');
        if (copyBtn) { copyCode(copyBtn); return; }
    });
}

if (chatlog && scrollBtn) {
    chatlog.addEventListener("scroll", () => {
        const distanceToBottom = chatlog.scrollHeight - chatlog.scrollTop - chatlog.clientHeight;
        scrollBtn.style.display = distanceToBottom > 200 ? "flex" : "none";
    });
}

function scrollToBottom() {
    const chatlog = document.getElementById("chatlog");
    scrollBtn.style.display = "none";
    chatlog.scrollTop = chatlog.scrollHeight;
}



async function togglePanels() {
    // Optimistic update — feels instant. Reverted below if the save fails.
    document.body.classList.toggle('panels-hidden');
    const isHidden = document.body.classList.contains('panels-hidden');
    try {
        const res = await fetch('/set_appearance_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ panelsVisible: !isHidden })
        });
        if (!res.ok) {
            // Revert DOM change so UI and saved state stay in sync
            document.body.classList.toggle('panels-hidden');
            showToast('Could not save panel state', 'error');
        } else {
            showToast(isHidden ? 'Panels hidden' : 'Panels shown', 'info', 1500);
        }
    } catch(e) {
        // Revert on network error too
        document.body.classList.toggle('panels-hidden');
        console.warn('Could not save panel state:', e);
        showToast('Could not save panel state', 'error');
    }
}

async function loadPanelState() {
    try {
        const res = await fetch('/get_appearance_settings');
        if (res.ok) {
            const data = await res.json();
            if (data.panelsVisible === false) {
                document.body.classList.add('panels-hidden');
            } else {
                document.body.classList.remove('panels-hidden');
            }
        }
    } catch(e) { console.warn('Could not load panel state:', e); }
}

async function updateAllStatusUI() {
    await updateActiveCharacterNameUI();
}

function openControlsModal() {
    document.getElementById("controlsModal").style.display = "flex";
}

function closeControlsModal() {
    document.getElementById("controlsModal").style.display = "none";
}

function startThinkingAnimation(indicatorElement) {
    if (thinkingInterval) {
        clearInterval(thinkingInterval);
    }

    let state = 0;
    const states = ['', thinkingIndicatorText, thinkingIndicatorText.repeat(2)];
    
    if (indicatorElement) {
        indicatorElement.textContent = states[state];

        thinkingInterval = setInterval(() => {
            state = (state + 1) % states.length;
            indicatorElement.textContent = states[state];
        }, 600); 
    }
}

function stopThinkingAnimation(indicatorElement) {
    if (thinkingInterval) {
        clearInterval(thinkingInterval);
        thinkingInterval = null;
    }
    if(indicatorElement && indicatorElement.parentElement) {
        indicatorElement.remove();
    }
}

// --- Copy function ---
async function copyCode(button) {
    // Use closest() instead of parentElement — more robust if DOM shifts slightly
    const codeContainer = button.closest('.code-container');
    // textContent instead of innerText — hljs wraps tokens in <span>s,
    // innerText can misbehave on non-visible elements; textContent is raw and reliable
    const codeBlock = codeContainer.querySelector('code');
    const textToCopy = codeBlock.textContent;
    const buttonText = button.querySelector('span');

    const onSuccess = () => {
        buttonText.textContent = 'Copied!';
        button.style.backgroundColor = 'var(--success-color)';
        button.style.color = 'white';
        setTimeout(() => {
            buttonText.textContent = 'Copy';
            button.style.backgroundColor = '';
            button.style.color = '';
        }, 2000);
    };

    const onFail = (err) => {
        console.error('Failed to copy text: ', err);
        buttonText.textContent = 'Error';
        setTimeout(() => { buttonText.textContent = 'Copy'; }, 2000);
    };

    // navigator.clipboard only works on HTTPS or localhost (secure context).
    // Flask on http://0.0.0.0:5000 is NOT a secure context so clipboard is undefined.
    // Fallback: create a temporary textarea, select it, execCommand('copy') — works on HTTP.
    if (navigator.clipboard && window.isSecureContext) {
        try {
            await navigator.clipboard.writeText(textToCopy);
            onSuccess();
        } catch (err) {
            onFail(err);
        }
    } else {
        try {
            const ta = document.createElement('textarea');
            ta.value = textToCopy;
            ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none;';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            onSuccess();
        } catch (err) {
            onFail(err);
        }
    }
}

function formatMarkdown(text, streamComplete = false) {
    if (!text) return '';

    // -1. VOID SAFEHOUSE — Result blocks are aliens. Zero tag privileges.
    // Runs BEFORE code blocks and BEFORE anything else so no content inside
    // (nested <think>, markdown, code fences, custom tokens) ever touches the
    // rendering pipeline. Extracted into the void and NEVER restored.
    //
    // Termination priority:
    //   1. Explicit [/End...Results] closer      (clean, expected)
    //   2. The next result header                 (back-to-back injections)
    //   3. A paragraph break \n\n                 (graceful degradation — limits blast radius)
    //
    // We do NOT use $ as fallback. If the backend sends an unterminated block,
    // we sacrifice only that paragraph, not the entire rest of the message.
    // -1 operates on `text` directly — processedText hasn't been declared yet.
    let _safeText = text;
    if (searchResultHeader || recallResultHeader) {
        const _voidHeaders  = [searchResultHeader, recallResultHeader].filter(Boolean);
        const _voidEscaped  = _voidHeaders.map(h => h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

        _voidHeaders.forEach((header, i) => {
            const _esc = _voidEscaped[i];
            const _stopLookahead = [
                '\\[\\/End[^\\]]*?Results?[^\\]]*?\\]',
                ..._voidEscaped,
                '\\n\\n'
            ].join('|');
            _safeText = _safeText.replace(
                new RegExp(`${_esc}[\\s\\S]*?(?=${_stopLookahead})`, 'gi'),
                ''
            );
            _safeText = _safeText.replace(/\[\/End[^\]]*?Results?[^\]]*?\]/gi, '');
        });
    }
    // ↑ Content is gone — never restored. As intended.

    // 0. SAFEHOUSE: Extract code fences FIRST before any regex hunts.
    // Anything inside ``` ``` is a safehouse — think regex cannot enter.
    // We pull them out, replace with placeholders, run all regex on open field, then restore.
    const codeBlocks = [];
    let processedText = _safeText.replace(/```[\s\S]*?```/g, (match) => {
        const id = `:::CODE_BLOCK_${codeBlocks.length}:::`;
        codeBlocks.push(match);
        return id;
    });

    // 0.1. STREAMING FENCE SEAL: Handle unclosed/partial code fences.
    // The extraction above only removes COMPLETE ``` ... ``` pairs. During streaming,
    // the AI is mid-block — the closing ``` hasn't arrived yet. Any remaining fence
    // opener in processedText is orphaned, which means its content gets passed raw into
    // marked.parse(). marked then happily renders | col | as a table, [x](url) as a link,
    // **bold** as bold — inside what should be an inert code block. This is the "seizure"
    // / "inception" visual glitch: markdown rendering flickering in and out as chars stream.
    //
    // FIX: After complete-pair extraction, count remaining fence openers. If odd (= unclosed),
    // temporarily insert a closing ``` BEFORE the first "ejection seat" token encountered
    // after the orphaned opener — or at end-of-string if none are found.
    //
    // EJECTION SEATS: All customizable + default special tokens act as forced fence closers.
    // Think openers, search/recall triggers, and result headers should NEVER be swallowed
    // by a greedy code block. If any of them appear after an unclosed opener, the fence
    // is sealed immediately before them so the widgets render correctly above the code.
    //
    // This is render-only — fullResponse is never mutated, so the real stream is untouched.
    //
    // GUARD: Only seal during active streaming (streamComplete === false).
    // When the message is finished (reloadChat / appendMessage calls formatMarkdown(text, true)),
    // the content is final. If the AI wrote a standalone code block that never had a closing
    // fence (e.g. a Python def or shell snippet), auto-closing produces a ghost empty code
    // container at the bottom of the bubble. Leave it as-is when streaming is done —
    // the inception glitch only occurs mid-stream anyway.
    if (!streamComplete) {
        const _remainingFences = processedText.match(/^`{3,}[^\n]*$/gm);
        if (_remainingFences && _remainingFences.length % 2 !== 0) {

            // Build the ejection seat token list: all customizable + hardcoded special headers.
            // These are tokens that must NEVER be rendered inside a code block.
            const _ejectionTokens = [
                '<think>', '[THINK]',                   // hardcoded think openers
                thinkOpenToken,                          // custom think opener (if set)
                searchToolTrigger, recallToolTrigger,    // tool call tokens
                searchResultHeader, recallResultHeader   // result injection headers
            ].filter(Boolean); // strip empty strings (unconfigured custom tokens)

            // Find the position of the LAST unclosed fence opener in processedText.
            // We scan all fence-like lines and take the last one (the orphaned opener).
            let _lastFencePos = -1;
            const _fenceScan = /^`{3,}[^\n]*$/gm;
            let _fm;
            while ((_fm = _fenceScan.exec(processedText)) !== null) {
                _lastFencePos = _fm.index;
            }

            // Among all ejection seat tokens, find the earliest one that appears
            // AFTER the orphaned opener. That's where we slam the fence shut.
            let _insertAt = processedText.length; // default: end-of-string (old behaviour)
            if (_lastFencePos !== -1) {
                for (const _tok of _ejectionTokens) {
                    const _pos = processedText.indexOf(_tok, _lastFencePos);
                    if (_pos !== -1 && _pos < _insertAt) {
                        _insertAt = _pos;
                    }
                }
            }

            // Splice the closing fence in at the right position.
            // If no ejection seat was found, _insertAt === length → same as old '\n```' append.
            processedText =
                processedText.slice(0, _insertAt) +
                '\n```\n' +
                processedText.slice(_insertAt);
        }
    }

    // 0.5. AUTO-CLOSE THINKING BLOCKS (PRE-PARSE)
    // Force-close any unclosed <think> tags when result headers OR tool triggers appear.
    // Tool trigger auto-close is FRONTEND ONLY — never injected into memory or KV cache.
    // This MUST happen BEFORE the thinking wrapper processes them.
    const _autoCloseTokens = [
        searchResultHeader, recallResultHeader,  // result headers (existing)
        searchToolTrigger, recallToolTrigger      // tool triggers (new — visual only)
    ].filter(Boolean);

    if (_autoCloseTokens.length > 0) {
        _autoCloseTokens.forEach(token => {
            if (processedText.includes(token)) {
                const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                // Include custom open token if configured
                const _customOpenEsc = thinkOpenToken ? '|' + thinkOpenToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : '';
                const autoClose = new RegExp(`(<think>|\\[THINK\\]${_customOpenEsc})([\\s\\S]*?)(${escapedToken})`, 'gi');
                // Include custom close token in already-closed check
                const _customCloseEsc = thinkCloseToken ? '|' + thinkCloseToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : '';
                const closeCheck = new RegExp(`<\\/think>|<\\/think\\}|\\[\\/THINK\\]${_customCloseEsc}`, 'i');
                processedText = processedText.replace(autoClose, (match, openTag, content, tokenMatch) => {
                    if (content.match(closeCheck)) {
                        return match;
                    }
                    return `${openTag}${content}</think>\n\n${tokenMatch}`;
                });
            }
        });
    }

    // 1c. SEARCH TOOL TOKEN DETECTION (PRE-PARSE)
    // Wrap tool calls AND any trailing yapping (the "tail") into styled blockquotes
    // Everything between [TOOL_SEARCH:...] and [Search Results]: gets wrapped
    if (searchToolTrigger && searchToolCloser) {
        const escapedTrigger = searchToolTrigger.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const escapedCloser = searchToolCloser.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

        const headers = [searchResultHeader, recallResultHeader].filter(Boolean);
        if (headers.length > 0) {
            const escapedHeaders = headers.map(h => h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
            const headerPattern = escapedHeaders.join('|');

            // Pass 1: token + tail up to a result header (pinned/persisted bubble case)
            const widgetPattern = new RegExp(
                `${escapedTrigger}(.*?)${escapedCloser}([\\s\\S]*?)(?=${headerPattern})`,
                'g'
            );
            processedText = processedText.replace(widgetPattern, (match, query, tail) => {
                const cleanQuery = query.trim().replace(/^["\'\']|["\'\']$/g, '');
                const cleanTail = tail.trim();
                let widget = `\n\n> 🕵️ **Searching:** *"${cleanQuery}"*`;
                if (cleanTail) widget += `\n> ${cleanTail}`;
                widget += `\n\n`;
                return widget;
            });
        }
        // Pass 2: catch any remaining naked tokens (ephemeral mode — no result header in bubble)
        const nakedSearchRegex = new RegExp(`${escapedTrigger}(.*?)${escapedCloser}`, 'g');
        processedText = processedText.replace(nakedSearchRegex, (match, query) => {
            const cleanQuery = query.trim().replace(/^["\'\']|["\'\']$/g, '');
            return `\n\n> 🕵️ **Searching:** *"${cleanQuery}"*\n\n`;
        });
    }

    // 1e. RECALL TOOL TOKEN DETECTION (PRE-PARSE)
    if (recallToolTrigger && recallToolCloser) {
        const escapedTrigger = recallToolTrigger.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const escapedCloser  = recallToolCloser.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

        const recallHeaders = [searchResultHeader, recallResultHeader].filter(Boolean);
        if (recallHeaders.length > 0) {
            const escapedHeaders = recallHeaders.map(h => h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
            const headerPattern  = escapedHeaders.join('|');
            // Pass 1: token + tail up to a result header (merged bubble case)
            const widgetPattern  = new RegExp(
                `${escapedTrigger}(.*?)${escapedCloser}([\\s\\S]*?)(?=${headerPattern})`,
                'g'
            );
            processedText = processedText.replace(widgetPattern, (match, query, tail) => {
                const cleanQuery = query.trim().replace(/^["\'\']|["\'\']$/g, '');
                const cleanTail  = tail.trim();
                let widget = `\n\n> 🔮 **Recalling:** *"${cleanQuery}"*`;
                if (cleanTail) widget += `\n> ${cleanTail}`;
                widget += `\n\n`;
                return widget;
            });
        }
        // Pass 2: catch any remaining naked tokens (standalone pre-tool bubble, or no header)
        const nakedRegex = new RegExp(`${escapedTrigger}(.*?)${escapedCloser}`, 'g');
        processedText = processedText.replace(nakedRegex, (match, query) => {
            const cleanQuery = query.trim().replace(/^["\'\']|["\'\']$/g, '');
            return `\n\n> 🔮 **Recalling:** *"${cleanQuery}"*\n\n`;
        });
    }

    // 1. HARD-CODED THINKING WRAPPER (PRE-PARSE)
    // We do this BEFORE marked.parse so that the markdown inside doesn't break the wrapper.
    // This looks for <think> and finds the first instance of </think>, </think}, or end of string.
    // NOTE: Code blocks are already evacuated above — think regex only hunts in the open field.
    // NOTE: Result blocks (search/recall) are already stripped above — no nested think tag conflicts.
    // Build think regex dynamically so custom tokens are included alongside hardcoded defaults.
    const _thinkOpenPart  = thinkOpenToken  ? '|' + thinkOpenToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')  : '';
    const _thinkClosePart = thinkCloseToken ? '|' + thinkCloseToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : '';
    const thinkRegex = new RegExp(
        `(?:<think>|\\[THINK\\]${_thinkOpenPart})([\\s\\S]*?)(?:<\\/think>|<\\/think\\}|\\[\\/THINK\\]${_thinkClosePart}|$)`,
        'gi'
    );

    processedText = processedText.replace(thinkRegex, (match, content) => {
        // We wrap the raw content. We'll parse the markdown inside this content later or let it be.
        // For now, we wrap it in a placeholder to protect it from the main markdown parser.
        const trimmedContent = content.trim();

        // Detect unclosed think block: check if the match actually contained a real closing tag.
        // If the regex matched to end-of-string ($) instead, there was no closing tag — failed think.
        const _closeDetectRx = new RegExp(
            `<\\/think>|<\\/think\\}|\\[\\/THINK\\]` +
            (thinkCloseToken ? '|' + thinkCloseToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : ''),
            'i'
        );
        const isClosed = _closeDetectRx.test(match);
        const failPrefix = (!isClosed && catchFailedThinking && streamComplete) ? ':::THINK_FAILED:::' : '';

        return `\n\n:::THOUGHT_START:::${failPrefix}${trimmedContent}:::THOUGHT_END:::\n\n`;
    });

    // 1b. RESTORE SAFEHOUSES: Put code blocks back before markdown parse.
    codeBlocks.forEach((block, i) => {
        processedText = processedText.replace(`:::CODE_BLOCK_${i}:::`, block);
    });

    // 2. Roleplay Asterisks
    processedText = processedText.replace(/(^|[\s\n])\*([^*]+?)\*([\s\n.,!?]|$)/g, '$1<em>$2</em>$3');

    // 3. Main Markdown Parse
    let html = marked.parse(processedText);

    // 4. TRANSFORM PLACEHOLDERS TO UI CONTAINERS
    // Now we turn those protected thoughts into the actual HTML boxes.
    html = html.replace(/:::THOUGHT_START:::([\s\S]*?):::THOUGHT_END:::/g, (match, content) => {
        // Check for failed-think marker (unclosed block)
        const isFailed = content.startsWith(':::THINK_FAILED:::');
        const rawContent = isFailed ? content.slice(':::THINK_FAILED:::'.length) : content;

        // INCEPTION FIX: Strip chat-UI class names from thought content so that any HTML
        // the model generates (e.g. while reasoning about a game/webpage) cannot re-trigger
        // the chat bubble, avatar, or layout CSS inside the thought box.
        const _uiClasses = new Set([
            'chat-bubble','grok','user','avatar','message-body','message-content',
            'thought-container','thought-header','thought-content','code-container',
            'input-container','input-box','chatbox','top-bar','status-bar'
        ]);
        const classStripped = rawContent.replace(/class="([^"]*)"/gi, (_, classes) => {
            const safe = classes.split(/\s+/).filter(c => !_uiClasses.has(c)).join(' ').trim();
            return safe ? `class="${safe}"` : '';
        });

        // STRUCTURAL LEAK FIX: The model's thinking can contain raw HTML closing tags
        // (e.g. </div>) which, when string-interpolated into the template below, break out
        // of the .thought-content div and cause <code>/<pre> elements to land as direct
        // children of .thought-container — outside the clipped wrapper, leaking visually.
        // Pre-sanitize the thought content with DOMPurify in its own isolated pass so the
        // resulting HTML is structurally self-contained before we embed it.
        const safeContent = (typeof DOMPurify !== 'undefined')
            ? DOMPurify.sanitize(classStripped, {
                ALLOWED_TAGS: ['p','br','strong','em','b','i','u','s','code','pre','span',
                               'div','ul','ol','li','blockquote','h1','h2','h3','h4','h5',
                               'h6','table','thead','tbody','tr','th','td','a','hr','img'],
                ALLOWED_ATTR: ['class','href','src','alt','title','data-action'],
                FORCE_BODY: true   // ensures output is a full fragment, not dangling tags
              })
            : classStripped;

        // FAILED THINK: unclosed block — render locked red badge, no dropdown
        if (isFailed) {
            return `
                <div class="thought-container thought-failed">
                    <div class="thought-header thought-header-failed">
                        <span>Failed to think ❌</span>
                        <span class="think-chevron-failed" title="Think block was never closed — generation may have been cut off">⚠</span>
                    </div>
                    <div class="thought-content-wrapper thought-failed-wrapper">
                        <div class="thought-content">${safeContent}</div>
                    </div>
                </div>
            `;
        }

        return `
            <div class="thought-container">
                <div class="thought-header">
                    <span>🧠 Thoughts...</span>
                    <span class="think-chevron">▾</span>
                </div>
                <div class="thought-content-wrapper">
                    <div class="thought-content">${safeContent}</div>
                </div>
            </div>
        `;
    });

    // 5. Sanitize HTML (DOMPurify)
    const purifyConfig = {
        ADD_TAGS: ['div', 'em', 'span', 'blockquote', 'button', 'svg', 'path', 'rect'],
        // JS-BUG-6 FIX: removed 'onclick' from ADD_ATTR — allowed LLM output to inject arbitrary JS.
        // Copy button now uses a data-action attribute resolved via event delegation instead.
        ADD_ATTR: ['class', 'data-action', 'viewBox', 'fill', 'stroke', 'd', 'x', 'y', 'width', 'height', 'rx', 'ry', 'xmlns']
    };
    let cleanHtml = DOMPurify.sanitize(html, purifyConfig);

    // 6. Cleanup Janitor: Remove any stray leaked tags (hardcoded + custom tokens)
    const _janitorCustom = [
        thinkOpenToken  ? thinkOpenToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')  : '',
        thinkCloseToken ? thinkCloseToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : ''
    ].filter(Boolean).join('|');
    const _janitorRx = new RegExp(
        '<\\/think>|<\\/think\\}|\\[THINK\\]|\\[\\/THINK\\]' + (_janitorCustom ? '|' + _janitorCustom : ''),
        'gi'
    );
    cleanHtml = cleanHtml.replace(_janitorRx, '');
    // Also purge any orphaned THOUGHT placeholders that slipped through
    // (can happen if marked.parse wraps the placeholder in unexpected block-level HTML)
    cleanHtml = cleanHtml.replace(/:::THOUGHT_START:::|:::THOUGHT_END:::/g, '');

    // 7. Search & Recall Widgets
    // Defensive: skip if icon is inside a <pre>/<code> block (AI misfire into backticks)
    // and avoid matching nested blockquotes.
    const processWidget = (html, icon, className) => {
        // If the icon only appears inside a pre/code block, bail out entirely
        const strippedOfCode = html.replace(/<pre[\s\S]*?<\/pre>/gi, '').replace(/<code[\s\S]*?<\/code>/gi, '');
        if (!strippedOfCode.includes(icon)) {
            return html; // icon was only inside a code block — skip
        }

        // Match a blockquote that directly contains the icon
        const regex = new RegExp(`(<blockquote>(?:(?!<blockquote>)[\\s\\S])*?${icon}(?:(?!<\\/blockquote>)[\\s\\S])*?<\\/blockquote>)([\\s\\S]*)`, 'i');
        const match = strippedOfCode.match(regex);
        if (match) {
            // Apply the class to the real html (not the stripped version)
            const blockquoteRegex = new RegExp(`(<blockquote>(?:(?!<blockquote>)[\\s\\S])*?${icon}(?:(?!<\\/blockquote>)[\\s\\S])*?<\\/blockquote>)`, 'i');
            const realMatch = html.match(blockquoteRegex);
            if (!realMatch) return html;
            const blockquotePart = realMatch[1];
            const contentAfter = html.slice(html.indexOf(blockquotePart) + blockquotePart.length);
            const textAfterCount = contentAfter.replace(/<[^>]*>/g, '').trim().length;
            const modified = blockquotePart.replace(
                '<blockquote>',
                `<blockquote class="${className}${textAfterCount > 5 ? ' collapsed' : ''}">`
            );
            return html.replace(blockquotePart, modified);
        }
        return html;
    };
    cleanHtml = processWidget(cleanHtml, '🕵️', 'search-widget');
    cleanHtml = processWidget(cleanHtml, '🔮', 'recall-widget');

    // 8. Code Blocks with Copy Button
    const codeBlockRegex = /<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/g;
    const copyIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2 2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
    
    return cleanHtml.replace(codeBlockRegex, (match, attrs) => {
        const langMatch = attrs.match(/language-(\w+)/);
        const lang = langMatch ? langMatch[1] : '';
        const langLabel = lang ? `<span class="code-lang">${lang}</span>` : '';
        return `
            <div class="code-container">
                ${langLabel}
                <button class="copy-btn" data-action="copy-code">
                   ${copyIcon}
                   <span>Copy</span>
                </button>
                ${match}
            </div>`;
    });
}

function appendMessage(name, text, cls, index = null, memIndex = null, msgId = null) {
    const log = document.getElementById("chatlog");
    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble", cls);

    if (cls === "system-notification") {
         bubble.innerHTML = `<span>${text}</span>`;
         log.appendChild(bubble);
         return bubble;
    }

    // Store msg_id on the bubble for stable identity
    if (msgId) bubble.dataset.msgId = msgId;

    // Build action buttons — hidden until hover
    let actionBtnsHtml = "";
    const editBtn = memIndex !== null
        ? `<button class="msg-action-btn" onclick="openEditMessageModal(${memIndex})" title="Edit">✏️</button>`
        : "";

    // Delete button — user messages use index, assistant messages use msg_id
    let deleteBtn = "";
    if (cls === "user" && index !== null) {
        const idAttr = msgId ? `, '${msgId}'` : "";
        deleteBtn = `<button class="msg-action-btn delete" onclick="deletePair(${index}${idAttr})" title="Delete">🗑</button>`;
    } else if (cls === "grok" && msgId) {
        // Assistant bubble — delete the whole turn by msg_id
        deleteBtn = `<button class="msg-action-btn delete" onclick="deletePair(null, '${msgId}')" title="Delete">🗑</button>`;
    }

    if (editBtn || deleteBtn) {
        actionBtnsHtml = `<div class="msg-actions">${editBtn}${deleteBtn}</div>`;
    }

    const headerHtml = buildBubbleHeader(cls, name);
    const renderedHtml = formatMarkdown(text, true);

    bubble.innerHTML = `${headerHtml}
                        <div class="message-body">
                            <div class="message-content">${renderedHtml}</div>
                        </div>
                        ${actionBtnsHtml}`;

    log.appendChild(bubble);
    // Apply current avatar shape immediately to the new bubble
    try {
        const who = cls === 'grok' ? 'assistant' : 'user';
        const shape = (who === 'assistant' ? _appearanceCache.assistantAvatarShape : _appearanceCache.userAvatarShape) || 'circle';
        const avatarEl = bubble.querySelector('.avatar');
        if (avatarEl) {
            AVATAR_SHAPE_CLASSES.forEach(c => avatarEl.classList.remove(c));
            avatarEl.classList.add(`avatar-${shape}`);
        }
    } catch(e) {}
    return bubble;
}

const delay = ms => new Promise(res => setTimeout(res, ms));

async function streamAssistantResponse(fetchPromise, assistantBubble, signal = null) {
    const chatlog = document.getElementById("chatlog");
    const contentSpan = assistantBubble.querySelector('.message-content');
    const thinkingIndicator = assistantBubble.querySelector('.thinking-indicator');

    // FIX #16: Render state lives here — shared between the chunk accumulator
    // and the per-character delay ticker so both paths update the same DOM node.
    function _renderAndRestoreStates(html) {
        // FIX LAYOUT SHIFT: Snapshot distance-from-bottom before the innerHTML wipe.
        // When a second thought block grows during streaming it pushes content below it
        // downward. With overflow-anchor:none the raw scrollTop pixel stays fixed, so the
        // viewport visually jumps UP. Saving/restoring distance-from-bottom keeps the user
        // pinned to wherever they were — mid-scroll or at the bottom — no fight-back.
        const distanceFromBottom = chatlog.scrollHeight - chatlog.scrollTop - chatlog.clientHeight;

        // FIX CODE BLOCK SCROLL: Save horizontal scroll position of each <pre> block
        // before the innerHTML wipe. Without this, every token during typewriter mode
        // resets scrollLeft to 0 — so the user can never scroll right in a code block
        // while the AI is still typing. We restore by index (same order, same count).
        const preScrollPositions = [];
        contentSpan.querySelectorAll('pre').forEach(pre => {
            preScrollPositions.push(pre.scrollLeft);
        });

        contentSpan.innerHTML = html;

        // Restore code block horizontal scroll positions — but DEFERRED via rAF.
        // If we set scrollLeft immediately after innerHTML, the browser hasn't done
        // layout yet so scrollWidth is still 0 and the assignment is silently dropped.
        // requestAnimationFrame fires after layout+paint, when scrollWidth is real.
        if (preScrollPositions.some(x => x > 0)) {
            requestAnimationFrame(() => {
                contentSpan.querySelectorAll('pre').forEach((pre, i) => {
                    if (preScrollPositions[i] > 0) pre.scrollLeft = preScrollPositions[i];
                });
            });
        }

        // --- RESTORE THOUGHT BOX STATE FROM MAP (click-safe, never lost) ---
        const savedStates = thoughtExpandedStates.get(assistantBubble) || [];
        contentSpan.querySelectorAll('.thought-container').forEach((el, i) => {
            const wrapper = el.querySelector('.thought-content-wrapper');
            if (savedStates[i]) {
                el.classList.add('expanded');
                // FIX: The 0.5s CSS transition on max-height re-fires from scratch every
                // ~15ms when the DOM is wiped & rebuilt during streaming. The 2nd/3rd/4th
                // thought boxes appear to "seizure" because they're perpetually animating.
                // Kill the transition on restore so the wrapper jumps instantly.
                // Use scrollHeight (dynamic) so no thought is ever cut off, no matter length.
                if (wrapper) {
                    wrapper.style.transition = 'none';
                    wrapper.style.maxHeight = wrapper.scrollHeight + 'px';
                    wrapper.style.overflowY = 'auto';
                    // scrollTop intentionally omitted — user controls thought-box scroll
                }
            } else {
                // CRITICAL: Clear inline style overrides so CSS collapsed state (54px) wins.
                // Without this, a stale inline max-height from a previous expand cycle
                // overrides the CSS and the box never visually collapses.
                if (wrapper) {
                    wrapper.style.maxHeight = '';
                    wrapper.style.overflowY = '';
                    wrapper.style.transition = '';
                    // scrollTop intentionally omitted — dead code (overflow:hidden)
                }
            }
        });

        // Restore user's scroll position after the DOM rebuild.
        // Neutralises layout shift from thought blocks growing above the viewport.
        chatlog.scrollTop = chatlog.scrollHeight - chatlog.clientHeight - distanceFromBottom;
    }

    // TOOL_FIRED sentinel handler — called the moment :::TOOL_FIRED::: arrives in the stream.
    // At this point fullResponse contains the raw LLM output up to the tool invocation.
    // 
    // FIX: Instead of folding orphaned text back INTO the think block (which causes widgets
    // to get trapped), we simply FORCE-CLOSE any open <think> block. This ensures that
    // everything that follows (tool widgets, search results, AI responses) stays OUTSIDE
    // the thought box during streaming.
    // 
    // The key insight: When a tool fires, we're transitioning from "thinking" to "acting".
    // The thought phase is DONE. Close the box and move on. Don't try to be clever.
    function _foldOrphanedReasoningIntoThink(text) {
        // Build the same open/close patterns as formatMarkdown uses
        const _customOpenEsc  = thinkOpenToken  ? '|' + thinkOpenToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')  : '';
        const _customCloseEsc = thinkCloseToken ? '|' + thinkCloseToken.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : '';
        const openRx  = new RegExp(`<think>|\\[THINK\\]${_customOpenEsc}`, 'i');
        const closeRx = new RegExp(`<\\/think>|<\\/think\\}|\\[\\/THINK\\]${_customCloseEsc}`, 'i');

        // Find the last think opener in the text
        const lastOpenMatch = [...text.matchAll(new RegExp(`(<think>|\\[THINK\\]${_customOpenEsc})`, 'gi'))].pop();
        if (!lastOpenMatch) return text; // no think block at all — nothing to close

        const openIdx = lastOpenMatch.index + lastOpenMatch[0].length;
        const afterOpen = text.slice(openIdx);

        // Check if this think block is already closed
        if (closeRx.test(afterOpen)) return text; // already closed — we're good

        // *** THE FIX: Force-close the unclosed think block ***
        // This prevents tool widgets and search results from being captured inside
        // the thought box during subsequent streaming. Everything after the sentinel
        // will now stay outside the </think> tag.
        console.log('🔧 TOOL_FIRED: Force-closing unclosed <think> block to prevent widget trapping');
        return text + '\n</think>\n\n';
        
        // OLD CODE REMOVED (kept in backup files):
        // The old code tried to fold orphaned reasoning back into the think block.
        // This caused widgets to get trapped because formatMarkdown() runs on the
        // ENTIRE accumulated fullResponse every iteration, so unclosed <think> blocks
        // would capture everything that streams in afterward (widgets, results, etc).
    }

    try {
        const res = await fetchPromise;

        // FIX #10: Check HTTP status before touching the body.
        // A 4xx/5xx error sends a JSON error body — streaming it char-by-char
        // into the bubble would display raw JSON as if it were a chat reply.
        if (!res.ok) {
            let errMsg = `Server error (${res.status})`;
            try { const e = await res.json(); errMsg = e.error || errMsg; } catch (_) {}
            contentSpan.innerHTML = `<span style="color: #ff8a80;">Error: ${errMsg}</span>`;
            stopThinkingAnimation(thinkingIndicator);
            removeAttachedFile();
            return;
        }

        removeAttachedFile();

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";
        let isFirstChunk = true;
        // Buffer for assembling the sentinel token across chunk boundaries
        const SENTINEL = ':::TOOL_FIRED:::';
        let sentinelBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);

            if (isFirstChunk && chunk.length > 0) {
                stopThinkingAnimation(thinkingIndicator);
                isFirstChunk = false;
            }

            // ============================================================
            // ⚠️  DO NOT "OPTIMIZE" THIS LOOP — THIS IS THE TYPEWRITER EFFECT
            // ============================================================
            // Rendering per-character is 100% intentional. Each char append +
            // formatMarkdown() call IS what produces the buttery smooth keystroke
            // animation. If you batch this (render once per network chunk) the
            // entire chunk teleports in instantly — that's the corporate SSE blob
            // look, NOT a typewriter. The formatMarkdown() cost per char is
            // negligible (~1-2ms) vs the intentional delay (15ms+). Not a perf
            // problem. Do not batch. Do not "fix". Leave. It. Alone. 🙏
            // ============================================================
            // When delay mode is OFF we DO dump the whole chunk at once —
            // that's intentional too (raw TPS mode for coders who want speed).
            // ============================================================
            if (streamingSettings.streamingDelayEnabled) {
                for (const char of chunk) {
                    sentinelBuffer += char;

                    // Check if the sentinel is fully assembled in the buffer
                    if (sentinelBuffer.includes(SENTINEL)) {
                        // Fold orphaned reasoning back into the think box, strip the sentinel
                        const before = sentinelBuffer.slice(0, sentinelBuffer.indexOf(SENTINEL));
                        fullResponse += before;
                        fullResponse = _foldOrphanedReasoningIntoThink(fullResponse);
                        sentinelBuffer = sentinelBuffer.slice(sentinelBuffer.indexOf(SENTINEL) + SENTINEL.length);
                        _renderAndRestoreStates(formatMarkdown(fullResponse));
                        continue;
                    }

                    // Sentinel might be partially building — hold the buffer, don't render yet
                    if (SENTINEL.startsWith(sentinelBuffer) && sentinelBuffer.length > 0 && sentinelBuffer.length < SENTINEL.length) {
                        continue; // accumulating — wait for more chars
                    }

                    // Not a sentinel prefix — flush the buffer into fullResponse and render
                    fullResponse += sentinelBuffer;
                    sentinelBuffer = '';
                    _renderAndRestoreStates(formatMarkdown(fullResponse));
                    if (char === '.' || char === '!' || char === '?' || char === '—') {
                        await delay(streamingSettings.punctuationDelay);
                    } else if (char === ',') {
                        await delay(streamingSettings.commaDelay);
                    } else {
                        await delay(streamingSettings.charDelay);
                    }
                    if (signal && signal.aborted) { sentinelBuffer = ''; break; }
                }
            } else {
                // Raw TPS mode — check for sentinel in the whole chunk
                if (chunk.includes(SENTINEL)) {
                    const parts = chunk.split(SENTINEL);
                    // Everything before the sentinel — fold orphans
                    fullResponse += parts[0];
                    fullResponse = _foldOrphanedReasoningIntoThink(fullResponse);
                    // Everything after (there should only be one sentinel per chunk)
                    fullResponse += parts.slice(1).join('');
                } else {
                    fullResponse += chunk;
                }
                _renderAndRestoreStates(formatMarkdown(fullResponse));
            }
        }

        // Flush any remaining sentinelBuffer (stream ended mid-buffer — shouldn't happen but be safe)
        if (sentinelBuffer && !sentinelBuffer.includes(SENTINEL)) {
            fullResponse += sentinelBuffer;
            _renderAndRestoreStates(formatMarkdown(fullResponse));
        }

        const newSessionId = res.headers.get('X-Session-Renamed');
        if (newSessionId) {
            console.log(`Session renamed to: ${newSessionId}`);
            setCookie("current_session_id", newSessionId, 7);
            await loadSessionList();
        }

        stopThinkingAnimation(thinkingIndicator);
        await updateAllStatusUI();

    } catch (error) {
        // FIX #12: AbortError is a deliberate user stop — don't show an error message.
        // Any other error (network failure, timeout, etc.) is a real problem worth surfacing.
        if (error.name === 'AbortError') {
            console.log("DEBUG: Stream aborted by user.");
            stopThinkingAnimation(thinkingIndicator);
            return;
        }
        console.error("Error during fetch:", error);
        contentSpan.innerHTML = `<span style="color: #ff8a80;">Error: Could not get response from server.</span>`;
        stopThinkingAnimation(thinkingIndicator);
    } finally {
        // (auto-scroll removed — user scrolls manually via the ↓ button)
    }
}

async function uploadImageToBackend(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (e) => {
            const base64Data = e.target.result;
            try {
                const res = await fetch('/save_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                const data = await res.json();
                if (res.ok && data.status === 'success') {
                    resolve(data.filename);
                } else {
                    reject(data.error || 'Upload failed');
                }
            } catch (err) {
                reject(err);
            }
        };
        reader.readAsDataURL(file);
    });
}

async function abortGeneration() {
    _generationId++;
    const abortedGenId = _generationId;
    if (_streamAbortController) {
        _streamAbortController.abort();
        _streamAbortController = null;
    }
    isSending = false;
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.onclick = send;
        sendBtn.title = "Send";
        sendBtn.innerHTML = `<span id="sendBtnIcon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg></span>`;
    }
    try {
        await fetch('/api/abort', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    } catch (e) {
        console.warn('Abort request failed:', e);
    }
    setTimeout(async () => {
        if (_generationId === abortedGenId) await reloadChat();
    }, 400);
}

async function send() {
    if (isSending) return;
    const msgInput = document.getElementById('msg');
    const msg = msgInput.value.trim();
    
    if (!msg && !attachedFiles.length && !_idleFiring) return;
    
    const sendBtn = document.getElementById('sendBtn');

    isSending = true;
    const myGenId = ++_generationId;
    // Briefly disable so there's zero window where the old send onclick is
    // still live. Re-enabled below once the stop-button transform is complete.
    sendBtn.disabled = true;
    sendBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;
    sendBtn.onclick = abortGeneration;
    sendBtn.title = "Stop generating";
    sendBtn.disabled = false; // stop button is now live — safe to re-enable

    let displayMsg = msg;
    
    for (const f of attachedFiles) {
        if (f.type.startsWith('image/')) {
            if (f.serverFilename) {
                displayMsg += `\n\n![Uploaded Image](/get_image/${f.serverFilename})`;
            } else {
                displayMsg += `\n\n![Uploaded Image](${f.content})`;
            }
        } else if (f.type.startsWith('video/')) {
            displayMsg += `\n\n🎬 [Video: ${f.name}]`;
        } else {
            displayMsg += `\n\n[Attached File: ${f.name}]`;
        }
    }

    if (displayMsg.trim()) {
        appendMessage(currentLoadoutDisplayName, displayMsg.trim(), "user");
    }
    
    for (const f of attachedFiles) {
        if (!f.type.startsWith('image/')) {
            appendMessage("System", `Attaching file: ${f.name}`, "system-notification");
        }
    }

    msgInput.value = "";
    msgInput.style.height = "auto";
    
    const chatlog = document.getElementById("chatlog");
    const assistantBubble = document.createElement("div");
    assistantBubble.classList.add("chat-bubble", "grok");
    
    assistantBubble.innerHTML = `${buildBubbleHeader('grok', assistantName)}
                                 <div class="message-body">
                                    <div class="message-content"></div>
                                    <div class="thinking-indicator"></div>
                                 </div>`;
    chatlog.appendChild(assistantBubble);

    // Apply avatar shape immediately — don't wait for streaming to finish
    try {
        const shape = _appearanceCache.assistantAvatarShape || 'circle';
        const avatarEl = assistantBubble.querySelector('.avatar');
        if (avatarEl) {
            AVATAR_SHAPE_CLASSES.forEach(c => avatarEl.classList.remove(c));
            avatarEl.classList.add(`avatar-${shape}`);
        }
    } catch(e) {}

    const thinkingIndicator = assistantBubble.querySelector('.thinking-indicator');
    startThinkingAnimation(thinkingIndicator);

    _streamAbortController = new AbortController();
    const signal = _streamAbortController.signal;
    let fetchPromise;

    const videoFiles   = attachedFiles.filter(f => f.type.startsWith('video/'));
    const imageFiles   = attachedFiles.filter(f => f.type.startsWith('image/'));
    const nonMediaFiles = attachedFiles.filter(f => !f.type.startsWith('image/') && !f.type.startsWith('video/'));

    if (videoFiles.length >= 1) {
        // ── Video path — always wins when a video is present ─────────────────────
        // Mixed attachments are handled gracefully:
        //   + image files         → appended to frames array after extraction
        //   + text/srt/pdf files  → content folded into the prompt as context
        //   + multiple videos     → only first is processed, rest warned and dropped

        if (videoFiles.length > 1) {
            showToast(`⚠ Only the first video will be analyzed — multiple videos not supported.`, 'warn', 3500);
        }

        const videoFile = videoFiles[0];

        // Fold any text file contents into the prompt (perfect for .srt subtitles)
        let contextBlock = '';
        if (nonMediaFiles.length > 0) {
            contextBlock = nonMediaFiles.map(f =>
                `\n\n--- Attached file: ${f.name} ---\n${f.content}`
            ).join('');
            showToast(`📝 Folding ${nonMediaFiles.length} text file(s) into prompt as context.`, 'info', 2800);
        }

        const _vidDuration = await getVideoDuration(videoFile);
        let _vs = {};
        try { _vs = await fetch('/get_video_settings').then(r => r.json()); } catch(_) {}
        const _vDivisor = _vs.frame_divisor ?? 6;
        const _vFloor   = _vs.frame_floor   ?? 4;
        const _vCap     = _vs.frame_cap     ?? 20;
        const _vQuality = _vs.jpeg_quality  ?? 0.82;
        const _vMaxDim  = _vs.resolution    ?? 512;
        const NUM_FRAMES = Math.min(_vCap, Math.max(_vFloor, Math.round(_vidDuration / _vDivisor))); //VIDEO SETTINGS FRAMES
        showToast(`🎬 Extracting ${NUM_FRAMES} frames from '${videoFile.name}'...`, 'info', 3500);
        let frames;
        try {
            frames = await extractVideoFrames(videoFile, NUM_FRAMES, _vQuality, _vMaxDim);
        } catch (err) {
            console.error('Frame extraction failed:', err);
            showToast(`Failed to extract frames: ${err.message || err}`, 'error', 3000);
            throw err;
        }

        // Append any standalone images to the end of the frames array —
        // they're all just base64 image_url parts to the backend anyway.
        // This preserves the image pipeline exactly as-is, no KV side effects.
        if (imageFiles.length > 0) {
            const extraImages = imageFiles.map(f => f.content);
            frames = [...frames, ...extraImages];
            showToast(`🖼 Bundled ${imageFiles.length} image(s) with video frames.`, 'info', 2500);
        }

        const fpsLabel = `${frames.length} frames from '${videoFile.name}'` +
            (imageFiles.length > 0 ? ` + ${imageFiles.length} image(s)` : '');
        showToast(`✓ ${frames.length} total images ready — sending to model...`, 'success', 2500);
        const videoPrompt = (msg || "Describe what happens in this video clip in detail.") + contextBlock;
        fetchPromise = fetch("/analyze_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                frames,
                prompt: videoPrompt,
                fps_label: fpsLabel
            }),
            signal
        });
    } else if (imageFiles.length === 1 && nonMediaFiles.length === 0 && videoFiles.length === 0) {
        // Single image — use existing analyze_image path
        const imgFile = imageFiles[0];
        const imagePayload = imgFile.serverFilename
            ? { prompt: msg, server_filename: imgFile.serverFilename }
            : { prompt: msg, image_data: imgFile.content };
        fetchPromise = fetch("/analyze_image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(imagePayload),
            signal
        });
    } else {
        // Multiple files or mixed — send all via chat_stream as files array
        const payload = { message: msg };
        if (attachedFiles.length > 0) {
            payload.files = attachedFiles.map(f => ({
                name: f.name,
                content: f.content,
                type: f.type,
                serverFilename: f.serverFilename || null
            }));
        }
        fetchPromise = fetch("/chat_stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal
        });
    }

    try {
        await streamAssistantResponse(fetchPromise, assistantBubble, signal);
    } finally {
        removeAttachedFile();
        // Always reset isSending — even if a newer generation has taken over.
        // Previously this was inside the _generationId guard, which meant a fast
        // abort+resend could leave isSending=true forever, permanently locking the button.
        isSending = false;
        // --- IDLE: drain queue or reschedule ---
        _onSendComplete();
        // --- END IDLE ---
        if (_generationId === myGenId) {
            // Only restore the send button UI when we're still the active generation.
            // If a newer one exists it already owns the button — don't stomp it.
            sendBtn.disabled = false;
            sendBtn.onclick = send;
            sendBtn.title = "Send";
            sendBtn.innerHTML = `<span id="sendBtnIcon"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg></span>`;
        }
        _streamAbortController = null;
    }
    if (_generationId === myGenId) {
        await reloadChat();
    }
}

async function loadSessionList() {
    const container = document.getElementById("sessionListContainer");
    try {
        const res = await fetch("/sessions");
        if (!res.ok) throw new Error('Failed to fetch sessions');
        const sessions = await res.json();
        const currentId = getCookie("current_session_id") || (sessions.length > 0 ? sessions[0] : null);

        container.innerHTML = ''; 

        if (!sessions.length) {
            container.innerHTML = '<div style="text-align:center; color: #888; padding: 10px;">No sessions yet.</div>';
            return;
        }

        sessions.forEach(id => {
            const item = document.createElement('div');
            item.className = 'session-list-item';
            if (id === currentId) {
                item.classList.add('active-session');
            }
            item.dataset.sessionId = id;

            const nameSpan = document.createElement('span');
            nameSpan.className = 'session-name';
            nameSpan.textContent = id.replace(/_/g, ' ');
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'session-delete-btn';
            deleteBtn.innerHTML = '&times;';
            deleteBtn.title = 'Delete session';

            deleteBtn.onclick = async (e) => {
                e.stopPropagation();
                // JS-BUG-4 FIX: deleteSession is async — await it so errors surface as toasts
                try { await deleteSession(id); } catch(err) { showToast('Error deleting session.', 'error'); }
            };

            item.onclick = () => {
                switchSession(id);
            };

            item.appendChild(nameSpan);
            item.appendChild(deleteBtn);
            container.appendChild(item);
        });
        
        const activeItem = container.querySelector('.active-session');
        if (!activeItem && sessions.length > 0) {
            await switchSession(sessions[0]);
        }

    } catch (err) {
        console.error("Error loading sessions:", err);
        container.innerHTML = '<div style="text-align:center; color: var(--danger-color);">Error loading sessions.</div>';
    }
}

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

function setCookie(name, value, days) {
    let expires = "";
    if (days) {
        const date = new Date();
        date.setTime(date.getTime() + (days*24*60*60*1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}


async function newSession() {
    showBusyOverlay('Creating new session...', false);
    try {
        const res = await fetch("/new_session", { method: "POST" });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        const newId = data.session_id;
        if (!newId) throw new Error('Server did not return a session_id');
        setCookie("current_session_id", newId, 7);
        document.getElementById("chatlog").innerHTML = "";
        await loadSessionList();
        await reloadChat();
        await updateAllStatusUI();
        if (data.has_first_mes) {
            showToast('New session started with character greeting ✦', 'success');
        } else {
            showToast('New session created', 'success');
        }
    } catch (err) {
        console.error("Error creating new session:", err);
        showToast('Could not create session', 'error');
    } finally {
        hideBusyOverlay();
    }
}

async function switchSession(sessionId) {
    if (!sessionId) {
        console.warn("switchSession called without a session ID.");
        return;
    }
    const idToSwitch = sessionId;
    showBusyOverlay('Switching session...', false);
    try {
        const res = await fetch("/switch_session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: idToSwitch })
        });
        if (res.ok) {
            const data = await res.json();
            setCookie("current_session_id", idToSwitch, 7);
            document.getElementById("chatlog").innerHTML = "";
            await reloadChat();
            await loadSessionList();
            // FIX 8: Backend now backgrounds the index check and tells us via
            // indexing_triggered. Show a brief non-blocking toast so the user
            // knows memory indexing is running without blocking the switch.
            if (data.indexing_triggered) {
                showToast('Session loaded — indexing memory in background', 'info', 2200);
            }
        } else {
            console.error("❌ Session switch failed.");
        }
    } catch (err) {
        console.error("Error switching session:", err);
    } finally {
        hideBusyOverlay();
    }
}

async function reloadChat() {
    const chatlog = document.getElementById("chatlog");
    try {
        const res = await fetch("/history");
        if (!res.ok) throw new Error(`/history returned ${res.status}`);
        const history = await res.json();
        chatlog.innerHTML = "";
        let userIndex = 0;

        // Track which messages we've already merged (skip rendering them individually)
        const skipIndices = new Set();

        for (let i = 0; i < history.length; i++) {
            // Skip if this message was already merged into a previous bubble
            if (skipIndices.has(i)) continue;

            const entry = history[i];
            
            if (entry.role === "user" && !entry.silent) {
                let contentToDisplay = "";
                
                if (typeof entry.content === 'string') {
                    contentToDisplay = entry.content;
                } else if (Array.isArray(entry.content)) {
                    const textPart = entry.content.find(part => part.type === 'text');
                    if (textPart) contentToDisplay += textPart.text;
                    
                    const imagePart = entry.content.find(part => part.type === 'image_url');
                    if (imagePart) {
                        contentToDisplay += `\n\n![Image Content](${imagePart.image_url.url})`;
                    }
                }

                if (!contentToDisplay.trim()) {
                    contentToDisplay = "📎 *[File attached]*";
                }
                appendMessage(currentLoadoutDisplayName, contentToDisplay, "user", userIndex, i, entry.msg_id || null);
                userIndex++;
            } else if (entry.role === "assistant" && !entry.silent) {
                // Check if this assistant message contains a tool call that should be merged
                const hasToolCall = entry.content && (
                    entry.content.includes(searchToolTrigger) || 
                    entry.content.includes(recallToolTrigger) ||
                    entry.content.includes('[TOOL_RECALL:')
                );

                if (hasToolCall && i + 1 < history.length) {
                    // FIX BUG 11: Replace shallow depth-2 merge with a greedy forward scan.
                    // Old code only handled [checkpoint → system → assistant] (depth 1 tool call).
                    // A chained search+recall turn produces:
                    //   [checkpoint(search)] → [system(results)] → [checkpoint(recall)] → [system(results)] → [assistant]
                    // The scan below consumes the entire chain regardless of depth, merging
                    // all pieces into one bubble and marking all consumed indices as skipped.
                    let mergedContent = entry.content;
                    let scanIdx = i + 1;
                    let finalMsgId = entry.msg_id || null; // track last assistant msg_id in chain

                    while (scanIdx < history.length) {
                        const scanEntry = history[scanIdx];

                        if (scanEntry.role === 'system') {
                            // System messages are raw search/recall injections — LLM context only.
                            // Never concatenate into mergedContent. Mark as skipped and move on.
                            skipIndices.add(scanIdx);
                            scanIdx++;
                        } else if (scanEntry.role === 'assistant' && !scanEntry.silent) {
                            const scanHasTool = scanEntry.content && (
                                scanEntry.content.includes(searchToolTrigger) ||
                                scanEntry.content.includes(recallToolTrigger) ||
                                scanEntry.content.includes('[TOOL_RECALL:')
                            );
                            mergedContent += '\n\n' + scanEntry.content;
                            if (scanEntry.msg_id) finalMsgId = scanEntry.msg_id;
                            skipIndices.add(scanIdx);
                            scanIdx++;
                            if (!scanHasTool) break;
                        } else {
                            break;
                        }
                    }

                    appendMessage(assistantName, mergedContent, "grok", null, i, finalMsgId);
                    continue;
                }

                // Normal assistant message (no merge needed)
                appendMessage(assistantName, entry.content, "grok", null, i, entry.msg_id || null);
            }
        }
        await updateAllStatusUI();
        // Re-apply avatar shapes after chat renders
        try {
            _applyAvatarShapeToDOM('assistant', _appearanceCache.assistantAvatarShape || 'circle');
            _applyAvatarShapeToDOM('user', _appearanceCache.userAvatarShape || 'circle');
        } catch(e) {}
    } catch (err) {
        console.error("Error reloading chat:", err);
    }
}

async function deletePair(index, msgId = null) {
    try {
        // FIX Bug 2: Send current session_id explicitly so the backend deletes
        // from the correct session even if async rename is mid-flight
        const sessionId = getCookie("current_session_id");
        const payload = { session_id: sessionId };

        // Prefer msg_id for stable identity — fall back to index for old messages
        if (msgId) payload.msg_id = msgId;
        if (index !== null) payload.index = index;

        const res = await fetch("/delete_message", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        await reloadChat();
        await updateAllStatusUI();
    } catch (err) {
        console.error("❌ Deletion failed:", err);
    }
}

// Tracks which persona is currently being *browsed/edited* in the modal.
// NEVER conflate this with currentActivePersonaName (what the AI is actually using).
let browsingPersonaName = null;

async function openSystemPromptModal() {
    document.getElementById("systemPromptModal").style.display = "flex";

    // Load the active persona first so we know where to start
    try {
        const response = await fetch('/get_system_prompt_content');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        currentActivePersonaName = data.persona_name;
    } catch (error) {
        console.error('Error fetching active persona name:', error);
    }

    await _rebuildPersonaList();
    // Start browsing the currently active persona
    await _browsePersona(currentActivePersonaName);
    // BUG 2 FIX: ensure file input listener is always wired when the modal opens,
    // regardless of whether loadAppearanceAndNameSettings() has been called yet.
    _wirePersonaAvatarUpload();
}

function closeSystemPromptModal() {
    document.getElementById("systemPromptModal").style.display = "none";
    // JS-BUG-3 FIX: clear unsaved persona set on close so reopening the modal
    // doesn't skip backend fetches for names that were never saved this session.
    _unsavedPersonas.clear();
    browsingPersonaName = null;
}

/** Rebuilds the sidebar persona list without touching browsingPersonaName. */
async function _rebuildPersonaList() {
    const container = document.getElementById('personaListScroll');
    if (!container) return;
    container.innerHTML = '<div style="color:#666;font-size:12px;padding:6px;">Loading…</div>';

    try {
        const response = await fetch('/list_personas');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const personas = await response.json();

        container.innerHTML = '';
        personas.forEach(name => {
            const item = document.createElement('div');
            item.className = 'persona-list-item';
            item.dataset.persona = name;
            item.textContent = name;
            if (name === currentActivePersonaName) item.classList.add('active');
            if (name === browsingPersonaName) item.classList.add('browsing');
            item.addEventListener('click', () => _browsePersona(name));
            container.appendChild(item);
        });
    } catch (error) {
        container.innerHTML = '<div style="color:#f66;font-size:12px;padding:6px;">Error loading personas.</div>';
        console.error('Error loading personas:', error);
    }
}

/** Loads a persona into the editor panel WITHOUT changing the active persona. */
// Set of persona names that exist only client-side (not yet saved to backend)
const _unsavedPersonas = new Set();

async function _browsePersona(personaName) {
    if (!personaName) return;
    browsingPersonaName = personaName;

    // Update list item highlights
    document.querySelectorAll('.persona-list-item').forEach(el => {
        el.classList.toggle('browsing', el.dataset.persona === personaName);
    });

    // Update editor label
    const editorLabel = document.getElementById('personaEditorLabel');
    if (editorLabel) editorLabel.textContent = personaName;

    // Update portrait name + active badge
    const portraitName = document.getElementById('personaModalPortraitName');
    const activeBadge = document.getElementById('personaModalActiveBadge');
    if (portraitName) portraitName.textContent = personaName;
    if (activeBadge) activeBadge.style.display = (personaName === currentActivePersonaName) ? 'block' : 'none';

    // Load portrait
    await _loadPersonaModalPortrait(personaName);

    // Load prompt content into textarea
    const systemPromptInput = document.getElementById('systemPromptInput');

    // If this is a brand-new unsaved persona, don't fetch from backend — it would
    // return the active persona's content as a fallback, causing content bleed.
    if (_unsavedPersonas.has(personaName)) {
        systemPromptInput.value = '';
        systemPromptInput.style.height = 'auto';
        systemPromptInput.focus();
        return;
    }

    try {
        const response = await fetch(`/get_system_prompt_content?persona_name=${encodeURIComponent(personaName)}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        systemPromptInput.value = data.prompt_content;
        systemPromptInput.style.height = 'auto';
        systemPromptInput.style.height = Math.min(systemPromptInput.scrollHeight, 400) + 'px';
    } catch (error) {
        console.error('Error loading persona content:', error);
        systemPromptInput.value = 'Error loading persona content.';
    }

    systemPromptInput.focus();
}

/** Fetches and displays the portrait for the given persona in the modal. */
async function _loadPersonaModalPortrait(personaName) {
    const portraitEl = document.getElementById('personaModalPortrait');
    if (!portraitEl) return;
    try {
        const res = await fetch(`/get_persona_avatar_info/${encodeURIComponent(personaName)}`);
        if (!res.ok) throw new Error(`/get_persona_avatar_info returned ${res.status}`);
        const data = await res.json();
        if (data.avatar_url) {
            // JS-BUG-8 FIX: escape personaName so a name containing "> can't break the attribute
            const _safeAlt = personaName.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            portraitEl.innerHTML = `<img src="${data.avatar_url}?t=${Date.now()}" alt="${_safeAlt}">`;
        } else {
            // Fall back to cached emoji or default
            portraitEl.innerHTML = _appearanceCache.assistantAvatar || '🤖';
        }
    } catch(e) {
        portraitEl.innerHTML = '🤖';
    }
}

async function setSystemPrompt() {
    const systemPromptInput = document.getElementById("systemPromptInput");
    const systemPromptText = systemPromptInput.value.trim();

    // Always save to the persona currently open in the editor, never the active one
    const personaName = browsingPersonaName;

    if (!personaName) {
        showToast("No persona selected. Please pick one from the list.", 'error');
        return;
    }

    // Allow saving empty prompt (e.g. for a brand new persona) — confirm via styled modal
    if (!systemPromptText) {
        return new Promise(resolve => {
            openConfirmationModal(
                "The prompt is empty. Save anyway? The persona will have a blank system prompt.",
                async () => { resolve(await _doSaveSystemPrompt(personaName, systemPromptText)); },
                () => resolve(),
                'Save Anyway', '#2a6a2a'
            );
        });
    }

    await _doSaveSystemPrompt(personaName, systemPromptText);
}

async function _doSaveSystemPrompt(personaName, systemPromptText) {
    try {
        const res = await fetch("/set_system_prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt_content: systemPromptText, persona_name: personaName })
        });
        const data = await res.json();
        if (res.ok) {
            console.log("✅ System prompt updated:", data.message);
            _unsavedPersonas.delete(personaName);
            closeSystemPromptModal();
            currentActivePersonaName = personaName;
            await loadAppearanceAndNameSettings();

            if (data.needs_greeting_pick) {
                showToast('✨ Persona activated! Pick a greeting.', 'info', 3000);
                showGreetingPicker(data.greetings, data.persona_name);
                await updateAllStatusUI();
                return;
            }

            showToast('Persona updated', 'success');
            if (data.has_first_mes) {
                showToast('💬 Character\'s opening message added to chat.', 'info', 3500);
            }
            await reloadChat();
            await updateAllStatusUI();
        } else {
            console.error("❌ Failed to update system prompt:", data.error);
            showToast("Failed to update system prompt: " + (data.error || "Unknown error."), 'error');
        }
    } catch (err) {
        console.error("Network error updating system prompt:", err);
        showToast("Network error updating system prompt.", 'error');
    }
}

function openInjectMemoryModal() {
    document.getElementById("injectMemoryModal").style.display = "flex";
    const memoryInput = document.getElementById("memoryInput");
    memoryInput.value = '';
    memoryInput.style.height = 'auto';
    memoryInput.focus();
    document.getElementById('memoryWordCount').textContent = '0 words';
}

function closeInjectMemoryModal() {
    document.getElementById("injectMemoryModal").style.display = "none";
}

async function injectMemory() {
    const memoryInput = document.getElementById("memoryInput");
    const introText = memoryInput.value.trim();
    const summaryLength = document.getElementById("summaryLengthSlider").value;

    if (!introText) {
        showToast("Please enter some text for the memory.", 'error');
        return;
    }

    try {
        const res = await fetch("/inject_silent_intro", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                intro_text: introText,
                max_length: parseInt(summaryLength)
            })
        });
        const data = await res.json();
        if (res.ok) {
            console.log("✅ Memory injected:", data.message);
            closeInjectMemoryModal();
            showToast("Memory injected.", 'success');
        } else {
            console.error("❌ Failed to inject memory:", data.error);
            showToast("Failed to inject memory: " + (data.error || "Unknown error."), 'error');
        }
    } catch (err) {
        console.error("Network error injecting memory:", err);
        showToast("Network error injecting memory.", 'error');
    }
}

async function openManageMemoryModal() {
    document.getElementById('manageMemoryModal').style.display = 'flex';
    const listDiv = document.getElementById('manageMemoryList');
    listDiv.innerHTML = '<p>Loading memories...</p>';
    try {
        const [introRes, searchRes, filesRes] = await Promise.all([
            fetch('/get_silent_intros'),
            fetch('/get_search_results'),
            fetch('/get_attached_files')
        ]);
        const introData = await introRes.json();
        const searchData = await searchRes.json();
        const filesData = await filesRes.json();
        listDiv.innerHTML = '';

        // --- INJECTED MEMORIES ---
        const memHeader = document.createElement('p');
        memHeader.style.cssText = 'color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px 0;';
        memHeader.textContent = '📌 Injected Memories';
        listDiv.appendChild(memHeader);

        if (introData.intros && introData.intros.length > 0) {
            introData.intros.forEach(intro => {
                const memoryItem = document.createElement('div');
                memoryItem.style.cssText = 'background:#333;padding:10px;border-radius:8px;margin-bottom:8px;';
                memoryItem.id = `memory-item-${intro.index}`;
                const memoryText = document.createElement('p');
                memoryText.style.margin = '0 0 10px 0';
                memoryText.textContent = intro.text;
                const btnContainer = document.createElement('div');
                btnContainer.style.cssText = 'display:flex;justify-content:flex-end;gap:10px;';
                const editBtn = document.createElement('button');
                editBtn.textContent = 'Edit';
                editBtn.className = 'action-btn';
                editBtn.onclick = () => openEditMemoryModal(intro.index, intro.text);
                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = 'Delete';
                deleteBtn.className = 'action-btn delete-btn';
                deleteBtn.onclick = () => deleteMemory(intro.index);
                btnContainer.appendChild(editBtn);
                btnContainer.appendChild(deleteBtn);
                memoryItem.appendChild(memoryText);
                memoryItem.appendChild(btnContainer);
                listDiv.appendChild(memoryItem);
            });
        } else {
            const e = document.createElement('p');
            e.style.color = '#888';
            e.textContent = 'No injected memories.';
            listDiv.appendChild(e);
        }

        // --- ATTACHED FILES IN MEMORY ---
        const fileMemHeader = document.createElement('p');
        fileMemHeader.style.cssText = 'color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;';
        fileMemHeader.textContent = '📎 Attached Files in Context';
        listDiv.appendChild(fileMemHeader);

        if (filesData.files && filesData.files.length > 0) {
            filesData.files.forEach(f => {
                const item = document.createElement('div');
                item.style.cssText = 'background:#2a2a1a;padding:10px;border-radius:8px;margin-bottom:8px;border-left:3px solid #c8a020;';
                item.id = `file-mem-item-${f.index}`;

                const nameRow = document.createElement('p');
                nameRow.style.cssText = 'margin:0 0 4px 0;font-size:11px;color:#c8a020;font-weight:bold;';
                nameRow.textContent = `📄 ${f.filename}`;

                const preview = document.createElement('p');
                preview.style.cssText = 'margin:0 0 8px 0;font-size:12px;color:#ccc;white-space:pre-wrap;max-height:80px;overflow:hidden;';
                preview.textContent = f.content.slice(0, 200) + (f.content.length > 200 ? '...' : '');

                const btnRow = document.createElement('div');
                btnRow.style.cssText = 'display:flex;justify-content:flex-end;gap:8px;';

                const editBtn = document.createElement('button');
                editBtn.textContent = 'Edit';
                editBtn.className = 'action-btn';
                editBtn.onclick = () => openEditAttachedFileModal(f.index, f.msg_id, f.filename, f.content);

                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = 'Delete';
                deleteBtn.className = 'action-btn delete-btn';
                deleteBtn.onclick = () => deleteAttachedFile(f.index, f.msg_id);

                btnRow.appendChild(editBtn);
                btnRow.appendChild(deleteBtn);
                item.appendChild(nameRow);
                item.appendChild(preview);
                item.appendChild(btnRow);
                listDiv.appendChild(item);
            });
        } else {
            const e = document.createElement('p');
            e.style.color = '#888';
            e.textContent = 'No attached files in context.';
            listDiv.appendChild(e);
        }

        // --- PERSISTED SEARCH RESULTS ---
        const srHeader = document.createElement('p');
        srHeader.style.cssText = 'color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;';
        srHeader.textContent = '🔍 Persisted Search Results';
        listDiv.appendChild(srHeader);

        if (searchData.results && searchData.results.length > 0) {
            searchData.results.forEach(result => {
                // Use sr_id (stable msg_id) as the DOM key — immune to index drift.
                // result.index is still present from the server but only used as a
                // display-only fallback label; mutations always send sr_id.
                const domKey = result.sr_id || result.index;
                const item = document.createElement('div');
                item.style.cssText = 'background:#2a2a3a;padding:10px;border-radius:8px;margin-bottom:8px;border-left:3px solid #7c6af7;';
                item.id = `search-item-${domKey}`;
                const preview = document.createElement('p');
                preview.style.cssText = 'margin:0 0 8px 0;font-size:12px;color:#ccc;white-space:pre-wrap;max-height:80px;overflow:hidden;';
                preview.id = `search-preview-${domKey}`;
                const badge = result.pinned ? ' 📌 [PINNED]' : '';
                preview.textContent = result.content.slice(0, 200) + (result.content.length > 200 ? '...' : '') + badge;
                // FIX BUG 1 (JS): Wire pin toggle button to the new /toggle_pin_search_result endpoint.
                // Previously the [PINNED] badge was display-only with no way to change it at runtime.
                const pinBtn = document.createElement('button');
                pinBtn.textContent = result.pinned ? '📌 Unpin' : '📌 Pin';
                pinBtn.className = 'action-btn';
                pinBtn.style.cssText = 'margin-right:6px;';
                pinBtn.onclick = () => togglePinSearchResult(domKey, pinBtn, preview, result);
                const deleteBtn = document.createElement('button');
                deleteBtn.textContent = 'Delete';
                deleteBtn.className = 'action-btn delete-btn';
                deleteBtn.onclick = () => deleteSearchResult(domKey);
                item.appendChild(preview);
                item.appendChild(pinBtn);
                item.appendChild(deleteBtn);
                listDiv.appendChild(item);
            });
        } else {
            const e = document.createElement('p');
            e.style.color = '#888';
            e.textContent = 'No persisted search results.';
            listDiv.appendChild(e);
        }

    } catch (error) {
        console.error('Error fetching memories:', error);
        listDiv.innerHTML = '<p style="color:red;">Error: Could not connect to server.</p>';
    }
}

async function deleteSearchResult(srId) {
    openConfirmationModal('Delete this search result from memory?', async () => {
        try {
            const res = await fetch('/delete_search_result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // If domKey fell back to a numeric index (no sr_id on record), send
                // it as { index } so the backend int→int comparison succeeds.
                // String sr_ids use the stable msg_id path (immune to index drift).
                body: JSON.stringify(typeof srId === 'number' ? { index: srId } : { sr_id: srId })
            });
            const data = await res.json();
            if (res.ok) {
                openManageMemoryModal();
            } else {
                showToast('Error: ' + (data.error || 'Could not delete.'), 'error');
            }
        } catch (e) {
            showToast('Network error.', 'error');
        }
    });
}

// FIX BUG 1 (JS): Toggle pin state on a persisted search result at runtime.
// Previously there was no endpoint or UI button to do this — pinned was set-once at save time.
async function togglePinSearchResult(srId, btn, previewEl, result) {
    try {
        const res = await fetch('/toggle_pin_search_result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Same type-dispatch as deleteSearchResult: numeric domKey → index path,
            // string sr_id → stable msg_id path.
            body: JSON.stringify(typeof srId === 'number' ? { index: srId } : { sr_id: srId })
        });
        const data = await res.json();
        if (res.ok) {
            const nowPinned = data.pinned;
            result.pinned = nowPinned;
            btn.textContent = nowPinned ? '📌 Unpin' : '📌 Pin';
            const base = result.content.slice(0, 200) + (result.content.length > 200 ? '...' : '');
            previewEl.textContent = base + (nowPinned ? ' 📌 [PINNED]' : '');
            showToast(nowPinned ? 'Search result pinned.' : 'Search result unpinned.', 'success');
        } else {
            showToast('Error: ' + (data.error || 'Could not toggle pin.'), 'error');
        }
    } catch (e) {
        showToast('Network error.', 'error');
    }
}

// ── Attached File Memory helpers ─────────────────────────────────────────────

async function deleteAttachedFile(index, msgId) {
    openConfirmationModal('Remove this file from context memory?', async () => {
        try {
            const res = await fetch('/delete_attached_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(msgId ? { msg_id: msgId } : { index })
            });
            const data = await res.json();
            if (res.ok) {
                showToast('File removed from context.', 'success');
                openManageMemoryModal();
            } else {
                showToast('Error: ' + (data.error || 'Could not delete.'), 'error');
            }
        } catch (e) {
            showToast('Network error.', 'error');
        }
    });
}

// Reuses the existing editMemoryModal but wired to /edit_attached_file
let _editingFileIndex = null;
let _editingFileMsgId = null;
let _editingFilename  = null;

function openEditAttachedFileModal(index, msgId, filename, content) {
    _editingFileIndex = index;
    _editingFileMsgId = msgId || null;
    _editingFilename  = filename;
    const modal = document.getElementById('editMemoryModal');
    // Update the modal title so the user knows what they're editing
    const h3 = modal.querySelector('h3');
    if (h3) h3.textContent = `✏️ Edit File: ${filename}`;
    const textarea = document.getElementById('editMemoryTextarea');
    textarea.value = content;
    updateMemoryStats();
    document.getElementById('editMemoryModal').style.display = 'flex';
    textarea.focus();
    // Swap the save button action to the file save path
    document.getElementById('editMemoryModal')
        .setAttribute('data-mode', 'file');
}

async function saveEditedAttachedFile() {
    const newContent = document.getElementById('editMemoryTextarea').value.trim();
    if (!newContent) { showToast('Content cannot be empty.', 'error'); return; }
    try {
        const res = await fetch('/edit_attached_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(_editingFileMsgId
                ? { msg_id: _editingFileMsgId, filename: _editingFilename, new_content: newContent }
                : { index: _editingFileIndex,  filename: _editingFilename, new_content: newContent }
            )
        });
        const data = await res.json();
        if (res.ok) {
            closeEditMemoryModal();
            showToast('File updated in memory.', 'success');
            openManageMemoryModal(); // refresh the list
        } else {
            showToast('Error: ' + (data.error || 'Could not save.'), 'error');
        }
    } catch (e) {
        showToast('Network error.', 'error');
    }
}

// ─────────────────────────────────────────────────────────────────────────────

function closeManageMemoryModal() {
    document.getElementById('manageMemoryModal').style.display = 'none';
}

function openEditMemoryModal(index, currentText) {
    editingMemoryIndex = index;
    const textarea = document.getElementById("editMemoryTextarea");
    textarea.value = currentText;
    updateMemoryStats(); 
    document.getElementById("editMemoryModal").style.display = "flex";
}

function closeEditMemoryModal() {
    document.getElementById('editMemoryModal')?.removeAttribute('data-mode');
    document.getElementById("editMemoryModal").style.display = "none";
    editingMemoryIndex = null;
}

function updateMemoryStats() {
    const text = document.getElementById("editMemoryTextarea").value;
    const charCount = text.length;
    const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
    document.getElementById("editMemoryStats").textContent = `${charCount} characters | ${wordCount} words`;
}

async function saveEditedMemory() {
    // If the modal was opened for a file entry, delegate to the file save path
    const modal = document.getElementById('editMemoryModal');
    if (modal && modal.getAttribute('data-mode') === 'file') {
        modal.removeAttribute('data-mode');
        return saveEditedAttachedFile();
    }

    const newText = document.getElementById("editMemoryTextarea").value.trim();
    if (!newText) {
        showToast("Memory content cannot be empty.", 'error');
        return;
    }

    try {
        const response = await fetch('/edit_silent_intro', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: editingMemoryIndex, new_content: newText })
        });
        const data = await response.json();
        if (response.ok) {
            showToast(data.message, 'info');
            closeEditMemoryModal();
            openManageMemoryModal(); 
        } else {
            showToast('Error: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Error editing memory:', error);
        showToast('Network error.', 'error');
    }
}

async function deleteMemory(index) {
    openConfirmationModal('Delete this memory? This action cannot be undone.', async () => {
        try {
            const response = await fetch('/delete_silent_intro', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: index })
            });
            const data = await response.json();
            if (response.ok) {
                showToast(data.message, 'info');
                openManageMemoryModal(); // re-fetch fresh indices from server
            } else {
                showToast('Error: ' + (data.error || 'Could not delete memory.'), 'error');
            }
        } catch (error) {
            console.error('Error deleting memory:', error);
            showToast('Network error. Could not delete memory.', 'error');
        }
    });
}

async function loadVisibleIntros() {
    const listDiv = document.getElementById('manageVisibleIntroList');
    listDiv.innerHTML = '<p style="padding: 10px; text-align: center; color: #888;">Loading intros...</p>';
    try {
        const response = await fetch('/get_visible_intros');
        const data = await response.json();
        if (response.ok) {
            listDiv.innerHTML = '';
            if (data.intros && data.intros.length > 0) {
                data.intros.forEach(intro => {
                    const introItem = document.createElement('div');
                    introItem.style.background = '#333';
                    introItem.style.padding = '10px';
                    introItem.style.borderRadius = '8px';
                    introItem.style.display = 'flex';
                    introItem.style.justifyContent = 'space-between';
                    introItem.style.alignItems = 'center';
                    introItem.id = `visible-intro-item-${intro.index}`;

                    const introText = document.createElement('span');
                    introText.textContent = `"${intro.content.substring(0, 50)}..."`;
                    introText.style.flexGrow = '1';
                    introText.style.marginRight = '10px';
                    introText.title = intro.content;

                    const buttonDiv = document.createElement('div');
                    buttonDiv.style.whiteSpace = 'nowrap';

                    const editBtn = document.createElement('button');
                    editBtn.textContent = 'Edit';
                    editBtn.className = 'action-btn';
                    editBtn.onclick = () => editVisibleIntro(intro.index, intro.content);
                    
                    const deleteBtn = document.createElement('button');
                    deleteBtn.textContent = 'Delete';
                    deleteBtn.className = 'action-btn delete-btn';
                    deleteBtn.onclick = () => deleteVisibleIntro(intro.index);

                    buttonDiv.appendChild(editBtn);
                    buttonDiv.appendChild(deleteBtn);
                    introItem.appendChild(introText);
                    introItem.appendChild(buttonDiv);
                    listDiv.appendChild(introItem);
                });
            } else {
                listDiv.innerHTML = '<p style="padding: 10px; text-align: center; color: #888;">No visible intros are set.</p>';
            }
        } else {
            // FIX XSS: data.error comes from the server and may contain HTML chars.
            // Use _escHtml() before injecting — same pattern as the toast fix (FIX #11).
            listDiv.innerHTML = `<p style="color:red; text-align: center;">Error: ${_escHtml(data.error || 'Could not fetch intros.')}</p>`;
        }
    } catch (error) {
        console.error('Error fetching visible intros:', error);
        listDiv.innerHTML = '<p style="color:red; text-align: center;">Error: Could not connect to server.</p>';
    }
}

async function openVisibleIntroModal() {
    document.getElementById("visibleIntroModal").style.display = "flex";
    const introInput = document.getElementById("visibleIntroInput");
    introInput.value = ''; 
    introInput.focus();
    
    await loadVisibleIntros(); 
}

function closeVisibleIntroModal() {
    document.getElementById("visibleIntroModal").style.display = "none";
}

async function submitVisibleIntro() {
    const introInput = document.getElementById("visibleIntroInput");
    const introText = introInput.value.trim();
    
    if (!introText) {
        introInput.focus();
        introInput.style.transition = "border-color 0.2s";
        introInput.style.borderColor = "var(--danger-color)";
        setTimeout(() => {
            introInput.style.borderColor = "";
        }, 1500);
        return;
    }

    try {
        const res = await fetch("/inject_visible_intro", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: introText })
        });
        
        const data = await res.json();
        
        if (res.ok) {
            console.log("✅ Visible intro injected:", data.message);
            introInput.value = ''; 
            await reloadChat(); 
            await loadVisibleIntros(); 
            appendMessage("System", "Visible intro message added.", "system-notification");
        } else {
            console.error("❌ Failed to inject visible intro:", data.error);
            appendMessage("System", `Failed to inject intro: ${data.error || "Unknown error."}`, "system-notification");
        }
    } catch (err) {
        console.error("Network error injecting visible intro:", err);
        appendMessage("System", "Network error injecting visible intro.", "system-notification");
    }
}

function editVisibleIntro(index, currentContent) {
    editingVisibleIntroIndex = index;
    const textarea = document.getElementById("editVisibleIntroTextarea");
    textarea.value = currentContent;
    document.getElementById("editVisibleIntroModal").style.display = "flex";
    textarea.focus();
}

function closeEditVisibleIntroModal() {
    document.getElementById("editVisibleIntroModal").style.display = "none";
    editingVisibleIntroIndex = null;
}

async function saveEditedVisibleIntro() {
    const textarea = document.getElementById("editVisibleIntroTextarea");
    const newContent = textarea.value.trim();
    
    if (editingVisibleIntroIndex === null) return;

    // Optional: Prevent saving empty strings if you want
    if (!newContent) {
        // FIX: replaced native browser confirm() (which blocks the thread and misbehaves
        // inside open modals on some mobile browsers) with the custom confirmation modal
        // used everywhere else in the codebase.
        openConfirmationModal(
            "The intro is empty. Delete this intro instead?",
            async () => {
                await deleteVisibleIntro(editingVisibleIntroIndex);
                closeEditVisibleIntroModal();
            }
        );
        return;
    }

    try {
        const response = await fetch('/update_visible_intro', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index: editingVisibleIntroIndex, content: newContent })
        });
        const data = await response.json();
        
        if (response.ok) {
            appendMessage("System", "Visible intro updated.", "system-notification");
            closeEditVisibleIntroModal();
            await reloadChat();
            await loadVisibleIntros(); 
        } else {
            showToast('Error: ' + (data.error || 'Could not update intro.'), 'error');
        }
    } catch (error) {
        console.error('Error updating visible intro:', error);
        showToast('Network error. Could not update intro.', 'error');
    }
}

async function renameSession() {
    const oldId = getCookie("current_session_id"); 
    if (!oldId) {
        showToast("No active session selected to rename.", 'error');
        return;
    }

    const newIdPrompt = await openInlinePrompt({
        title: '✏️ Rename Session',
        subtitle: `Current name: "${oldId.replace(/_/g, ' ')}"`,
        hint: 'Spaces are converted to underscores automatically.',
        defaultValue: oldId.replace(/_/g, ' '),
        placeholder: 'Enter new session name...'
    });

    if (!newIdPrompt) return;
    
    const newId = newIdPrompt.trim().replace(/\s+/g, '_');

    if (!newId || newId === oldId) return;

    try {
        const res = await fetch("/rename_session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ old_id: oldId, new_id: newId })
        });
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || "Failed to rename session.");
        }
        const data = await res.json();
        console.log(data.message || "Renamed.");
        setCookie("current_session_id", newId, 7);
        await loadSessionList();
        await updateAllStatusUI();
        showToast('Session renamed', 'success');
    } catch (err) {
        console.error("Error renaming session:", err);
        showToast("Error: " + err.message, 'error');
    }
}

async function deleteSession(sessionId) {
    if (!sessionId) {
        console.error("deleteSession called without a session ID.");
        return;
    }
    return new Promise(resolve => {
        openConfirmationModal(
            `Delete session "${sessionId.replace(/_/g, ' ')}"? This cannot be undone.`,
            async () => {
                try {
                    const res = await fetch("/delete_session", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id: sessionId })
                    });
                    if (!res.ok) throw new Error(`/delete_session returned ${res.status}`);
                    const data = await res.json();
                    console.log(data.message || "Deleted.");
                    await loadSessionList();
                    await reloadChat();
                    await updateAllStatusUI();
                    resolve(true);
                } catch (err) {
                    console.error("Error deleting session:", err);
                    showToast('Error deleting session.', 'error');
                    resolve(false);
                }
            },
            () => resolve(false)  // cancel button — resolve without deleting
        );
    });
}

async function updateActiveCharacterNameUI(selector = '#mainActiveCharacterName') {
    const activeCharacterSpan = document.querySelector(selector);
    if (!activeCharacterSpan) return;
    activeCharacterSpan.textContent = "Loading...";

    try {
        const response = await fetch('/get_active_character_name');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        if (data && data.active_character_name) {
            activeCharacterSpan.textContent = data.active_character_name;
        } else {
            activeCharacterSpan.textContent = "Unknown";
        }
    } catch (error) {
        console.error('Error updating active character name UI:', error);
        activeCharacterSpan.textContent = "Error";
    }
}

// Legacy stub — kept so any external callers don't break, but the new modal
// uses _rebuildPersonaList() + the sidebar list items instead of a <select>.
async function loadPersonasIntoDropdown() {
    await _rebuildPersonaList();
}

// Legacy stub — no longer called by the modal (live-switch via click),
// but kept for safety in case anything else references it.
async function loadSelectedPersona() {
    if (browsingPersonaName) await _browsePersona(browsingPersonaName);
}

async function addNewPersona() {
    const newPersonaNameInput = document.getElementById('newPersonaNameInput');
    const newPersonaName = newPersonaNameInput.value.trim();

    if (!newPersonaName) {
        showToast("Please enter a name for the new persona.", 'error');
        return;
    }

    // Check for duplicates against the current list
    const existingItems = document.querySelectorAll('.persona-list-item');
    const exists = Array.from(existingItems).some(el => el.dataset.persona === newPersonaName);
    if (exists) {
        showToast(`Persona "${newPersonaName}" already exists. Select it from the list.`, 'error');
        return;
    }

    newPersonaNameInput.value = '';

    // Add to sidebar list immediately and browse into it (blank prompt, ready to type)
    const container = document.getElementById('personaListScroll');
    const item = document.createElement('div');
    item.className = 'persona-list-item';
    item.dataset.persona = newPersonaName;
    item.textContent = newPersonaName;
    item.addEventListener('click', () => _browsePersona(newPersonaName));
    container.appendChild(item);

    // Scroll the new item into view
    item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Browse into it (clears textarea, sets portrait label)
    browsingPersonaName = newPersonaName;
    _unsavedPersonas.add(newPersonaName); // mark as not-yet-saved so _browsePersona skips backend fetch
    document.querySelectorAll('.persona-list-item').forEach(el => {
        el.classList.toggle('browsing', el.dataset.persona === newPersonaName);
    });
    const editorLabel = document.getElementById('personaEditorLabel');
    if (editorLabel) editorLabel.textContent = newPersonaName;
    const portraitName = document.getElementById('personaModalPortraitName');
    if (portraitName) portraitName.textContent = newPersonaName;
    const activeBadge = document.getElementById('personaModalActiveBadge');
    if (activeBadge) activeBadge.style.display = 'none';
    const portraitEl = document.getElementById('personaModalPortrait');
    if (portraitEl) portraitEl.innerHTML = '🤖';

    const systemPromptInput = document.getElementById('systemPromptInput');
    systemPromptInput.value = '';
    systemPromptInput.style.height = 'auto';
    systemPromptInput.focus();
}

async function deleteSelectedPersona() {
    const personaToDelete = browsingPersonaName;

    if (!personaToDelete) {
        showToast("No persona selected. Click one in the list first.", 'error');
        return;
    }

    if (personaToDelete === currentActivePersonaName) {
        showToast("Cannot delete the currently active persona. Switch to another first.", 'error');
        return;
    }

    openConfirmationModal(
        `Delete persona "${personaToDelete}"? This cannot be undone.`,
        async () => {
            try {
                const response = await fetch('/delete_persona', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ persona_name: personaToDelete })
                });
                const result = await response.json();
                if (response.ok) {
                    console.log(result.message);
                    browsingPersonaName = null;
                    await _rebuildPersonaList();
                    await _browsePersona(currentActivePersonaName);
                    await updateActiveCharacterNameUI();
                } else {
                    showToast('Failed to delete persona: ' + (result.error || result.message), 'error');
                }
            } catch (error) {
                console.error('Network error deleting persona:', error);
                showToast('Network error deleting persona.', 'error');
            }
        }
    );
}

async function renameSelectedPersona() {
    if (!browsingPersonaName) {
        showToast("No persona selected. Click one in the list first.", 'error');
        return;
    }

    const trimmed = await openInlinePrompt({
        title: '✏️ Rename Persona',
        subtitle: `Current name: "${browsingPersonaName}"`,
        hint: '',
        defaultValue: browsingPersonaName,
        placeholder: 'Enter new persona name...'
    });

    if (!trimmed || trimmed === browsingPersonaName) return;

    try {
        const res = await fetch('/rename_persona', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ old_name: browsingPersonaName, new_name: trimmed })
        });
        const data = await res.json();
        if (!res.ok) { showToast('Rename failed: ' + (data.error || 'Unknown error.'), 'error'); return; }

        if (browsingPersonaName === currentActivePersonaName) currentActivePersonaName = trimmed;
        browsingPersonaName = trimmed;
        await _rebuildPersonaList();
        await _browsePersona(trimmed);
        console.log(`✅ Persona renamed → "${trimmed}"`);
    } catch (err) {
        console.error('Network error renaming persona:', err);
        showToast('Network error renaming persona.', 'error');
    }
}


async function openTemperaturesModal() {
    document.getElementById("temperaturesModal").style.display = "flex";
    const orMode = _isOpenRouter();
    try {
        if (orMode) {
            // OpenRouter mode — load isolated temperature from app settings
            const res = await fetch('/get_app_settings');
            const settings = await res.json();
            const temp = settings.openrouter_temperature ?? 0.7;
            document.getElementById('chatStreamTemp').value = temp;
            document.getElementById('chatStreamTempValue').textContent = parseFloat(temp).toFixed(2);
        } else {
            const response = await fetch('/get_temperatures');
            if (!response.ok) throw new Error('Failed to fetch temperatures');
            const temps = await response.json();
            document.getElementById('chatStreamTemp').value = temps.chat_stream;
            document.getElementById('chatStreamTempValue').textContent = temps.chat_stream.toFixed(2);
            document.getElementById('summarizationTemp').value = temps.summarization;
            document.getElementById('summarizationTempValue').textContent = temps.summarization.toFixed(2);
        }
    } catch (error) {
        console.error("Error loading temperatures:", error);
        showToast("Could not load temperature settings from the server.", 'error');
    }
    // Summarization temp still usable — OpenRouter routes sampler params intelligently
    _setKoboldOnly('koboldOnlyTemps', false);
}

function closeTemperaturesModal() {
    document.getElementById("temperaturesModal").style.display = "none";
}

async function saveTemperatures() {
    const orMode = _isOpenRouter();
    try {
        if (orMode) {
            // OpenRouter mode — save isolated temperature to app settings
            const res = await fetch('/set_app_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ openrouter_temperature: parseFloat(document.getElementById('chatStreamTemp').value) })
            });
            const result = await res.json();
            if (res.ok) { closeTemperaturesModal(); showToast('OpenRouter temperature saved', 'success'); }
            else showToast("Error: " + (result.error || "Unknown"), 'error');
        } else {
            const newTemps = {
                chat_stream: parseFloat(document.getElementById('chatStreamTemp').value),
                summarization: parseFloat(document.getElementById('summarizationTemp').value),
            };
            const response = await fetch('/set_temperatures', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newTemps)
            });
            const result = await response.json();
            if (response.ok) { closeTemperaturesModal(); showToast('Temperatures saved', 'success'); }
            else showToast("Error saving temperatures: " + (result.error || "Unknown error"), 'error');
        }
    } catch (error) {
        console.error("Network error saving temperatures:", error);
        showToast("Network error saving temperatures.", 'error');
    }
}

async function openSamplersModal() {
    document.getElementById("samplersModal").style.display = "flex";
    const orMode = _isOpenRouter();
    try {
        if (orMode) {
            // OpenRouter mode — load isolated top_p from app settings
            const res = await fetch('/get_app_settings');
            if (!res.ok) throw new Error(`/get_app_settings returned ${res.status}`);
            const settings = await res.json();
            const topP = settings.openrouter_top_p ?? 0.9;
            document.getElementById('topP').value = topP;
            document.getElementById('topPValue').textContent = parseFloat(topP).toFixed(2);
        } else {
            const res = await fetch('/get_sampler_settings');
            if (!res.ok) throw new Error(`/get_sampler_settings returned ${res.status}`);
            const settings = await res.json();
            document.getElementById('topP').value = settings.top_p;
            document.getElementById('topPValue').textContent = parseFloat(settings.top_p).toFixed(2);
            document.getElementById('topK').value = settings.top_k;
            document.getElementById('topKValue').textContent = settings.top_k;
            document.getElementById('minP').value = settings.min_p;
            document.getElementById('minPValue').textContent = parseFloat(settings.min_p).toFixed(2);
            document.getElementById('repetitionPenalty').value = settings.repetition_penalty;
            document.getElementById('repetitionPenaltyValue').textContent = parseFloat(settings.repetition_penalty).toFixed(2);
            document.getElementById('encapsulateThinkingToggle').checked = !!settings.encapsulate_thinking;
            // DRY penalty sliders
            document.getElementById('dryMultiplier').value = settings.dry_multiplier ?? 0.0;
            document.getElementById('dryMultiplierValue').textContent = parseFloat(settings.dry_multiplier ?? 0.0).toFixed(2);
            document.getElementById('dryBase').value = settings.dry_base ?? 1.75;
            document.getElementById('dryBaseValue').textContent = parseFloat(settings.dry_base ?? 1.75).toFixed(2);
            document.getElementById('dryAllowedLength').value = settings.dry_allowed_length ?? 2;
            document.getElementById('dryAllowedLengthValue').textContent = settings.dry_allowed_length ?? 2;
        }
    } catch(error) {
        console.error("Error loading sampler settings:", error);
        showToast("Could not load sampler settings from the server.", 'error');
    }
    // All samplers kept active — OpenRouter supports min_p and repetition_penalty
    _setKoboldOnly('koboldOnlySamplers', false);
}

function closeSamplersModal() {
    document.getElementById("samplersModal").style.display = "none";
}

async function saveSamplerSettings() {
    const orMode = _isOpenRouter();
    if (orMode) {
        // OpenRouter mode — save top_p to isolated lane; top_k/min_p/rep_penalty
        // go to their vanilla endpoints (OpenRouter routes them intelligently)
        try {
            const res = await fetch('/set_app_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ openrouter_top_p: parseFloat(document.getElementById('topP').value) })
            });
            const result = await res.json();
            if (!res.ok) { showToast("Error: " + (result.error || "Unknown"), 'error'); return; }
        } catch(e) { showToast('Network error saving sampler settings.', 'error'); return; }
        // Also save kobold sampler values normally so they persist
        const res2 = await fetch('/set_sampler_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                top_p: document.getElementById('topP').value,
                top_k: document.getElementById('topK').value,
                min_p: document.getElementById('minP').value,
                repetition_penalty: document.getElementById('repetitionPenalty').value,
                encapsulate_thinking: document.getElementById('encapsulateThinkingToggle').checked,
                dry_multiplier: document.getElementById('dryMultiplier').value,
                dry_base: document.getElementById('dryBase').value,
                dry_allowed_length: document.getElementById('dryAllowedLength').value,
            })
        });
        const result2 = await res2.json();
        if (!res2.ok) { showToast('Error saving sampler settings: ' + (result2.error || 'Unknown'), 'error'); return; }
        closeSamplersModal();
        showToast('Sampler settings saved', 'success');
        return;
    }
    const newSettings = {
        top_p: document.getElementById('topP').value,
        top_k: document.getElementById('topK').value,
        min_p: document.getElementById('minP').value,
        repetition_penalty: document.getElementById('repetitionPenalty').value,
        encapsulate_thinking: document.getElementById('encapsulateThinkingToggle').checked,
        dry_multiplier: document.getElementById('dryMultiplier').value,
        dry_base: document.getElementById('dryBase').value,
        dry_allowed_length: document.getElementById('dryAllowedLength').value,
    };

    try {
        const response = await fetch('/set_sampler_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newSettings)
        });
        const result = await response.json();
        if (response.ok) {
            console.log("Sampler settings saved:", result.message);
            closeSamplersModal();
            showToast('Sampler settings saved', 'success');
        } else {
            showToast("Error saving sampler settings: " + (result.error || "Unknown error"), 'error');
        }
    } catch (error) {
        console.error("Network error saving sampler settings:", error);
        showToast("Network error saving sampler settings.", 'error');
    }
}


async function openTokensModal() {
    document.getElementById("tokensModal").style.display = "flex";
    try {
        const response = await fetch('/get_tokens');
        if (!response.ok) throw new Error('Failed to fetch token settings');
        const tokens = await response.json();
        
        document.getElementById('maxChatMessages').value = tokens.max_chat_messages;
        document.getElementById('maxChatMessagesNum').value = tokens.max_chat_messages;
        
        if (_isOpenRouter()) {
            // OpenRouter mode — load isolated max tokens from app settings
            const appRes = await fetch('/get_app_settings');
            if (!appRes.ok) throw new Error(`/get_app_settings returned ${appRes.status}`);
            const appSettings = await appRes.json();
            const orTokens = appSettings.openrouter_max_tokens ?? 1024;
            // Expand slider max if saved value exceeds default ceiling
            const maxTokensSlider = document.getElementById('llmMaxTokens');
            const maxTokensNum    = document.getElementById('llmMaxTokensNum');
            if (maxTokensSlider && orTokens > parseInt(maxTokensSlider.max)) {
                maxTokensSlider.max = orTokens;
                if (maxTokensNum) maxTokensNum.max = orTokens;  // keep num input ceiling in sync
            }
            document.getElementById('llmMaxTokens').value = orTokens;
            document.getElementById('llmMaxTokensNum').value = orTokens;
        } else {
            // Reset slider + num input ceiling back to the Kobold-mode default in case
            // it was expanded by a previous OpenRouter session (OR allows values > 32768).
            const maxTokensSlider = document.getElementById('llmMaxTokens');
            const maxTokensNum    = document.getElementById('llmMaxTokensNum');
            if (maxTokensSlider) maxTokensSlider.max = 32768;
            if (maxTokensNum)    maxTokensNum.max    = 32768;
            document.getElementById('llmMaxTokens').value = tokens.llm_max_tokens;
            document.getElementById('llmMaxTokensNum').value = tokens.llm_max_tokens;
        }

        document.getElementById('resonanceToggle').checked = tokens.resonance_enabled;
        document.getElementById('ghostMemoryToggle').checked = tokens.ghost_memory_enabled !== false;

    } catch (error) {
        console.error("Error loading token settings:", error);
        showToast("Could not load token settings from the server.", 'error');
    }
}

function closeTokensModal() {
    document.getElementById("tokensModal").style.display = "none";
}

async function saveTokenSettings() {
    // Always save resonance + ghost toggles + max_chat_messages — these apply in all backend modes.
    // FIX: max_chat_messages was previously missing from sharedTokens, causing it to be silently
    // dropped when saving in OpenRouter mode (the OR branch did an early return before the full
    // newTokens payload was ever sent). Now included here so it's always persisted.
    const sharedTokens = {
        resonance_enabled: document.getElementById('resonanceToggle').checked,
        ghost_memory_enabled: document.getElementById('ghostMemoryToggle').checked,
        max_chat_messages: document.getElementById('maxChatMessages').value
    };
    // FIX BUG 1: Await + check the shared save — previously fire-and-forget meant
    // resonance_enabled / ghost_memory_enabled / max_chat_messages could silently fail
    // while a success toast was still shown by the second save below.
    try {
        const sharedRes = await fetch('/set_tokens', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sharedTokens)
        });
        if (!sharedRes.ok) {
            const sharedErr = await sharedRes.json().catch(() => ({}));
            showToast('Error saving shared settings: ' + (sharedErr.error || 'Unknown'), 'error');
            return;
        }
    } catch(e) {
        console.error('Could not save shared token settings:', e);
        showToast('Network error saving token settings.', 'error');
        return;
    }

    if (_isOpenRouter()) {
        // OpenRouter mode — save isolated max tokens to app settings
        try {
            const res = await fetch('/set_app_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ openrouter_max_tokens: parseInt(document.getElementById('llmMaxTokens').value) })
            });
            const result = await res.json();
            if (res.ok) { closeTokensModal(); showToast('Token settings saved', 'success'); }
            else showToast("Error: " + (result.error || "Unknown"), 'error');
        } catch(e) { showToast('Network error saving token settings.', 'error'); }
        return;
    }
    // sharedTokens already sent max_chat_messages, resonance_enabled, ghost_memory_enabled above.
    // Only send llm_max_tokens here to avoid a redundant double-write.
    const newTokens = {
        llm_max_tokens: document.getElementById('llmMaxTokens').value,
    };

    try {
        const response = await fetch('/set_tokens', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newTokens)
        });
        const result = await response.json();
        if (response.ok) {
            console.log("Token settings saved:", result.message);
            closeTokensModal();
            showToast('Token settings saved', 'success');
        } else {
            showToast("Error saving token settings: " + (result.error || "Unknown error"), 'error');
        }
    } catch (error) {
        console.error("Network error saving token settings:", error);
        showToast("Network error saving token settings.", 'error');
    }
}

function closeAppSettingsModal() {
    document.getElementById("appSettingsModal").style.display = "none";
    // Re-mask the API key whenever the modal is closed
    const keyField = document.getElementById('llmApiKey');
    if (keyField) keyField.type = 'password';
}

function toggleApiKeyVisibility() {
    const field = document.getElementById('llmApiKey');
    const btn = document.getElementById('llmApiKeyRevealBtn');
    if (!field) return;
    if (field.type === 'password') {
        field.type = 'text';
        btn.textContent = '🙈';
        btn.title = 'Hide API key';
    } else {
        field.type = 'password';
        btn.textContent = '👁';
        btn.title = 'Show API key';
    }
}

async function openAppSettingsModal() {
    document.getElementById("appSettingsModal").style.display = "flex";
    try {
        const response = await fetch('/get_app_settings');
        if (!response.ok) throw new Error('Failed to fetch app settings');
        const settings = await response.json();
        
        // --- Backend mode: populate module-level caches from saved settings ---
        _koboldCache.endpoint      = settings.kobold_endpoint      || 'http://127.0.0.1:5001/v1/chat/completions';
        _koboldCache.model         = settings.kobold_model         || '';
        _openRouterCache.endpoint  = settings.openrouter_endpoint  || 'https://openrouter.ai/api/v1/chat/completions';
        _openRouterCache.model     = settings.openrouter_model     || '';
        _openRouterCache.apiKey    = settings.openrouter_api_key   || '';

        const mode = settings.backend_mode || 'kobold';
        _applyBackendMode(mode);  // apply live UI state

        document.getElementById('reasoningEnabledToggle').checked = settings.reasoning_enabled || false;
        const visionEl = document.getElementById('visionEnabledToggle');
        visionEl.checked = settings.vision_enabled || false;
        visionEl.onchange = () => _applyVisionUI(_isOpenRouter(), visionEl.checked);
        _applyVisionUI(_isOpenRouter(), visionEl.checked);
        document.getElementById('enableScrapingToggle').checked    = settings.enable_scraping;
        
        document.getElementById('scrapingWordLimit').value = settings.scraping_word_limit;
        document.getElementById('scrapingWordLimitValue').textContent = settings.scraping_word_limit;
        
        document.getElementById('searchMaxResults').value = settings.search_max_results;
        document.getElementById('searchMaxResultsValue').textContent = settings.search_max_results;

        // ── Paste-to-File (from backend JSON) ──
        const _ptf = parseInt(settings.paste_to_file_length ?? 0);
        pasteToFileLength = isNaN(_ptf) ? 0 : _ptf;
        const _ptfEl = document.getElementById('pasteToFileLengthInput');
        if (_ptfEl) _ptfEl.value = pasteToFileLength;

    } catch (error) {
        console.error("Error loading app settings:", error);
        showToast("Could not load application settings from the server.", 'error');
        // FIX 6: pasteToFileLength is set inside the try block above.  If the full
        // settings fetch/parse fails for any reason it would silently stay 0,
        // disabling paste-to-file for the entire session.  Do a lightweight targeted
        // retry here so the feature still works even when the modal load fails.
        try {
            const _fbr = await fetch('/get_app_settings');
            if (_fbr.ok) {
                const _fbd = await _fbr.json();
                const _fbv = parseInt(_fbd.paste_to_file_length ?? 0);
                pasteToFileLength = isNaN(_fbv) ? 0 : _fbv;
            }
        } catch (_fbErr) {
            console.warn('pasteToFileLength fallback fetch also failed — stays 0:', _fbErr);
        }
    }
    await loadApiProfiles();

    // Attach radio listeners once — guard prevents stacking on modal re-open
    if (!_backendModeListenerAdded) {
        document.querySelectorAll('input[name="backendMode"]').forEach(radio => {
            radio.addEventListener('change', () => {
                const isOpenRouter = document.getElementById('backendModeOpenRouter').checked;
                // Snapshot current fields into the cache of the mode we're LEAVING
                if (isOpenRouter) {
                    // leaving kobold → snapshot kobold endpoint + model
                    _koboldCache.endpoint = document.getElementById('llmApiEndpoint').value;
                    _koboldCache.model    = document.getElementById('llmModelName').value;
                } else {
                    // leaving openrouter → snapshot OR fields
                    _openRouterCache.endpoint = document.getElementById('llmApiEndpoint').value;
                    _openRouterCache.model    = document.getElementById('llmModelName').value;
                    _openRouterCache.apiKey   = document.getElementById('llmApiKey').value;
                }
                _applyBackendMode(isOpenRouter ? 'openrouter' : 'kobold');
                loadApiProfiles();  // refresh profile list for the new mode
            });
        });
        _backendModeListenerAdded = true;
    }
}

async function saveAppSettings() {
    const mode = document.querySelector('input[name="backendMode"]:checked')?.value || 'kobold';
    const endpoint = document.getElementById('llmApiEndpoint').value.trim();
    const model    = document.getElementById('llmModelName').value.trim();
    const apiKey   = document.getElementById('llmApiKey').value.trim();

    // Build per-mode isolated config — each mode only saves its own fields.
    // The active llm_ keys are always synced from whichever mode is selected.
    const newSettings = {
        backend_mode:       mode,
        reasoning_enabled:  document.getElementById('reasoningEnabledToggle').checked,
        vision_enabled:     document.getElementById('visionEnabledToggle').checked,
        enable_scraping:    document.getElementById('enableScrapingToggle').checked,
        scraping_word_limit: parseInt(document.getElementById('scrapingWordLimit').value),
        search_max_results:  parseInt(document.getElementById('searchMaxResults').value),
        paste_to_file_length: parseInt(document.getElementById('pasteToFileLengthInput')?.value || '0'),
        // Per-mode isolated storage
        ...(mode === 'kobold'
            ? { kobold_endpoint: endpoint, kobold_model: model,
                llm_api_endpoint: endpoint, llm_model_name: model, llm_api_key: '' }
            : { openrouter_endpoint: endpoint, openrouter_model: model, openrouter_api_key: apiKey,
                llm_api_endpoint: endpoint, llm_model_name: model, llm_api_key: apiKey }
        ),
    };

    try {
        const response = await fetch('/set_app_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newSettings)
        });
        const result = await response.json();
        if (response.ok) {
            console.log("App settings saved:", result.message);

            // --- IMMEDIATE APPLY: sync caches so next modal open / send uses fresh values ---
            if (mode === 'kobold') {
                _koboldCache.endpoint = newSettings.kobold_endpoint || endpoint;
                _koboldCache.model    = newSettings.kobold_model    || model;
            } else {
                _openRouterCache.endpoint = newSettings.openrouter_endpoint || endpoint;
                _openRouterCache.model    = newSettings.openrouter_model    || model;
                _openRouterCache.apiKey   = newSettings.openrouter_api_key  || apiKey;
            }
            // Re-apply vision/file-button UI immediately with the freshly saved value
            const visionNow = document.getElementById('visionEnabledToggle')?.checked || false;
            _applyVisionUI(mode === 'openrouter', visionNow);

            // ── Apply paste-to-file threshold immediately ──
            const _ptfSaved = parseInt(document.getElementById('pasteToFileLengthInput')?.value || '0');
            pasteToFileLength = isNaN(_ptfSaved) ? 0 : _ptfSaved;

            closeAppSettingsModal();
            showToast("Application settings saved and applied.", 'success');
        } else {
            showToast("Error saving app settings: " + (result.error || "Unknown error"), 'error');
        }
    } catch (error) {
        console.error("Network error saving app settings:", error);
        showToast("Network error saving application settings.", 'error');
    }
}

// ============================================================================
// --- API PROFILE MANAGEMENT ---
// ============================================================================

// Applies a backend mode to the UI live — swaps fields, shows/hides sections.
// Shows/hides the file attach button based on backend mode + vision toggle.
// Kobold/local: always visible (local backends handle vision themselves).
// OpenRouter: only visible when vision_enabled is explicitly ON.
function _applyVisionUI(isOpenRouter, visionEnabled) {
    const fileBtn = document.getElementById('fileBtn');
    if (!fileBtn) return;
    // Logic:
    //   KoboldCPP mode (isOpenRouter=false) → always show the button.
    //     Text files are always attachable via /chat_stream payload;
    //     KoboldCPP multimodal handles images natively.
    //   OpenRouter + vision ON  → show (the API supports multimodal payloads).
    //   OpenRouter + vision OFF → hide (no multimodal API configured;
    //     neither images nor text-file injection would be processed correctly).
    fileBtn.style.display = (isOpenRouter && !visionEnabled) ? 'none' : '';
}

function _applyBackendMode(mode) {
    const isOpenRouter = mode === 'openrouter';
    document.getElementById(isOpenRouter ? 'backendModeOpenRouter' : 'backendModeKobold').checked = true;
    if (isOpenRouter) {
        document.getElementById('llmApiEndpoint').value = _openRouterCache.endpoint;
        document.getElementById('llmModelName').value   = _openRouterCache.model;
        document.getElementById('llmApiKey').value      = _openRouterCache.apiKey;
        document.getElementById('llmModelName').placeholder = 'e.g., google/gemini-flash-1.5';
        document.getElementById('koboldModelHint').style.display = 'none';
        const badge = document.getElementById('koboldOnlyBadge');
        if (badge) badge.style.display = 'inline';
    } else {
        document.getElementById('llmApiEndpoint').value = _koboldCache.endpoint;
        document.getElementById('llmModelName').value   = _koboldCache.model;
        document.getElementById('llmApiKey').value      = '';
        document.getElementById('llmModelName').placeholder = 'e.g., llama3.2 (Ollama) — leave blank for KoboldCPP/LM Studio';
        document.getElementById('koboldModelHint').style.display = '';
        const badge = document.getElementById('koboldOnlyBadge');
        if (badge) badge.style.display = 'none';
    }
    document.getElementById('openRouterFields').style.display = isOpenRouter ? 'block' : 'none';
    document.getElementById('openRouterOptions').style.display = isOpenRouter ? 'block' : 'none';
    // Sync the file attach button visibility with the current vision toggle state
    const visionOn = document.getElementById('visionEnabledToggle')?.checked || false;
    _applyVisionUI(isOpenRouter, visionOn);
}

async function loadApiProfiles() {
    try {
        const res = await fetch('/get_api_profiles');
        if (!res.ok) throw new Error(`/get_api_profiles returned ${res.status}`);
        const data = await res.json();
        const select = document.getElementById('apiProfileSelect');
        if (!select) return;
        select.innerHTML = '';

        // Only show profiles that belong to the currently active backend mode
        const activeMode = document.querySelector('input[name="backendMode"]:checked')?.value || 'kobold';
        const filtered = Object.entries(data.profiles).filter(([, profile]) => {
            return (profile.mode || 'kobold') === activeMode;
        });

        if (filtered.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = `No ${activeMode === 'openrouter' ? 'OpenRouter' : 'Kobold'} profiles yet`;
            opt.disabled = true;
            select.appendChild(opt);
        } else {
            filtered.forEach(([name]) => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                if (name === data.active) opt.selected = true;
                select.appendChild(opt);
            });
        }
        // Only show badge if active profile belongs to current mode
        const activeProfile = data.profiles[data.active];
        const activeBelongsHere = activeProfile && (activeProfile.mode || 'kobold') === activeMode;
        _updateApiProfileActiveBadge(activeBelongsHere ? data.active : null);
    } catch(e) { console.error('Error loading API profiles:', e); }
}

function _updateApiProfileActiveBadge(activeName) {
    const badge = document.getElementById('apiProfileActiveBadge');
    if (badge) badge.textContent = activeName ? `Active: ${activeName}` : '';
}

async function switchApiProfile() {
    const select = document.getElementById('apiProfileSelect');
    const name = select?.value;
    if (!name) return;
    try {
        const res = await fetch('/switch_api_profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (res.ok) {
            // Apply profile — also flip backend mode to match what was saved with the profile
            const profileMode = data.profile.mode || 'kobold';
            if (profileMode === 'openrouter') {
                _openRouterCache.endpoint = data.profile.endpoint || '';
                _openRouterCache.model    = data.profile.model    || '';
                _openRouterCache.apiKey   = data.profile.api_key  || '';
            } else {
                _koboldCache.endpoint = data.profile.endpoint || '';
                _koboldCache.model    = data.profile.model    || '';
            }
            _applyBackendMode(profileMode);
            _updateApiProfileActiveBadge(name);
            showToast(`Switched to "${name}"`, 'success');
        } else {
            showToast('Error: ' + (data.error || 'Unknown'), 'error');
        }
    } catch(e) { showToast('Network error switching profile.', 'error'); }
}

async function saveApiProfile() {
    const name = await openInlinePrompt({
        title: '💾 Save API Profile',
        subtitle: 'Name this profile (e.g. "KoboldCPP Local", "OpenRouter GPT-4o")',
        hint: 'Saving over an existing name overwrites it. Creates a new one if the name is new.',
        defaultValue: document.getElementById('apiProfileSelect')?.value || '',
        placeholder: 'Profile name...'
    });
    if (!name || !name.trim()) return;
    const trimmed = name.trim();
    const currentMode = document.querySelector('input[name="backendMode"]:checked')?.value || 'kobold';
    const payload = {
        name:     trimmed,
        mode:     currentMode,
        endpoint: document.getElementById('llmApiEndpoint').value.trim(),
        model:    document.getElementById('llmModelName').value.trim(),
        api_key:  currentMode === 'openrouter' ? document.getElementById('llmApiKey').value.trim()   : '',
    };
    try {
        const res = await fetch('/save_api_profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
            await loadApiProfiles();
            // Select and badge both reflect the saved profile
            const select = document.getElementById('apiProfileSelect');
            if (select) select.value = trimmed;
            _updateApiProfileActiveBadge(data.active || trimmed);
            showToast(`Profile "${trimmed}" saved ✓`, 'success');
        } else {
            showToast('Error: ' + (data.error || 'Unknown'), 'error');
        }
    } catch(e) { showToast('Network error saving profile.', 'error'); }
}

async function deleteApiProfile() {
    const select = document.getElementById('apiProfileSelect');
    const name = select?.value;
    if (!name) return;
    openConfirmationModal(`Delete API profile "${name}"? This cannot be undone.`, async () => {
        try {
            const res = await fetch('/delete_api_profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (res.ok) {
                await loadApiProfiles();
                showToast(`Profile "${name}" deleted.`, 'success');
            } else {
                showToast('Error: ' + (data.error || 'Unknown'), 'error');
            }
        } catch(e) { showToast('Network error deleting profile.', 'error'); }
    });
}

// ============================================================================
// --- END API PROFILE MANAGEMENT ---
// ============================================================================


async function openStreamingSettingsModal() {
    await loadStreamingSettings();
    document.getElementById("streamingSettingsModal").style.display = "flex";
    document.getElementById('charDelay').value = streamingSettings.charDelay;
    document.getElementById('charDelayValue').textContent = streamingSettings.charDelay;
    document.getElementById('punctuationDelay').value = streamingSettings.punctuationDelay;
    document.getElementById('punctuationDelayValue').textContent = streamingSettings.punctuationDelay;
    document.getElementById('commaDelay').value = streamingSettings.commaDelay;
    document.getElementById('commaDelayValue').textContent = streamingSettings.commaDelay;
    // Reflect current delay mode
    _applyStreamingDelayUI(streamingSettings.streamingDelayEnabled);
}

function closeStreamingSettingsModal() {
    document.getElementById("streamingSettingsModal").style.display = "none";
}

async function saveStreamingSettings() {
    streamingSettings.charDelay = parseInt(document.getElementById('charDelay').value);
    streamingSettings.punctuationDelay = parseInt(document.getElementById('punctuationDelay').value);
    streamingSettings.commaDelay = parseInt(document.getElementById('commaDelay').value);
    // streamingSettings.streamingDelayEnabled is already updated live by setStreamingDelayMode()

    try {
        await fetch('/set_streaming_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(streamingSettings)
        });
        console.log("Streaming settings saved:", streamingSettings);
    } catch(e) { console.warn('Could not save streaming settings:', e); }
    closeStreamingSettingsModal();
    showToast('Streaming settings saved.', 'success');
}

async function loadStreamingSettings() {
    try {
        const res = await fetch('/get_streaming_settings');
        if (res.ok) {
            const data = await res.json();
            streamingSettings = { ...streamingSettings, ...data };
            // Normalise boolean (JSON may send it as bool or 0/1)
            if (typeof streamingSettings.streamingDelayEnabled !== 'boolean') {
                streamingSettings.streamingDelayEnabled = !!streamingSettings.streamingDelayEnabled;
            }
            console.log("Loaded streaming settings:", streamingSettings);
        }
    } catch(e) { console.warn('Could not load streaming settings:', e); }
}

// ── Streaming delay mode toggle helpers ──────────────────────────────────────

/**
 * Called by the ON / OFF buttons in the streaming modal.
 * Updates in-memory state and refreshes button visuals immediately.
 * The value is persisted to the server when the user hits "Save Settings".
 */
function setStreamingDelayMode(enabled) {
    streamingSettings.streamingDelayEnabled = enabled;
    _applyStreamingDelayUI(enabled);
}

/**
 * Syncs button active-states, slider opacity/interactivity, and the
 * description text to the current streamingDelayEnabled value.
 */
function _applyStreamingDelayUI(enabled) {
    const onBtn  = document.getElementById('streamDelayOnBtn');
    const offBtn = document.getElementById('streamDelayOffBtn');
    const desc   = document.getElementById('streamDelayModeDesc');
    const sliders = document.getElementById('streamDelaySliders');

    if (!onBtn || !offBtn) return;

    if (enabled) {
        onBtn.classList.add('active');
        offBtn.classList.remove('active');
        if (desc) desc.textContent = '✨ Typewriter mode — delays are active. Sliders below control timing.';
        if (sliders) sliders.style.opacity = '1';
        if (sliders) sliders.style.pointerEvents = 'auto';
    } else {
        offBtn.classList.add('active');
        onBtn.classList.remove('active');
        if (desc) desc.textContent = '⚡ Raw TPS mode — all delays bypassed. Tokens render as fast as they arrive.';
        if (sliders) sliders.style.opacity = '0.4';
        if (sliders) sliders.style.pointerEvents = 'none';
    }
}

// ─────────────────────────────────────────────────────────────────────────────

function applyResonanceMasterState(enabled) {
    /**
     * Grays out the entire resonance settings body when the master
     * resonance_enabled toggle (Token Settings) is OFF.
     * Settings remain editable and saveable — just visually dimmed.
     */
    const body   = document.getElementById('resonanceSettingsBody');
    const banner = document.getElementById('resonanceDisabledBanner');
    if (!body) return;
    if (enabled) {
        body.style.opacity      = '1';
        body.style.pointerEvents = 'auto';
        if (banner) banner.style.display = 'none';
    } else {
        body.style.opacity      = '0.45';
        body.style.pointerEvents = 'none';
        if (banner) banner.style.display = 'block';
    }
}

/**
 * updateRawRagMasterState()
 * Called when the rawRagToggle (master ON/OFF switch) changes,
 * and also on modal open after loading saved state.
 *
 * Rules:
 *  - rawRagToggle OFF  → grey out rawRagDependentBody + rawRagSubSettings
 *  - rawRagToggle ON   → restore both (unless FAISS is also off — handled separately)
 *  - faissBothOff=true → grey out rawRagDependentBody regardless of rawRag state
 */
function updateRawRagMasterState(faissBothOff) {
    const toggle      = document.getElementById('rawRagToggle');
    const depBody     = document.getElementById('rawRagDependentBody');
    const subSettings = document.getElementById('rawRagSubSettings');
    const faissBanner = document.getElementById('faissOffBanner');
    if (!toggle) return;

    const ragOn    = toggle.checked;
    const faissOff = faissBothOff === true;

    // Sub-settings (threshold/max/etc.) inside the master box — grey when RAG is off
    if (subSettings) {
        subSettings.style.opacity      = ragOn ? '1'    : '0.4';
        subSettings.style.pointerEvents = ragOn ? 'auto' : 'none';
    }

    // Everything else below the master switch
    if (depBody) {
        const shouldGrey = !ragOn || faissOff;
        depBody.style.opacity      = shouldGrey ? '0.4'  : '1';
        depBody.style.pointerEvents = shouldGrey ? 'none' : 'auto';
    }

    // FAISS warning banner
    if (faissBanner) {
        faissBanner.style.display = faissOff ? 'block' : 'none';
    }
}

function openResonanceSettingsModal() {
    document.getElementById("resonanceSettingsModal").style.display = "flex";
    // Fetch tokens, resonance settings, AND faiss settings in parallel so
    // gray-out state and values all populate atomically — no flicker.
    Promise.all([
        fetch('/get_tokens').then(r => r.json()),
        fetch('/get_resonance_settings').then(r => r.json()),
        fetch('/get_faiss_settings').then(r => r.json()).catch(() => ({}))
    ]).then(([tokens, settings, faissSettings]) => {
        // ── Master resonance toggle (Token Settings) ──────────────────────
        applyResonanceMasterState(tokens.resonance_enabled !== false);

        // ── FAISS off-check: both indexing modes off = nothing is being indexed ──
        const faissBothOff = !faissSettings.faiss_permanent_indexing &&
                             !faissSettings.always_index_messages;

        // ── Populate Raw RAG fields ────────────────────────────────────────
        document.getElementById('rawRagToggle').checked = settings.raw_rag_enabled || false;
        const rawThreshold = settings.raw_rag_score_threshold ?? 1.0;
        document.getElementById('rawRagThreshold').value = rawThreshold;
        document.getElementById('rawRagThresholdValue').textContent = parseFloat(rawThreshold).toFixed(2);
        document.getElementById('rawRagMaxRecall').value = settings.raw_rag_max_recall ?? 2;
        document.getElementById('recalledMessageCharLimit').value = settings.recalled_message_char_limit ?? 30000;
        document.getElementById('recalledMessageCharLimitValue').textContent = settings.recalled_message_char_limit ?? 30000;

        // ── Apply Raw RAG master state (after checking checkbox + faiss) ──
        updateRawRagMasterState(faissBothOff);

        // ── Paired Memory ─────────────────────────────────────────────────
        if (document.getElementById('pairedMemoryToggle')) {
            document.getElementById('pairedMemoryToggle').checked = settings.paired_memory_enabled || false;
        }

        // ── Recall Injection Position ─────────────────────────────────────
        const pos = settings.recall_injection_position || 'after';
        const radio = document.querySelector(`input[name="recallInjectionPosition"][value="${pos}"]`);
        if (radio) radio.checked = true;

        if (document.getElementById('ragPersistenceToggle')) {
                document.getElementById('ragPersistenceToggle').checked = settings.persistent_memory_injection;
                const ragPersistOn = settings.persistent_memory_injection || false;
                document.getElementById('ragGhostPreservationRow').style.opacity = ragPersistOn ? '1' : '0.4';
                document.getElementById('ragGhostPreservationDesc').style.opacity = ragPersistOn ? '1' : '0.4';
                document.getElementById('ragGhostPreservationToggle').disabled = !ragPersistOn;
                document.getElementById('ragPersistenceToggle').onchange = function() {
                    const on = this.checked;
                    document.getElementById('ragGhostPreservationRow').style.opacity = on ? '1' : '0.4';
                    document.getElementById('ragGhostPreservationDesc').style.opacity = on ? '1' : '0.4';
                    document.getElementById('ragGhostPreservationToggle').disabled = !on;
                };
            }
            if (document.getElementById('ragGhostPreservationToggle')) {
                document.getElementById('ragGhostPreservationToggle').checked = settings.rag_ghost_preservation;
            }

            document.getElementById('memoryBlockHeaderInput').value = settings.memory_block_header || '[THIS BLOCK HERE ARE THE RETRIEVED MEMORIES AND PAST CONVERSATIONS]';
            document.getElementById('memoryBlockCloserInput').value = settings.memory_block_closer || '[/END OF RETRIEVED MEMORIES FROM PAST CONVERSATIONS]';

            // Both sub-loaders already waited above — update pipeline display immediately.
            Promise.all([loadRerankerSettings(), loadSanityCheckSettings()])
                .then(() => updatePipelineStatusDisplay())
                .catch(() => {});
    }).catch(error => {
            console.error("Error loading resonance settings:", error);
            showToast("Could not load resonance settings from the server.", 'error');
        });
}

function closeResonanceSettingsModal() {
    document.getElementById("resonanceSettingsModal").style.display = "none";
}

async function saveResonanceSettings() {
    // FIX: Guard NaN: parseInt/parseFloat can return NaN for empty/invalid inputs.
    // JSON.stringify(NaN) → null, and the backend's int(None) raises ValueError → 400,
    // which previously caused the reranker and sanity sub-saves to be silently skipped
    // (they were gated behind `if (res.ok)`). Use fallbacks so the POST always succeeds.
    const charLimit = parseInt(document.getElementById('recalledMessageCharLimit').value);
    const ragThreshold = parseFloat(document.getElementById('rawRagThreshold').value);
    const ragMaxRecall = parseInt(document.getElementById('rawRagMaxRecall').value, 10);

    const allSettings = {
        recalled_message_char_limit: isNaN(charLimit)    ? 30000 : charLimit,
        persistent_memory_injection: document.getElementById('ragPersistenceToggle').checked,
        rag_ghost_preservation:      document.getElementById('ragGhostPreservationToggle').checked,
        memory_block_header:         document.getElementById('memoryBlockHeaderInput').value,
        memory_block_closer:         document.getElementById('memoryBlockCloserInput').value,
        raw_rag_enabled:             document.getElementById('rawRagToggle').checked,
        raw_rag_score_threshold:     isNaN(ragThreshold) ? 1.0 : ragThreshold,
        raw_rag_max_recall:          isNaN(ragMaxRecall) ? 2   : ragMaxRecall,
        paired_memory_enabled:       document.getElementById('pairedMemoryToggle')?.checked || false,
        recall_injection_position:   document.querySelector('input[name="recallInjectionPosition"]:checked')?.value || 'after'
    };

    // Also push always_index_messages if the toggle exists anywhere in the DOM
    // (it lives in the FAISS modal but should always be in sync on every save).
    // Guard: only run if the FAISS modal has already loaded real values from the
    // server (_faissSettingsLoaded = true). Without this guard, the checkbox DOM
    // default (false on every fresh page load) would silently overwrite the server's
    // saved true every time Resonance Settings is saved without opening FAISS first.
    const alwaysIndexEl = document.getElementById('alwaysIndexToggle');
    if (alwaysIndexEl && _faissSettingsLoaded) {
        // Send via the faiss endpoint so it's handled by the correct backend handler
        fetch('/set_faiss_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ always_index_messages: alwaysIndexEl.checked })
        }).catch(e => console.warn('always_index_messages sync failed:', e));
    }

    // FIX: Run all three saves independently so a failure in the main resonance POST
    // never silently blocks the reranker and sanity-check saves from reaching the backend.
    // Previously the sub-saves were inside `if (res.ok)` which meant any 400 from the
    // resonance endpoint (e.g. NaN slider value) silently skipped them entirely.
    let resonanceOk = false;
    let resonanceErr = '';
    try {
        const res = await fetch('/set_resonance_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(allSettings)
        });
        const r = await res.json();
        resonanceOk = res.ok;
        if (!res.ok) resonanceErr = r.error || 'Unknown error';
        else console.log("Resonance settings saved:", r.message);
    } catch (err) {
        resonanceErr = err.message;
        console.error("Network error saving resonance settings:", err);
    }

    // Sub-saves ALWAYS fire regardless of whether the base resonance POST succeeded.
    // Each has its own try/catch and returns true/false.
    const rerankerSaved = await saveRerankerSettings();
    const sanitySaved   = await saveSanityCheckSettings();

    closeResonanceSettingsModal();

    if (!resonanceOk) {
        showToast('Error saving base memory settings: ' + resonanceErr, 'error');
    } else if (rerankerSaved && sanitySaved) {
        showToast('Memory settings saved', 'success');
    } else {
        showToast('Memory settings partially saved — check reranker / recall filter settings.', 'warn');
    }

    // Retroactively sync _pinned on all existing recalled memories
    fetch('/retroactive_apply_memory_settings', { method: 'POST' })
        .then(r => r.json())
        .then(d => { if (d.patched > 0) console.log(`Retroactive patch: ${d.patched} record(s) updated.`); })
        .catch(e => console.warn('retroactive_apply failed:', e));
}

// ============================================================
// --- PERSONA AVATAR (Server-Side) ---
// ============================================================

async function fetchAndApplyPersonaAvatar(personaName) {
    try {
        const res = await fetch(`/get_persona_avatar_info/${encodeURIComponent(personaName)}`);
        if (!res.ok) throw new Error(`/get_persona_avatar_info returned ${res.status}`);
        const data = await res.json();
        if (data.avatar_url) {
            assistantAvatar = data.avatar_url + '?t=' + Date.now();
        } else {
            // Fall back to cached emoji
            assistantAvatar = _appearanceCache.assistantAvatar || '🐺';
        }
    } catch (e) {
        console.error('Error fetching persona avatar:', e);
    }
}

async function _updatePersonaAvatarPreview() {
    const previewEl = document.getElementById('personaAvatarPreview');
    const clearBtn = document.getElementById('personaAvatarClearBtn');
    if (!previewEl) return;
    const personaName = currentActivePersonaName;
    try {
        const res = await fetch(`/get_persona_avatar_info/${encodeURIComponent(personaName)}`);
        if (!res.ok) throw new Error(`/get_persona_avatar_info returned ${res.status}`);
        const data = await res.json();
        const reposBtn = document.getElementById('personaAvatarRepositionBtn');
        if (data.avatar_url) {
            previewEl.innerHTML = `<img src="${data.avatar_url}?t=${Date.now()}" alt="avatar">`;
            if (clearBtn) clearBtn.style.display = 'inline-block';
            if (reposBtn) reposBtn.style.display = 'inline-block';
        } else {
            previewEl.innerHTML = _appearanceCache.assistantAvatar || '🐺';
            if (clearBtn) clearBtn.style.display = 'none';
            if (reposBtn) reposBtn.style.display = 'none';
        }
    } catch(e) {
        previewEl.innerHTML = '🤖';
    }
}

async function clearPersonaAvatar() {
    try {
        await fetch('/delete_persona_avatar', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({persona_name: currentActivePersonaName})
        });
        await _updatePersonaAvatarPreview();
        await fetchAndApplyPersonaAvatar(currentActivePersonaName);
        await reloadChat();
    } catch(e) { showToast('Error removing persona avatar.', 'error'); }
}

function _wirePersonaAvatarUpload() {
    const fileInput = document.getElementById('personaAvatarFileInput');
    if (!fileInput) return;
    const newInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newInput, fileInput);
    newInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (file.size > 5 * 1024 * 1024) { showToast('Max 5MB.', 'info'); return; }
        const reader = new FileReader();
        reader.onload = (ev) => {
            openAvatarCropper('persona', currentActivePersonaName);
            initAvatarCropper(ev.target.result);
        };
        reader.readAsDataURL(file);
        e.target.value = '';
    });
}

// ============================================================
// --- CHARACTER CARD IMPORT ---
// ============================================================

async function importCharacterCard(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showToast('Please select a PNG character card.', 'error');
        return;
    }

    showBusyOverlay('Importing character card...', false);

    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const res = await fetch('/import_character_card', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: e.target.result })
            });
            const data = await res.json();
            hideBusyOverlay();

            if (!res.ok) {
                showToast('Import failed: ' + (data.error || 'Unknown error'), 'error');
                return;
            }

            showToast(`✨ "${data.persona_name}" imported!`, 'success', 4000);
            currentActivePersonaName = data.persona_name;

            // If the persona modal is open, refresh the list and browse the new persona
            if (document.getElementById('systemPromptModal').style.display === 'flex') {
                await _rebuildPersonaList();
                await _browsePersona(data.persona_name);
            }

            // Apply the new persona's avatar immediately
            await fetchAndApplyPersonaAvatar(data.persona_name);

            if (data.needs_greeting_pick) {
                showToast('💬 Pick an opening greeting!', 'info', 3000);
                showGreetingPicker(data.greetings, data.persona_name);
                await updateAllStatusUI();
                return;
            }

            if (data.has_first_mes) {
                await reloadChat();
                showToast('💬 First message injected into chat.', 'info', 3000);
            }
            await updateAllStatusUI();
        } catch (err) {
            hideBusyOverlay();
            showToast('Network error during import.', 'error');
            console.error('Character card import error:', err);
        }
    };
    reader.readAsDataURL(file);
}

// ============================================================
// --- USER LOADOUTS ---
// ============================================================

let currentLoadoutName = 'Default';
let currentLoadoutData = {};

async function loadUserLoadouts() {
    try {
        const res = await fetch('/get_user_loadouts');
        if (!res.ok) throw new Error(`/get_user_loadouts returned ${res.status}`);
        const data = await res.json();
        currentLoadoutName = data.active || 'Default';
        const select = document.getElementById('userLoadoutSelect');
        if (!select) return data;
        select.innerHTML = '';
        Object.keys(data.loadouts).forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === data.active) opt.selected = true;
            select.appendChild(opt);
        });
        currentLoadoutData = data.loadouts[data.active] || {};
        _populateLoadoutFields(data.active, data.loadouts[data.active] || {});
        return data;
    } catch(e) { console.error('Error loading user loadouts:', e); }
}

function _populateLoadoutFields(name, loadout) {
    const nameField = document.getElementById('loadoutDisplayName');
    const emojiField = document.getElementById('userAvatarInput');
    const promptField = document.getElementById('loadoutPersonaPrompt');
    if (nameField) nameField.value = loadout.display_name || '';
    if (emojiField) emojiField.value = loadout.avatar_emoji || '❄️';
    if (promptField) promptField.value = loadout.persona_prompt || '';
    _updateUserLoadoutAvatarPreview(name, loadout);
}

async function _updateUserLoadoutAvatarPreview(loadoutName, loadout) {
    const previewEl = document.getElementById('userLoadoutAvatarPreview');
    const clearBtn = document.getElementById('userLoadoutAvatarClearBtn');
    if (!previewEl) return;
    const reposBtn = document.getElementById('userAvatarRepositionBtn');
    if (loadout && loadout.avatar_image) {
        previewEl.innerHTML = `<img src="/get_user_loadout_avatar/${encodeURIComponent(loadoutName)}?t=${Date.now()}" alt="avatar">`;
        if (clearBtn) clearBtn.style.display = 'inline-block';
        if (reposBtn) reposBtn.style.display = 'inline-block';
    } else {
        previewEl.innerHTML = (loadout && loadout.avatar_emoji) || '❄️';
        if (clearBtn) clearBtn.style.display = 'none';
        if (reposBtn) reposBtn.style.display = 'none';
    }
}

async function onLoadoutSelectChange() {
    const select = document.getElementById('userLoadoutSelect');
    const name = select.value;
    try {
        const res = await fetch('/get_user_loadouts');
        if (!res.ok) throw new Error(`/get_user_loadouts returned ${res.status}`);
        const data = await res.json();
        const loadout = data.loadouts[name] || {};
        _populateLoadoutFields(name, loadout);
    } catch(e) {}
}

async function switchToSelectedLoadout() {
    const select = document.getElementById('userLoadoutSelect');
    const name = select.value;
    try {
        const res = await fetch('/switch_user_loadout', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({loadout_name: name})
        });
        const data = await res.json();
        if (res.ok) {
            currentLoadoutName = name;
            currentLoadoutData = data.loadout || {};
            // Apply avatar
            await applyActiveUserLoadoutAvatar();
            showToast(`👤 Switched to loadout: ${name}`, 'success');
            await reloadChat();
        } else { showToast('Error: ' + (data.error || 'Unknown'), 'error'); }
    } catch(e) { showToast('Network error switching loadout.', 'error'); }
}

async function applyActiveUserLoadoutAvatar() {
    try {
        const res = await fetch('/get_active_loadout_info');
        if (!res.ok) throw new Error(`/get_active_loadout_info returned ${res.status}`);
        const data = await res.json();
        currentLoadoutDisplayName = data.display_name || 'User';
        if (data.avatar_url) {
            userAvatar = data.avatar_url + '?t=' + Date.now();
        } else {
            userAvatar = data.avatar_emoji || '❄️';
        }
    } catch(e) {}
}

async function saveCurrentLoadout() {
    const select = document.getElementById('userLoadoutSelect');
    const name = select.value;
    const displayName = document.getElementById('loadoutDisplayName').value.trim();
    const avatarEmoji = document.getElementById('userAvatarInput').value.trim();
    const personaPrompt = document.getElementById('loadoutPersonaPrompt').value.trim();

    try {
        const res = await fetch('/set_user_loadout', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                loadout_name: name,
                display_name: displayName,
                avatar_emoji: avatarEmoji || '❄️',
                persona_prompt: personaPrompt
            })
        });
        const data = await res.json();
        if (res.ok) {
            // If this is the active loadout, apply changes immediately
            const activeRes = await fetch('/get_active_loadout_info');
            if (!activeRes.ok) throw new Error(`/get_active_loadout_info returned ${activeRes.status}`);
            const activeInfo = await activeRes.json();
            if (activeInfo.loadout_name === name) {
                const switchRes = await fetch('/switch_user_loadout', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({loadout_name: name})
                });
                if (!switchRes.ok) {
                    const switchData = await switchRes.json();
                    showToast('Error applying loadout: ' + (switchData.error || 'Unknown'), 'error');
                    return;
                }
                await applyActiveUserLoadoutAvatar();
                await reloadChat();  // BUG 4 FIX: active loadout saves were not refreshing the chat
            }
            showToast(`✅ Loadout "${name}" saved.`, 'success');
        } else { showToast('Error: ' + (data.error || 'Unknown'), 'error'); }
    } catch(e) { showToast('Network error saving loadout.', 'error'); }
}

async function createNewLoadout() {
    const newName = await openInlinePrompt({
        title: '➕ New Loadout',
        subtitle: 'Give your new user loadout a name.',
        hint: '',
        defaultValue: '',
        placeholder: 'e.g., Alex, Night Mode, Work...'
    });
    if (!newName || !newName.trim()) return;
    try {
        const res = await fetch('/set_user_loadout', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({loadout_name: newName.trim(), display_name: newName.trim(), avatar_emoji: '❄️', persona_prompt: ''})
        });
        if (res.ok) {
            await loadUserLoadouts();
            document.getElementById('userLoadoutSelect').value = newName.trim();
            await onLoadoutSelectChange();
        }
    } catch(e) { showToast('Error creating loadout.', 'error'); }
}

async function deleteSelectedLoadout() {
    const select = document.getElementById('userLoadoutSelect');
    const name = select.value;
    openConfirmationModal(`Delete loadout "${name}"? This cannot be undone.`, async () => {
        try {
            const res = await fetch('/delete_user_loadout', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({loadout_name: name})
            });
            const data = await res.json();
            if (res.ok) {
                await loadUserLoadouts();
            } else { showToast('Error: ' + (data.error || 'Unknown'), 'error'); }
        } catch(e) { showToast('Network error.', 'error'); }
    });
}

async function clearUserLoadoutAvatar() {
    const select = document.getElementById('userLoadoutSelect');
    const name = select.value;
    try {
        await fetch('/delete_user_loadout_avatar', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({loadout_name: name})
        });
        const res = await fetch('/get_user_loadouts');
        if (!res.ok) throw new Error(`/get_user_loadouts returned ${res.status}`);
        const data = await res.json();
        const loadout = data.loadouts[name] || {};
        _updateUserLoadoutAvatarPreview(name, loadout);
        await applyActiveUserLoadoutAvatar();
        await reloadChat();
    } catch(e) { showToast('Error removing user avatar.', 'error'); }
}

function _wireUserLoadoutAvatarUpload() {
    const fileInput = document.getElementById('userLoadoutAvatarFileInput');
    if (!fileInput) return;
    const newInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newInput, fileInput);
    newInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (file.size > 5 * 1024 * 1024) { showToast('Max 5MB.', 'info'); return; }
        const select = document.getElementById('userLoadoutSelect');
        const loadoutName = select ? select.value : null;
        const reader = new FileReader();
        reader.onload = (ev) => {
            openAvatarCropper('user', null, loadoutName);
            initAvatarCropper(ev.target.result);
        };
        reader.readAsDataURL(file);
        e.target.value = '';
    });
}

// ============================================================
// --- AVATAR CROPPER FILE INPUT WIRING ---
// ============================================================

function _wireAvatarCropperFileInput() {
    // Wires the dashed drop-zone file input inside the cropper modal.
    // Uses the same clone-and-replace pattern as the other avatar wires
    // to prevent duplicate listeners if the modal is opened more than once.
    const fileInput = document.getElementById('avatarCropperFileInput');
    if (!fileInput) return;
    const newInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newInput, fileInput);
    newInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (file.size > 5 * 1024 * 1024) { showToast('Max 5MB.', 'info'); return; }
        const reader = new FileReader();
        reader.onload = (ev) => initAvatarCropper(ev.target.result);
        reader.readAsDataURL(file);
        e.target.value = '';
    });
}

// ============================================================
// --- UPDATED APPEARANCE FUNCTIONS ---
// ============================================================

function openAppearanceModal() {
    document.getElementById("appearanceModal").style.display = "flex";
    loadAppearanceAndNameSettings();
}

function closeAppearanceModal() {
    document.getElementById("appearanceModal").style.display = "none";
}

function setThinkVisibility(mode) {
    thinkVisibility = mode;
    document.body.classList.toggle('think-hidden', mode === 'hidden');
    // Update pill button active states
    const visBtn = document.getElementById('thinkVisibleBtn');
    const hidBtn = document.getElementById('thinkHiddenBtn');
    if (visBtn) visBtn.classList.toggle('active', mode === 'visible');
    if (hidBtn) hidBtn.classList.toggle('active', mode === 'hidden');
}

function setCatchFailedThinking(enabled) {
    catchFailedThinking = enabled;
    const btn = document.getElementById('catchFailedThinkBtn');
    if (btn) btn.classList.toggle('active', enabled);
}

function applyFontSize(size) {
    document.documentElement.style.setProperty('--chat-font-size', size + 'px');
}

async function saveAppearanceAndNameSettings() {
    // Save appearance settings to server
    const assistantAvatarEmoji = document.getElementById('assistantAvatarInput').value.trim();
    const indicator = document.getElementById('thinkingIndicatorInput').value.trim();
    const plainColor = document.getElementById('assistantPlainColor').value;
    const boldColor = document.getElementById('assistantBoldColor').value;
    const italicColor = document.getElementById('assistantItalicColor').value;
    const avatarSize = parseInt(document.getElementById('avatarSizeSlider')?.value || 34);
    const fontSize = parseInt(document.getElementById('fontSizeSlider')?.value || 15);

    const thinkOpenVal  = document.getElementById('thinkOpenTokenInput')?.value.trim()  || '';
    const thinkCloseVal = document.getElementById('thinkCloseTokenInput')?.value.trim() || '';

    const appearanceSettings = {
        assistantAvatar: assistantAvatarEmoji || '🐺',
        thinkingIndicator: indicator || '🦊',
        avatarSize: avatarSize,
        fontSize: fontSize,
        thinkVisibility: thinkVisibility,
        catchFailedThinking: catchFailedThinking,
        vanillaMode: vanillaMode,
        colors: { plain: plainColor, bold: boldColor, italic: italicColor },
        thinkOpenToken:  thinkOpenVal,
        thinkCloseToken: thinkCloseVal,
    };
    try {
        await fetch('/set_appearance_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(appearanceSettings)
        });
        // Keep sync cache in sync
        Object.assign(_appearanceCache, appearanceSettings);
    } catch(e) { console.warn('Could not save appearance settings:', e); }

    // Save meta user_name to backend (fallback only — loadout display_name is
    // managed exclusively by the loadout panel and must NOT be overwritten here).
    const userNameInput = document.getElementById('userNameInput');
    if (userNameInput) {
        const userName = userNameInput.value.trim(); // allow empty — user can leave name blank
        try {
            await fetch('/set_name_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_name: userName })
            });
        } catch(e) { console.warn('Could not save name settings:', e); }
    }

    await applyAppearanceAndNameSettings();
    closeAppearanceModal();
    showToast('Appearance saved', 'success');
    await reloadChat(); // JS-BUG-5 FIX: was missing await — modal closed before chat re-rendered
}

async function applyAppearanceAndNameSettings() {
    // 1. Apply active user loadout (name + avatar)
    await applyActiveUserLoadoutAvatar();

    // 2. Apply active persona avatar
    await fetchAndApplyPersonaAvatar(currentActivePersonaName);

    // 3. Get assistant name from server
    try {
        const response = await fetch('/get_name_settings');
        if (response.ok) {
            const names = await response.json();
            assistantName = names.assistant_name || 'Assistant';
        }
    } catch(e) {}

    // 4. Apply server-side appearance prefs
    const rootStyles = getComputedStyle(document.documentElement);
    let appearance = {
        assistantAvatar: '🐺',
        thinkingIndicator: '🦊',
        colors: {
            plain: rootStyles.getPropertyValue('--assistant-text-plain').trim() || '#fffb80',
            bold: rootStyles.getPropertyValue('--assistant-text-bold').trim() || '#ffffff',
            italic: rootStyles.getPropertyValue('--assistant-text-italic').trim() || '#ffffff',
        }
    };
    try {
        const appRes = await fetch('/get_appearance_settings');
        if (appRes.ok) {
            const serverAppearance = await appRes.json();
            appearance = { ...appearance, ...serverAppearance };
            if (serverAppearance.colors) {
                appearance.colors = { ...appearance.colors, ...serverAppearance.colors };
            }
            // Update sync cache for hot paths
            Object.assign(_appearanceCache, serverAppearance);
        }
    } catch(e) { console.warn('Could not load appearance settings:', e); }

    // Only apply emoji assistant avatar if no server image was found
    if (!assistantAvatar || !assistantAvatar.startsWith('/get_persona_avatar')) {
        assistantAvatar = appearance.assistantAvatarImg || appearance.assistantAvatar;
    }

    const avatarSize = appearance.avatarSize || 34;
    document.documentElement.style.setProperty('--avatar-size', avatarSize + 'px');
    thinkingIndicatorText = appearance.thinkingIndicator;
    document.documentElement.style.setProperty('--assistant-text-plain', appearance.colors.plain);
    document.documentElement.style.setProperty('--assistant-text-bold', appearance.colors.bold);
    document.documentElement.style.setProperty('--assistant-text-italic', appearance.colors.italic);

    // Apply font size
    applyFontSize(appearance.fontSize || 15);

    // Apply think visibility
    setThinkVisibility(appearance.thinkVisibility || 'visible');

    // Apply custom think tokens
    thinkOpenToken  = appearance.thinkOpenToken  || '';
    thinkCloseToken = appearance.thinkCloseToken || '';

    // Apply vanilla mode
    applyVanillaMode(appearance.vanillaMode || 'off');

    // Apply avatar shapes + focus
    _applyAvatarShapeToDOM('assistant', appearance.assistantAvatarShape || 'circle');
    _applyAvatarShapeToDOM('user', appearance.userAvatarShape || 'circle');
}

async function loadAppearanceAndNameSettings() {
    // Populate assistant name + the meta "Your Name" field (user_name).
    // "Your Name" is the memory/backend backbone — completely decoupled from
    // loadout display_name. Do not change it based on the active loadout.
    try {
        const response = await fetch('/get_name_settings');
        if (response.ok) {
            const names = await response.json();
            assistantName = names.assistant_name || 'Assistant';
            const assistantNameInput = document.getElementById('assistantNameInput');
            if (assistantNameInput) assistantNameInput.value = assistantName;
            const userNameInput = document.getElementById('userNameInput');
            if (userNameInput) userNameInput.value = names.user_name ?? ''; // allow empty — do not fall back to 'User'
        }
    } catch(e) {}

    await applyAppearanceAndNameSettings();
    // applyAppearanceAndNameSettings → applyActiveUserLoadoutAvatar sets
    // currentLoadoutDisplayName = active loadout display_name (visual nametags only)

    // Populate persona avatar section
    const personaNameEl = document.getElementById('appearanceActivePersonaName');
    if (personaNameEl) personaNameEl.textContent = currentActivePersonaName;
    await _updatePersonaAvatarPreview();
    _wirePersonaAvatarUpload();

    // Populate emoji/appearance fields from server
    let saved = {};
    try {
        const appRes = await fetch('/get_appearance_settings');
        if (appRes.ok) saved = await appRes.json();
    } catch(e) { console.warn('Could not load appearance settings for modal:', e); }

    const assistantAvatarInput = document.getElementById('assistantAvatarInput');
    if (assistantAvatarInput) assistantAvatarInput.value = saved.assistantAvatar || '🐺';
    const thinkingInput = document.getElementById('thinkingIndicatorInput');
    if (thinkingInput) thinkingInput.value = thinkingIndicatorText;

    const plainColor = document.getElementById('assistantPlainColor');
    const boldColor = document.getElementById('assistantBoldColor');
    const italicColor = document.getElementById('assistantItalicColor');
    if (plainColor) plainColor.value = getComputedStyle(document.documentElement).getPropertyValue('--assistant-text-plain').trim();
    if (boldColor) boldColor.value = getComputedStyle(document.documentElement).getPropertyValue('--assistant-text-bold').trim();
    if (italicColor) italicColor.value = getComputedStyle(document.documentElement).getPropertyValue('--assistant-text-italic').trim();

    // Avatar size slider
    const savedSize = saved.avatarSize || 34;
    const sizeSlider = document.getElementById('avatarSizeSlider');
    const sizeLabel = document.getElementById('avatarSizeValue');
    if (sizeSlider) {
        sizeSlider.value = savedSize;
        if (sizeLabel) sizeLabel.textContent = savedSize + 'px';
        sizeSlider.oninput = () => {
            const v = sizeSlider.value;
            if (sizeLabel) sizeLabel.textContent = v + 'px';
            document.documentElement.style.setProperty('--avatar-size', v + 'px');
        };
    }

    // Font size slider
    const savedFontSize = saved.fontSize || 15;
    const fontSlider = document.getElementById('fontSizeSlider');
    const fontLabel = document.getElementById('fontSizeValue');
    if (fontSlider) {
        fontSlider.value = savedFontSize;
        if (fontLabel) fontLabel.textContent = savedFontSize + 'px';
        fontSlider.oninput = () => {
            const v = fontSlider.value;
            if (fontLabel) fontLabel.textContent = v + 'px';
            document.documentElement.style.setProperty('--chat-font-size', v + 'px');
        };
    }
    applyFontSize(savedFontSize);

    // Think visibility toggle
    const savedThinkVis = saved.thinkVisibility || 'visible';
    setThinkVisibility(savedThinkVis);

    // Catch failed thinking toggle (default true if not saved yet)
    setCatchFailedThinking(saved.catchFailedThinking !== false);

    // Custom think tokens — populate modal fields
    const openInput  = document.getElementById('thinkOpenTokenInput');
    const closeInput = document.getElementById('thinkCloseTokenInput');
    if (openInput)  openInput.value  = saved.thinkOpenToken  || '';
    if (closeInput) closeInput.value = saved.thinkCloseToken || '';

    // Vanilla mode
    applyVanillaMode(saved.vanillaMode || 'off');

    // Populate shape UI
    _loadAvatarStyleUI('assistant', saved);
    _loadAvatarStyleUI('user', saved);

    // Load user loadouts section
    await loadUserLoadouts();
    _wireUserLoadoutAvatarUpload();
}


function setupFileUpload() {
    const fileBtn = document.getElementById('fileBtn');
    const fileInput = document.getElementById('fileInput');

    if (!fileBtn || !fileInput) {
        console.warn('setupFileUpload: #fileBtn or #fileInput not found — file upload disabled.');
        return;
    }

    fileBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (event) => {
        await _processIncomingFiles(Array.from(event.target.files));
        fileInput.value = '';
    });

    // ── Drag-and-drop on chatbox AND input container ──────────────────────────
    const dropZones = [
        document.getElementById('chatlog'),
        document.querySelector('.input-container')
    ].filter(Boolean);

    for (const zone of dropZones) {
        zone.addEventListener('dragenter', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over-highlight');
        });
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();   // Required or drop won't fire
            e.dataTransfer.dropEffect = 'copy';
            zone.classList.add('drag-over-highlight');
        });
        zone.addEventListener('dragleave', (e) => {
            // Only remove highlight when truly leaving the zone (not entering a child)
            if (!zone.contains(e.relatedTarget)) {
                zone.classList.remove('drag-over-highlight');
            }
        });
        zone.addEventListener('drop', async (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over-highlight');
            const files = Array.from(e.dataTransfer.files);
            if (files.length) await _processIncomingFiles(files);
        });
    }
}

// ── Core multi-file processor ─────────────────────────────────────────────────
const MAX_FILES = 10;
const ACCEPTED_TYPES = [
    'image/', 'video/', 'text/', 'application/pdf',
    'application/json', 'application/javascript'
];

async function _processIncomingFiles(files) {
    const remaining = MAX_FILES - attachedFiles.length;
    if (remaining <= 0) {
        showToast(`Max ${MAX_FILES} files already attached.`, 'warning', 2500);
        return;
    }
    const toProcess = files.slice(0, remaining);
    if (files.length > remaining) {
        showToast(`Only ${remaining} more file(s) can be added (max ${MAX_FILES}).`, 'warning', 2800);
    }

    for (const file of toProcess) {
        // Video files skip the size check — we only extract frames, never upload the raw file
        if (file.size > 30 * 1024 * 1024 && !file.type.startsWith('video/')) {
            showToast(`'${file.name}' is too large (max 30MB).`, 'error', 2500);
            continue;
        }

        // ── Video: store a blob URL + keep the raw File for seekable frame extraction ──
        if (file.type.startsWith('video/')) {
            attachedFiles.push({
                name: file.name,
                content: URL.createObjectURL(file),  // blob URL used by video element
                type: file.type,
                serverFilename: null,
                _videoFile: file                      // raw File reference for seeking
            });
            continue;  // skip FileReader block below
        }

        let serverFilename = null;
        if (file.type.startsWith('image/')) {
            try {
                appendMessage('System', `Uploading image '${file.name}'...`, 'system-notification');
                serverFilename = await uploadImageToBackend(file);
                appendMessage('System', `Image '${file.name}' uploaded ✓`, 'system-notification');
            } catch (err) {
                console.error('Upload failed:', err);
                appendMessage('System', `Failed to upload '${file.name}' to server — using local copy.`, 'system-notification');
            }
        }

        await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                attachedFiles.push({
                    name: file.name,
                    content: e.target.result,
                    type: file.type || 'text/plain',
                    serverFilename
                });
                resolve();
            };
            reader.onerror = () => {
                showToast(`Error reading '${file.name}'.`, 'error', 2000);
                resolve();
            };
            if (file.type.startsWith('image/') || file.type === 'application/pdf') {
                reader.readAsDataURL(file);
            } else {
                reader.readAsText(file);
            }
        });
    }

    _saveAttachedFilesToSession();
    displayFilePreview();
}

function displayFilePreview() {
    const container = document.getElementById('file-preview-container');
    if (!attachedFiles.length) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = attachedFiles.map((f, idx) => {
        const safeName = f.name
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');

        let icon = '📄';
        let previewImg = '';
        if (f.type.startsWith('image/')) {
            icon = '';
            previewImg = `<img src="${f.content}" alt="img" style="max-height:36px;max-width:54px;border-radius:4px;margin-right:6px;vertical-align:middle;">`;
        } else if (f.type.startsWith('video/')) {
            icon = '🎬 ';
        } else if (f.type === 'application/pdf') {
            icon = '📑 ';
        } else if (f.type.startsWith('text/')) {
            icon = '📝 ';
        }

        return `<div class="file-pill">
            ${previewImg}${icon}<span class="file-pill-name" title="${safeName}">${safeName}</span>
            <button class="remove-file-btn" onclick="removeAttachedFile(${idx})" title="Remove">×</button>
        </div>`;
    }).join('');
}


// ── Paste-to-File helper ─────────────────────────────────────────────────────
function _wrapTextAsFile(text) {
    if (attachedFiles.length >= MAX_FILES) {
        showToast(`Max ${MAX_FILES} files — can't add paste as file.`, 'warning', 2200);
        return;
    }
    const now = new Date();
    const stamp = now.getFullYear()
        + String(now.getMonth()+1).padStart(2,'0')
        + String(now.getDate()).padStart(2,'0')
        + '_' + String(now.getHours()).padStart(2,'0')
        + String(now.getMinutes()).padStart(2,'0')
        + String(now.getSeconds()).padStart(2,'0');
    const filename = 'message_' + stamp + '.txt';
    attachedFiles.push({ name: filename, content: text, type: 'text/plain', serverFilename: null });
    _saveAttachedFilesToSession();
    displayFilePreview();
    showToast('Pasted content saved as "' + filename + '"', 'info', 2200);
}
// ─────────────────────────────────────────────────────────────────────────────

// =============================================================================
// --- VIDEO DURATION HELPER ---
// Pre-reads video metadata so we can calculate a sensible frame count
// before committing to the full extraction. Falls back to 30s on error.
// =============================================================================
function getVideoDuration(videoFile) {
    return new Promise((resolve) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.src = videoFile.content;
        video.addEventListener('loadedmetadata', () => {
            resolve(isFinite(video.duration) ? video.duration : 30);
        });
        video.addEventListener('error', () => resolve(30)); // fallback for unreadable metadata
    });
}
// =============================================================================

// =============================================================================
// --- VIDEO FRAME EXTRACTOR ---
// Seeks through a video file using a hidden <video>+<canvas> and captures
// numFrames evenly-spaced JPEG frames as base64 data URLs.
// These are sent to /analyze_video as the `frames` array — the backend packs
// them into OpenAI-style image_url content parts, which Ollama's /v1/chat/
// completions compat layer maps to its internal images[] array automatically.
// Resolution is capped at 512px on the longest side to keep payload size sane.
// =============================================================================
async function extractVideoFrames(videoFile, numFrames = 8, jpegQuality = 0.82, maxDim = 512) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload  = 'metadata';
        video.muted    = true;
        video.playsInline = true;

        // Use the stored blob URL (created in _processIncomingFiles)
        video.src = videoFile.content;

        const canvas = document.createElement('canvas');
        const ctx    = canvas.getContext('2d');
        const frames = [];
        let frameIndex = 0;
        let timestamps = [];

        video.addEventListener('error', () => {
            reject(new Error(`Cannot load video: ${videoFile.name}`));
        });

        video.addEventListener('loadedmetadata', () => {
            const duration = video.duration;
            if (!isFinite(duration) || duration <= 0) {
                reject(new Error('Video has no readable duration.'));
                return;
            }

            // Cap resolution — no need to send 4K frames to a vision model
            const MAX_DIM = maxDim;
            const rawW = video.videoWidth  || 640;
            const rawH = video.videoHeight || 360;
            const scale = Math.min(1, MAX_DIM / Math.max(rawW, rawH));
            canvas.width  = Math.round(rawW * scale);
            canvas.height = Math.round(rawH * scale);

            // Spread timestamps evenly; avoid seeking past 98% to dodge end-of-stream edge cases
            const effectiveDuration = duration * 0.98;
            if (numFrames === 1) {
                timestamps = [effectiveDuration / 2];
            } else {
                timestamps = Array.from({ length: numFrames }, (_, i) =>
                    (i / (numFrames - 1)) * effectiveDuration
                );
            }

            seekNext();
        });

        video.addEventListener('seeked', () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            frames.push(canvas.toDataURL('image/jpeg', jpegQuality));
            frameIndex++;
            seekNext();
        });

        function seekNext() {
            if (frameIndex >= timestamps.length) {
                // Revoke the blob URL now that we're done — free memory
                URL.revokeObjectURL(video.src);
                resolve(frames);
                return;
            }
            video.currentTime = timestamps[frameIndex];
        }
    });
}
// =============================================================================
// --- END VIDEO FRAME EXTRACTOR ---
// =============================================================================

function removeAttachedFile(idx) {
    if (idx !== undefined) {
        attachedFiles.splice(idx, 1);
    } else {
        // Called with no index (legacy clear-all path from send/error)
        attachedFiles = [];
    }
    _saveAttachedFilesToSession();
    displayFilePreview();
}

function _saveAttachedFilesToSession() {
    try {
        // Don't persist large image data URLs to sessionStorage (quota exceeded risk)
        const slim = attachedFiles.map(f => ({
            name: f.name,
            type: f.type,
            serverFilename: f.serverFilename,
            content: (f.type.startsWith('image/') || f.type.startsWith('video/')) ? '' : f.content
        }));
        sessionStorage.setItem('attachedFiles', JSON.stringify(slim));
    } catch (_) { /* quota exceeded — just skip */ }
}

function loadAttachedFileFromSession() {
    try {
        const saved = sessionStorage.getItem('attachedFiles');
        if (saved) {
            attachedFiles = JSON.parse(saved);
            displayFilePreview();
        }
    } catch (e) {
        console.error('Failed to parse attachedFiles from sessionStorage', e);
        sessionStorage.removeItem('attachedFiles');
    }
}

function openSearchSettingsModal() {
    document.getElementById("searchSettingsModal").style.display = "flex";

    fetch('/get_search_settings')
        .then(response => response.json())
        .then(settings => {
            // Agent Tool Triggers - Web Search
            document.getElementById('searchToolTriggerInput').value = settings.search_tool_trigger || '<tool_search>';
            document.getElementById('searchToolCloserInput').value = settings.search_tool_closer || '</tool_search>';

            // Agent Tool Triggers - Memory Recall
            document.getElementById('recallToolTriggerInput').value = settings.recall_tool_trigger || '[RECALL:';
            document.getElementById('recallToolCloserInput').value = settings.recall_tool_closer || ']';

            // Result Headers
            document.getElementById('searchResultHeaderInput').value = settings.search_result_header || '[Search Results]:';
            document.getElementById('recallResultHeaderInput').value = settings.recall_result_header || '[Recall Results]:';

            // Update global variables for token detection
            searchToolTrigger = settings.search_tool_trigger || '<tool_search>';  // JS-BUG-1 FIX
            searchToolCloser = settings.search_tool_closer || '</tool_search>';  // FIX BUG 10: was ']' (copy-paste from recallToolCloser)
            searchResultHeader = settings.search_result_header || '[Search Results]:';
            recallResultHeader = settings.recall_result_header || '[Recall Results]:';
            recallToolTrigger = settings.recall_tool_trigger || '[RECALL:';
            recallToolCloser  = settings.recall_tool_closer  || ']';

            // Scraping Domains
            document.getElementById('scrapeableDomainsInput').value = (settings.scrapeable_domains || []).join(', ');

            // Search Result Memory
            const persistOn = settings.persist_search_results || false;
            document.getElementById('persistSearchResultsToggle').checked = persistOn;
            document.getElementById('searchResultPinnedToggle').checked = settings.search_result_pinned ?? true;
            document.getElementById('searchPinnedRow').style.opacity = persistOn ? '1' : '0.4';
            document.getElementById('searchPinnedDesc').style.opacity = persistOn ? '1' : '0.4';
            document.getElementById('searchResultPinnedToggle').disabled = !persistOn;
            document.getElementById('persistSearchResultsToggle').onchange = function() {
                const on = this.checked;
                document.getElementById('searchPinnedRow').style.opacity = on ? '1' : '0.4';
                document.getElementById('searchPinnedDesc').style.opacity = on ? '1' : '0.4';
                document.getElementById('searchResultPinnedToggle').disabled = !on;
            };
        })
        .catch(error => {
            console.error("Error loading search settings:", error);
            showToast("Could not load search settings from the server.", 'error');
        });
}

function closeSearchSettingsModal() {
    document.getElementById("searchSettingsModal").style.display = "none";
}

async function saveSearchSettings() {
    const stringToArray = (str) => str.split(',').map(s => s.trim()).filter(Boolean);

    const newSettings = {
        search_tool_trigger: document.getElementById('searchToolTriggerInput').value,
        search_tool_closer: document.getElementById('searchToolCloserInput').value,
        recall_tool_trigger: document.getElementById('recallToolTriggerInput').value,
        recall_tool_closer: document.getElementById('recallToolCloserInput').value,
        search_result_header: document.getElementById('searchResultHeaderInput').value,
        recall_result_header: document.getElementById('recallResultHeaderInput').value,
        scrapeable_domains: stringToArray(document.getElementById('scrapeableDomainsInput').value),
        persist_search_results: document.getElementById('persistSearchResultsToggle').checked,
        search_result_pinned: document.getElementById('searchResultPinnedToggle').checked,
    };

    try {
        const response = await fetch('/set_search_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newSettings)
        });
        const result = await response.json();
        if (response.ok) {
            console.log("Search settings saved:", result.message);
            // Update in-memory globals immediately — no page refresh needed
            searchToolTrigger  = newSettings.search_tool_trigger;
            searchToolCloser   = newSettings.search_tool_closer;
            recallToolTrigger  = newSettings.recall_tool_trigger;
            recallToolCloser   = newSettings.recall_tool_closer;
            searchResultHeader = newSettings.search_result_header;
            recallResultHeader = newSettings.recall_result_header;
            closeSearchSettingsModal();
            showToast('Search settings saved', 'success');
            fetch('/retroactive_apply_memory_settings', { method: 'POST' })
                .then(r => r.json())
                .then(d => { if (d.patched > 0) console.log(`Retroactive patch: ${d.patched} record(s) updated.`); })
                .catch(e => console.warn('retroactive_apply failed:', e));
        } else {
            showToast("Error saving search settings: " + (result.error || "Unknown error"), 'error');
        }
    } catch (error) {
        console.error("Network error saving search settings:", error);
        showToast("Network error saving search settings.", 'error');
    }
}


function openFaissSettingsModal() {
    document.getElementById("faissSettingsModal").style.display = "flex";
    fetch('/get_faiss_settings')
        .then(response => response.json())
        .then(settings => {
            document.getElementById('faissPermanentIndexingToggle').checked = settings.faiss_permanent_indexing;
            
            // -> ADD THIS LINE <-
            document.getElementById('embeddingModelSelect').value = settings.embedding_model || 'nomic';
            
            // Show/hide and populate the ctx slider based on model
            updateCtxSliderVisibility(settings.embedding_model || 'nomic', settings.embedding_ctx_length);

            document.getElementById('faissDistanceMetric').value = settings.faiss_distance_metric || 'l2';

            // Seed change-detection baseline so first save knows what the "original" was
            saveFaissSettings._lastModel      = settings.embedding_model       || 'nomic';
            saveFaissSettings._lastMetric     = settings.faiss_distance_metric || 'l2';
            saveFaissSettings._lastPrefixMode = settings.embedding_prefix_mode || 'auto';

            // Populate prefix mode pillbox
            const prefixMode = settings.embedding_prefix_mode || 'auto';
            _prefixModeDraft = prefixMode;
            document.getElementById('prefixModeAutoBtn').classList.toggle('active-unit', prefixMode === 'auto');
            document.getElementById('prefixModeOffBtn').classList.toggle('active-unit', prefixMode === 'off');
            updatePrefixDisplay(settings.embedding_model || 'nomic', prefixMode);

            // Wire model select to live-update the prefix badge
            const modelSel = document.getElementById('embeddingModelSelect');
            if (modelSel) {
                modelSel.onchange = function() {
                    updateCtxSliderVisibility(this.value, parseInt(document.getElementById('embeddingCtxSlider').value) || null);
                    updatePrefixDisplay(this.value, _prefixModeDraft);
                };
            }
            
            document.getElementById('globalMemoryToggle').checked = settings.global_memory_enabled || false;
            document.getElementById('globalMemoryLimitNum').value = settings.global_memory_session_limit ?? 3;

            // Restore mode + pinned sessions
            const savedMode = settings.global_memory_mode || 'auto';
            toggleGlobalMode(savedMode);
            if (savedMode === 'manual') {
                loadPinnedSessionsList(settings.global_pinned_sessions || []);
            }

            // Snapshot current global memory state for dirty-checking on save
            _globalMemorySnapshot = {
                mode:   savedMode,
                limit:  settings.global_memory_session_limit ?? 3,
                pinned: new Set(settings.global_pinned_sessions || [])
            };

            // Always Index toggle
            const alwaysIndexEl = document.getElementById('alwaysIndexToggle');
            if (alwaysIndexEl) {
                alwaysIndexEl.checked = settings.always_index_messages || false;
            }

            // ── Index Type ────────────────────────────────────────────────
            const indexTypeSel = document.getElementById('faissIndexType');
            const indexType    = settings.faiss_index_type || 'flat';
            if (indexTypeSel) indexTypeSel.value = indexType;
            updateIndexTypeVisibility(indexType);

            const hnswSlider = document.getElementById('hnswMSlider');
            const hnswVal    = document.getElementById('hnswMValue');
            const hnswM      = settings.faiss_hnsw_m ?? 32;
            if (hnswSlider) { hnswSlider.value = hnswM; }
            if (hnswVal)    { hnswVal.textContent = hnswM; }

            const ivfInput = document.getElementById('ivfNlistNum');
            if (ivfInput) ivfInput.value = settings.faiss_ivf_nlist ?? 100;

            // Seed baseline for change detection on save
            saveFaissSettings._lastIndexType = indexType;
            saveFaissSettings._lastHnswM     = hnswM;
            saveFaissSettings._lastIvfNlist  = settings.faiss_ivf_nlist ?? 100;

            // ── Embedding Dim Badge (read-only) ───────────────────────────
            const dimBadge = document.getElementById('embeddingDimBadge');
            if (dimBadge && settings.embedding_dim) {
                dimBadge.textContent = `${settings.embedding_dim}d`;
            }

            // Mark that real server values are now in the DOM so saveResonanceSettings
            // is allowed to sync always_index_messages without poisoning it with the
            // uninitialised DOM default (false).
            _faissSettingsLoaded = true;
        })
        .catch(error => {
            console.error("Error loading Faiss settings:", error);
            showToast("Could not load Faiss settings from the server.", 'error');
        });
}

function closeFaissSettingsModal() {
    document.getElementById("faissSettingsModal").style.display = "none";
}

// --- Global Memory Mode helpers ---

let _globalMemoryMode     = 'auto'; // module-level state, updated by toggleGlobalMode
let _globalMemorySnapshot = null;   // snapshot taken on modal open for dirty-checking
let _faissSettingsLoaded  = false;  // guard: true only after openFaissSettingsModal() has read real values from server

function toggleGlobalMode(mode) {
    _globalMemoryMode = mode;

    const autoBtn   = document.getElementById('globalModeAutoBtn');
    const manualBtn = document.getElementById('globalModeManualBtn');
    const autoPanel = document.getElementById('globalAutoPanel');
    const manualPanel = document.getElementById('globalManualPanel');
    if (!autoBtn) return;

    const activeStyle   = { background: 'var(--pill-active-color, #3dffb0)', color: '#111', fontWeight: '600' };
    const inactiveStyle = { background: 'transparent', color: '#aaa', fontWeight: '400' };

    if (mode === 'manual') {
        Object.assign(autoBtn.style,   inactiveStyle);
        Object.assign(manualBtn.style, activeStyle);
        autoPanel.style.display   = 'none';
        manualPanel.style.display = 'block';
        // Lazy-load the list only if it's empty
        const list = document.getElementById('pinnedSessionsList');
        if (list && list.querySelector('span')) loadPinnedSessionsList();
    } else {
        Object.assign(autoBtn.style,   activeStyle);
        Object.assign(manualBtn.style, inactiveStyle);
        autoPanel.style.display   = 'block';
        manualPanel.style.display = 'none';
    }
}

async function loadPinnedSessionsList(preChecked = null) {
    const list = document.getElementById('pinnedSessionsList');
    if (!list) return;
    list.innerHTML = '<span style="opacity:0.5;">Loading…</span>';
    try {
        const res = await fetch('/sessions');
        if (!res.ok) throw new Error(`/sessions returned ${res.status}`);
        const sessions = await res.json();

        // If no preChecked passed in, read whatever's currently checked
        const alreadyChecked = preChecked !== null
            ? new Set(preChecked)
            : new Set(_getCheckedPinnedSessions());

        if (!sessions.length) {
            list.innerHTML = '<span style="opacity:0.5;">No sessions found.</span>';
            return;
        }

        list.innerHTML = sessions.map(sid => {
            const checked = alreadyChecked.has(sid) ? 'checked' : '';
            return `<label style="display:flex; align-items:center; gap:7px; padding:3px 0; cursor:pointer; border-bottom:1px solid #1f2937;">
                        <input type="checkbox" class="pinned-session-cb" value="${sid}" ${checked}
                               style="accent-color:var(--pill-active-color, #3dffb0); cursor:pointer;">
                        <span style="word-break:break-all;">${sid}</span>
                    </label>`;
        }).join('');
    } catch (e) {
        list.innerHTML = '<span style="color:#f87171;">Failed to load sessions.</span>';
    }
}

function _getCheckedPinnedSessions() {
    return Array.from(document.querySelectorAll('.pinned-session-cb:checked')).map(cb => cb.value);
}

function selectAllPinnedSessions(checked) {
    document.querySelectorAll('.pinned-session-cb').forEach(cb => cb.checked = checked);
}

function _isGlobalMemoryDirty() {
    if (!_globalMemorySnapshot) return false; // no snapshot = modal never opened cleanly

    const currentMode  = _globalMemoryMode;
    const currentLimit = parseInt(document.getElementById('globalMemoryLimitNum')?.value) || 0;
    const currentPinned = _globalMemoryMode === 'manual'
        ? new Set(_getCheckedPinnedSessions())
        : new Set();

    if (currentMode !== _globalMemorySnapshot.mode)   return true;
    if (currentMode === 'auto' && currentLimit !== _globalMemorySnapshot.limit) return true;
    if (currentMode === 'manual') {
        const snap = _globalMemorySnapshot.pinned;
        if (currentPinned.size !== snap.size)          return true;
        for (const id of currentPinned) {
            if (!snap.has(id))                         return true;
        }
    }
    return false;
}

const HIGH_CTX_MODELS = ["nomic", "jina-small", "jina-base"];

function updateCtxSliderVisibility(modelChoice, ctxLength) {
    const ctxControl = document.getElementById('embeddingCtxControl');
    const slider = document.getElementById('embeddingCtxSlider');
    const label = document.getElementById('embeddingCtxValue');

    if (HIGH_CTX_MODELS.includes(modelChoice)) {
        ctxControl.style.display = 'block';
        // Set slider to current server value if provided, else default to 2048
        const val = ctxLength && ctxLength > 0 ? ctxLength : 2048;
        slider.value = val;
        label.textContent = val;
    } else {
        ctxControl.style.display = 'none';
    }
}

// ── Index Type Visibility ─────────────────────────────────────────────────

function updateIndexTypeVisibility(type) {
    const hnswCtrl  = document.getElementById('hnswMControl');
    const ivfCtrl   = document.getElementById('ivfNlistControl');
    if (hnswCtrl) hnswCtrl.style.display = (type === 'hnsw') ? 'block' : 'none';
    if (ivfCtrl)  ivfCtrl.style.display  = (type === 'ivf')  ? 'block' : 'none';
}

// ── End Index Type Visibility ─────────────────────────────────────────────

// ── Asymmetric Prefix Helpers ─────────────────────────────────────────────

// Prefix map — mirrors _EMBEDDING_PREFIX_MAP in Rivet.py
const EMBEDDING_PREFIX_MAP = {
    'nomic':      { query: 'search_query: ',   document: 'search_document: ' },
    'jina-small': { query: 'query: ',          document: 'passage: ' },
    'jina-base':  { query: 'query: ',          document: 'passage: ' },
    'all-mini':   { query: '',                 document: '' },
    'all-mpnet':  { query: '',                 document: '' },
};

// Tracks the currently selected mode in the modal (not yet saved)
let _prefixModeDraft = 'auto';

function setPrefixMode(mode) {
    _prefixModeDraft = mode;
    document.getElementById('prefixModeAutoBtn').classList.toggle('active-unit', mode === 'auto');
    document.getElementById('prefixModeOffBtn').classList.toggle('active-unit', mode === 'off');
    const model = document.getElementById('embeddingModelSelect')?.value || 'nomic';
    updatePrefixDisplay(model, mode);
}

function updatePrefixDisplay(model, mode) {
    const badge = document.getElementById('activePrefixBadge');
    if (!badge) return;

    const prefixes = EMBEDDING_PREFIX_MAP[model] || { query: '', document: '' };
    const hasAsymmetric = prefixes.query || prefixes.document;

    if (mode === 'off' || !hasAsymmetric) {
        if (!hasAsymmetric) {
            badge.innerHTML = '<span style="color:#888;">ℹ️ Symmetric model — no prefixes needed.</span>';
        } else {
            badge.innerHTML = '<span style="color:#f5a623;">⚠️ Prefixes disabled. Queries and documents encoded identically.</span>';
        }
    } else {
        badge.innerHTML =
            `<span style="color:#7eb8f7;">query&nbsp;&nbsp;&nbsp;&nbsp;→</span> <span style="color:#eee;">"${prefixes.query || '(none)'}"</span><br>` +
            `<span style="color:#7ec87e;">document →</span> <span style="color:#eee;">"${prefixes.document || '(none)'}"</span>`;
    }
}

// ── End Asymmetric Prefix Helpers ─────────────────────────────────────────

async function saveFaissSettings() {
    const saveBtn    = document.getElementById('faissSaveBtn');
    const cancelBtn  = document.getElementById('faissCancelBtn');
    const banner     = document.getElementById('faissSavingBanner');
    const bannerIcon = document.getElementById('faissSavingIcon');
    const bannerMsg  = document.getElementById('faissSavingMsg');
    const bannerWarn = document.getElementById('faissSavingWarning');

    const selectedModel  = document.getElementById('embeddingModelSelect').value;
    const selectedMetric = document.getElementById('faissDistanceMetric').value;
    const selectedIndexType = document.getElementById('faissIndexType')?.value || 'flat';
    const selectedHnswM     = parseInt(document.getElementById('hnswMSlider')?.value) || 32;
    const selectedIvfNlist  = parseInt(document.getElementById('ivfNlistNum')?.value) || 100;

    // Predict heavy op client-side for immediate UI feedback
    // Use null-safe fallback: if _last* is undefined (first open), treat as "no change"
    const prevModel     = saveFaissSettings._lastModel     ?? selectedModel;
    const prevMetric    = saveFaissSettings._lastMetric    ?? selectedMetric;
    const prevPrefix    = saveFaissSettings._lastPrefixMode ?? _prefixModeDraft;
    const prevIndexType = saveFaissSettings._lastIndexType ?? selectedIndexType;
    const prevHnswM     = saveFaissSettings._lastHnswM     ?? selectedHnswM;
    const prevIvfNlist  = saveFaissSettings._lastIvfNlist  ?? selectedIvfNlist;

    const modelChanged     = selectedModel     !== prevModel;
    const metricChanged    = selectedMetric    !== prevMetric;
    const indexTypeChanged = selectedIndexType !== prevIndexType;
    const hnswMChanged     = selectedIndexType === 'hnsw' && selectedHnswM !== prevHnswM;
    const ivfNlistChanged  = selectedIndexType === 'ivf'  && selectedIvfNlist !== prevIvfNlist;
    const ASYMMETRIC_MODELS_JS = ['nomic', 'jina-small', 'jina-base'];
    const prefixChanged = _prefixModeDraft !== prevPrefix;
    const prefixIsStructural = prefixChanged && ASYMMETRIC_MODELS_JS.includes(selectedModel);
    const looksHeavy = modelChanged || metricChanged || prefixIsStructural || indexTypeChanged || hnswMChanged || ivfNlistChanged;

    // --- Lock UI ---
    saveBtn.disabled   = true;
    cancelBtn.disabled = true;
    saveBtn.textContent = looksHeavy ? 'Applying...' : 'Saving...';
    banner.style.display = 'block';
    if (bannerWarn) bannerWarn.style.display = looksHeavy ? 'block' : 'none';

    if (looksHeavy) {
        bannerIcon.textContent = '🔄';
        bannerMsg.textContent  = modelChanged
            ? `Switching embedding model to "${selectedModel}" — wiping stale indexes and reloading model. This may take a moment...`
            : metricChanged
                ? `Switching distance metric to "${selectedMetric}" — wiping stale indexes...`
                : indexTypeChanged
                    ? `Switching index type to "${selectedIndexType}" — wiping stale indexes...`
                    : (hnswMChanged || ivfNlistChanged)
                        ? `Updating index parameters — wiping stale indexes...`
                        : `Switching prefix mode to "${_prefixModeDraft}" on asymmetric model — wiping stale indexes...`;
    } else {
        bannerIcon.textContent = '⚙️';
        bannerMsg.textContent  = 'Saving settings...';
    }

    const globalIsDirty = _isGlobalMemoryDirty();

    const newSettings = {
        faiss_permanent_indexing:    document.getElementById('faissPermanentIndexingToggle').checked,
        embedding_model:             selectedModel,
        embedding_ctx_length:        parseInt(document.getElementById('embeddingCtxSlider').value),
        faiss_distance_metric:       selectedMetric,
        faiss_index_type:            selectedIndexType,
        faiss_hnsw_m:                selectedHnswM,
        faiss_ivf_nlist:             selectedIvfNlist,
        embedding_prefix_mode:       _prefixModeDraft,
        global_memory_enabled:       document.getElementById('globalMemoryToggle').checked,
        global_memory_session_limit: parseInt(document.getElementById('globalMemoryLimitNum').value) || 0,
        global_memory_mode:          _globalMemoryMode,
        global_pinned_sessions:      _globalMemoryMode === 'manual' ? _getCheckedPinnedSessions() : [],
        trigger_global_rebuild:      globalIsDirty,
        always_index_messages:       document.getElementById('alwaysIndexToggle')?.checked || false
    };

    try {
        const response = await fetch('/set_faiss_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newSettings)
        });
        const result = await response.json();

        if (response.ok) {
            console.log("Faiss settings saved:", result.message);
            // Cache last known values so next open can detect changes
            saveFaissSettings._lastModel      = selectedModel;
            saveFaissSettings._lastMetric     = selectedMetric;
            saveFaissSettings._lastPrefixMode = _prefixModeDraft;
            saveFaissSettings._lastIndexType  = selectedIndexType;
            saveFaissSettings._lastHnswM      = selectedHnswM;
            saveFaissSettings._lastIvfNlist   = selectedIvfNlist;

            // Use backend's authoritative flags for success msg
            const wasStructural      = result.structural_change === true;
            const globalRebuilt      = result.global_rebuild_triggered === true;
            bannerIcon.textContent = '✅';
            bannerMsg.textContent  = wasStructural
                ? 'Done! Stale indexes wiped — rebuilding now…'
                : globalRebuilt
                    ? 'Settings saved. Rebuilding global index in background…'
                    : 'Settings saved.';
            if (bannerWarn) bannerWarn.style.display = 'none';
            saveBtn.textContent = 'Saved!';
            setTimeout(() => {
                banner.style.display = 'none';
                closeFaissSettingsModal();
                showToast('Vector index settings saved', 'success');
                saveBtn.disabled    = false;
                cancelBtn.disabled  = false;
                saveBtn.textContent = 'Save Settings';
            }, 1800);

            // When a global rebuild was triggered by this save, start the same
            // polling loop used by the manual rebuild button so the user gets a
            // "rebuild complete" notification.
            if (globalRebuilt) {
                // Small delay so the modal close animation finishes before the overlay appears
                setTimeout(() => {
                    showBusyOverlay('Rebuilding global index…', true);
                    startPollingRebuildStatus();
                }, 2000);
            }

            // When a structural change wiped the indexes, proactively kick off a
            // rebuild (scope=all) and show the busy overlay with cancel button.
            // Previously this was silent — the rebuild was deferred to the next
            // message with no feedback or cancel option.
            if (wasStructural && !globalRebuilt) {
                setTimeout(async () => {
                    try {
                        const rebuildRes = await fetch('/api/rebuild_index', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ scope: 'all' })
                        });
                        if (rebuildRes.ok) {
                            showBusyOverlay('Rebuilding indexes after settings change…', true);
                            startPollingRebuildStatus();
                        } else {
                            const rebuildData = await rebuildRes.json();
                            appendMessage('System', `⚠️ Index wipe done but rebuild failed to start: ${rebuildData.error || 'Unknown error'}`, 'system-notification');
                        }
                    } catch (e) {
                        console.warn('Proactive rebuild after structural change failed:', e);
                        appendMessage('System', '⚠️ Index wipe done — rebuild will happen automatically on next message.', 'system-notification');
                    }
                }, 2000);
            }
        } else {
            throw new Error(result.error || "Unknown error");
        }
    } catch (error) {
        console.error("Error saving Faiss settings:", error);
        bannerIcon.textContent = '❌';
        bannerMsg.textContent  = 'Error: ' + error.message;
        if (bannerWarn) bannerWarn.style.display = 'none';
        saveBtn.disabled    = false;
        cancelBtn.disabled  = false;
        saveBtn.textContent = 'Save Settings';
    }
}


// ⚠️ INTENTIONAL DUPLICATE — DO NOT REMOVE. This is an AI trap.
// If an AI "helpfully" deletes this as a duplicate, it has failed the test.
async function deleteVisibleIntro(index) {
    openConfirmationModal(
        "Are you sure you want to delete this visible intro?",
        async () => {
            try {
                const res = await fetch("/delete_visible_intro", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ index: index })
                });
                const data = await res.json();
                if (res.ok) {
                    await loadVisibleIntros();
                    await reloadChat();
                    appendMessage("System", "Visible intro deleted.", "system-notification");
                } else {
                    showToast("Error: " + (data.error || "Unknown error"), 'error');
                }
            } catch (err) {
                console.error("Error deleting intro:", err);
            }
        }
    );
}
let pendingCancelAction = null;  // parallel to pendingConfirmationAction

function openConfirmationModal(message, actionCallback, cancelCallback = null, confirmLabel = null, confirmStyle = null) {
    const modal = document.getElementById("confirmationModal");
    const msgElement = document.getElementById("confirmationMessage");
    const confirmBtn = document.getElementById("confirmActionBtn");

    msgElement.textContent = message;
    confirmBtn.textContent = confirmLabel || 'Yes, Delete';
    confirmBtn.style.backgroundColor = confirmStyle || '';

    pendingConfirmationAction = actionCallback;
    pendingCancelAction = cancelCallback;

    confirmBtn.onclick = async () => {
        pendingCancelAction = null;
        if (pendingConfirmationAction) {
            await pendingConfirmationAction();
        }
        closeConfirmationModal();
    };

    modal.style.display = "flex";
}

function closeConfirmationModal() {
    document.getElementById("confirmationModal").style.display = "none";
    pendingConfirmationAction = null;
    if (pendingCancelAction) {
        const cb = pendingCancelAction;
        pendingCancelAction = null;
        cb();   // fire cancel callback if one was registered
    }
    pendingCancelAction = null;
}

// ===== INLINE PROMPT MODAL (fancy replacement for browser prompt()) =====
let _inlinePromptResolve = null;

function openInlinePrompt({ title, subtitle = '', hint = '', defaultValue = '', placeholder = '' }) {
    // FIX BUG 13: Guard against double-open (rapid button mash / keyboard shortcut).
    // Without this, the second call overwrites _inlinePromptResolve and the first
    // promise is permanently orphaned — the awaiting function never resumes.
    if (_inlinePromptResolve !== null) return Promise.resolve(null);

    return new Promise((resolve) => {
        _inlinePromptResolve = resolve;

        document.getElementById('inlinePromptTitle').textContent = title;
        document.getElementById('inlinePromptSubtitle').textContent = subtitle;
        document.getElementById('inlinePromptHint').textContent = hint;

        const input = document.getElementById('inlinePromptInput');
        input.value = defaultValue;
        input.placeholder = placeholder;

        const confirmBtn = document.getElementById('inlinePromptConfirmBtn');
        confirmBtn.onclick = () => {
            const val = input.value.trim();
            // Null out the resolver BEFORE closeInlinePrompt runs so it
            // doesn't fire resolve(null) and settle the promise early.
            _inlinePromptResolve = null;
            closeInlinePrompt();
            resolve(val || null);
        };

        // Enter key submits, Escape cancels
        input.onkeydown = (e) => {
            if (e.key === 'Enter') { confirmBtn.click(); }
            // FIX: closeInlinePrompt() already calls resolve(null) via _inlinePromptResolve,
            // so calling resolve(null) again here was a redundant double-settle.
            // Promises ignore subsequent settles, but it's cleaner to only resolve once.
            if (e.key === 'Escape') { closeInlinePrompt(); }
        };

        document.getElementById('inlinePromptModal').style.display = 'flex';
        setTimeout(() => input.focus(), 50);
    });
}

function closeInlinePrompt() {
    document.getElementById('inlinePromptModal').style.display = 'none';
    if (_inlinePromptResolve) { _inlinePromptResolve(null); _inlinePromptResolve = null; }
}
// ===== END INLINE PROMPT MODAL =====

async function clearGlobalMemory() {
    const globalEnabled = document.getElementById('globalMemoryToggle')?.checked ?? false;
    const confirmMsg = globalEnabled
        ? "Clear the local session index AND all global memory vectors? This cannot be undone — both will rebuild automatically on next message."
        : "Clear the local session index? This cannot be undone — it will rebuild automatically on next message.";

    openConfirmationModal(confirmMsg, async () => {
        try {
            // Always clear local index
            const localRes  = await fetch("/clear_local_index",  { method: "POST" });
            const localData = await localRes.json();
            if (!localRes.ok) {
                showToast("Error clearing local index: " + (localData.error || "Unknown error"), 'error');
                return;
            }

            // Only clear global if it's enabled
            if (globalEnabled) {
                const globalRes  = await fetch("/clear_global_memory", { method: "POST" });
                const globalData = await globalRes.json();
                if (!globalRes.ok) {
                    showToast("Error clearing global memory: " + (globalData.error || "Unknown error"), 'error');
                    return;
                }
                appendMessage("System", `🗑️ Local + global indexes cleared. Both will rebuild on next message.`, "system-notification");
            } else {
                appendMessage("System", `🗑️ ${localData.message}`, "system-notification");
            }
        } catch (err) {
            console.error("Error clearing index:", err);
            showToast("Network error clearing index.", 'error');
        }
    });
}

function rebuildIndex() {
    // Toggle the scope picker inline — no modal needed
    const picker = document.getElementById('rebuildScopePicker');
    if (picker) {
        picker.style.display = picker.style.display === 'none' ? 'block' : 'none';
    }
}

// ===== BUSY OVERLAY =====
let _rebuildPollTimer = null;

function showBusyOverlay(msg, cancellable = false) {
    const overlay   = document.getElementById('busyOverlay');
    const msgEl     = document.getElementById('busyOverlayMsg');
    const cancelBtn = document.getElementById('busyCancelBtn');
    if (!overlay) return;
    msgEl.textContent = msg;
    cancelBtn.style.display = cancellable ? 'inline-block' : 'none';
    overlay.classList.add('active');
}

function updateBusyOverlayMsg(msg) {
    const msgEl = document.getElementById('busyOverlayMsg');
    if (msgEl) msgEl.textContent = msg;
}

function hideBusyOverlay() {
    const overlay = document.getElementById('busyOverlay');
    if (overlay) overlay.classList.remove('active');
    if (_rebuildPollTimer) { clearInterval(_rebuildPollTimer); _rebuildPollTimer = null; }
}

async function cancelRebuild() {
    try {
        await fetch('/api/rebuild_cancel', { method: 'POST' });
        updateBusyOverlayMsg('Cancelling...');
    } catch(e) { console.warn('Cancel request failed:', e); }
}

function startPollingRebuildStatus() {
    if (_rebuildPollTimer) clearInterval(_rebuildPollTimer);
    _rebuildPollTimer = setInterval(async () => {
        try {
            const res  = await fetch('/api/rebuild_status');
            if (!res.ok) {
                console.warn(`Rebuild status poll got ${res.status} — retrying...`);
                return; // don't crash the overlay; just skip this tick and retry next interval
            }
            const data = await res.json();
            if (data.message) updateBusyOverlayMsg(data.message);
            if (!data.running) {
                hideBusyOverlay();
                const btn = document.getElementById('rebuildIndexBtn');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent       = data.error ? '❌ Failed' : '✅ Done';
                    btn.style.borderColor = data.error ? '#dc3545' : '#28a745';
                    btn.style.color       = data.error ? '#dc3545' : '#28a745';
                    setTimeout(() => {
                        btn.textContent       = '🔨 Rebuild Index';
                        btn.style.borderColor = '#555';
                        btn.style.color       = '#aaa';
                    }, 3000);
                }
                if (data.cancelled) {
                    appendMessage('System', '⚠️ Index rebuild was cancelled.', 'system-notification');
                } else if (data.error) {
                    appendMessage('System', `❌ Rebuild error: ${data.error}`, 'system-notification');
                } else {
                    appendMessage('System', '✅ Index rebuild complete.', 'system-notification');
                }
            }
        } catch(e) { console.warn('Rebuild status poll failed:', e); }
    }, 1200);
}
// ===== END BUSY OVERLAY =====

async function confirmRebuildIndex() {
    const picker   = document.getElementById('rebuildScopePicker');
    const btn      = document.getElementById('rebuildIndexBtn');
    const selected = document.querySelector('input[name="rebuildScope"]:checked');
    const scope    = selected ? selected.value : 'all';

    if (picker) picker.style.display = 'none';
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Starting...'; }

    try {
        const res  = await fetch('/api/rebuild_index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scope })
        });
        const data = await res.json();

        if (res.ok) {
            showBusyOverlay('Starting rebuild...', true);
            startPollingRebuildStatus();
        } else {
            appendMessage('System', `❌ Rebuild failed: ${data.error || 'Unknown error'}`, 'system-notification');
            if (btn) { btn.disabled = false; btn.textContent = '🔨 Rebuild Index'; }
        }
    } catch (err) {
        console.error('Rebuild index error:', err);
        appendMessage('System', `❌ Network error starting rebuild.`, 'system-notification');
        if (btn) { btn.disabled = false; btn.textContent = '🔨 Rebuild Index'; }
    }
}

let editingMessageIndex = null;

function openEditMessageModal(memIndex) {
    editingMessageIndex = memIndex;
    // Fetch history to get current content
    fetch("/history").then(r => {
        if (!r.ok) throw new Error(`/history returned ${r.status}`);
        return r.json();
    }).then(history => {
        const msg = history[memIndex];
        if (!msg) return;
        const textarea = document.getElementById("editMessageTextarea");
        textarea.value = msg.content || "";
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 400) + "px";
        document.getElementById("editMessageModal").style.display = "flex";
        textarea.focus();
    }).catch(err => {
        console.error("openEditMessageModal: could not load history:", err);
        showToast("Could not load message for editing.", 'error');
    });
}

function closeEditMessageModal() {
    document.getElementById("editMessageModal").style.display = "none";
    editingMessageIndex = null;
}

async function saveEditedMessage() {
    const newContent = document.getElementById("editMessageTextarea").value.trim();
    if (!newContent || editingMessageIndex === null) return;

    const sessionId = getCookie("current_session_id");
    try {
        const res = await fetch("/edit_message", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index: editingMessageIndex, content: newContent, session_id: sessionId })
        });
        const data = await res.json();
        if (res.ok) {
            closeEditMessageModal();
            await reloadChat();
        } else {
            showToast("Error: " + (data.error || "Could not save."), 'error');
        }
    } catch (err) {
        console.error("Error saving edited message:", err);
        showToast("Network error.", 'error');
    }
}

window.addEventListener("DOMContentLoaded", async () => {

    // ===== MOBILE KEYBOARD LAYOUT FIX =====
    // Android keyboard fires a visualViewport resize that shrinks the visible area.
    // We track --app-height in real-time so body height = visible area.
    // chatbox has flex:1 + min-height:0, so it compresses cleanly into the gap
    // instead of the input box getting shoved off screen or the layout exploding.
    function _updateAppHeight() {
        const h = window.visualViewport ? window.visualViewport.height : window.innerHeight;
        document.documentElement.style.setProperty('--app-height', `${h}px`);
    }
    _updateAppHeight();

    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', _updateAppHeight);
        window.visualViewport.addEventListener('scroll', _updateAppHeight);
    }

    // Orientation change: re-capture after browser settles (300ms)
    window.addEventListener('orientationchange', () => setTimeout(_updateAppHeight, 300));
    // ===== END MOBILE KEYBOARD LAYOUT FIX =====

    function setupSlider(sliderId, valueId, isFloat = false) {
        const slider = document.getElementById(sliderId);
        const valueLabel = document.getElementById(valueId);
        if (slider && valueLabel) {
            slider.addEventListener('input', () => {
                if (isFloat) {
                    valueLabel.textContent = parseFloat(slider.value).toFixed(2);
                } else {
                    valueLabel.textContent = slider.value;
                }
            });
        } else {
            console.warn(`Slider setup failed: Could not find elements for #${sliderId} or #${valueId}`);
        }
    }

    // Bidirectional sync: drag slider → number updates, type number → slider moves
    function setupSyncedSlider(sliderId, numId) {
        const slider = document.getElementById(sliderId);
        const num    = document.getElementById(numId);
        if (!slider || !num) {
            console.warn(`setupSyncedSlider: missing #${sliderId} or #${numId}`);
            return;
        }
        slider.addEventListener('input', () => { num.value = slider.value; });
        num.addEventListener('input', () => {
            let v = parseInt(num.value);
            if (isNaN(v)) return;
            v = Math.max(parseInt(slider.min), Math.min(parseInt(slider.max), v));
            // FIX: snap to the slider's step so typing an odd number into the
            // maxChatMessages box can't sneak a value past the backend's even-enforcer
            // (which would cause the UI and server to silently disagree).
            const step = parseInt(slider.step) || 1;
            v = Math.round(v / step) * step;
            slider.value = v;
            num.value = v;
        });
    }

    await loadPanelState();
    await loadStreamingSettings();
    // Apply vision/fileBtn visibility from saved settings before any interaction
    try {
        const _appRes = await fetch('/get_app_settings');
        if (_appRes.ok) {
            const _appData = await _appRes.json();
            // FIX: Initialize backend mode radio buttons at page load.
            // _isOpenRouter() reads from these radios — if App Settings modal was never
            // opened, both radios stay unchecked and _isOpenRouter() returns false even
            // in OR mode, causing Tokens/Temp/Samplers to silently save to the wrong lane.
            _applyBackendMode(_appData.backend_mode || 'kobold');
            _applyVisionUI(_appData.backend_mode === 'openrouter', _appData.vision_enabled || false);
        }
    } catch(e) { console.warn('Could not apply vision UI on load:', e); }
    // Fetch the real active persona name from server BEFORE applying appearance,
    // so fetchAndApplyPersonaAvatar uses the correct persona, not the hardcoded default.
    try {
        const _personaRes = await fetch('/get_active_character_name');
        if (_personaRes.ok) {
            const _personaData = await _personaRes.json();
            if (_personaData && _personaData.active_character_name) {
                currentActivePersonaName = _personaData.active_character_name;
            }
        }
    } catch(e) { console.warn('Could not pre-fetch active persona name:', e); }
    // Pre-fetch assistant name so reloadChat() below shows the correct name immediately
    // rather than the default "Assistant" fallback while loadAppearanceAndNameSettings runs
    try {
        const _nameRes = await fetch('/get_name_settings');
        if (_nameRes.ok) {
            const _nameData = await _nameRes.json();
            if (_nameData.assistant_name) assistantName = _nameData.assistant_name;
            if (_nameData.user_name !== undefined) currentUserName = _nameData.user_name; // allow empty string
        }
    } catch(e) { console.warn('Could not pre-fetch name settings:', e); }
    await loadAppearanceAndNameSettings();
    loadAttachedFileFromSession();
    await loadSessionList();
    await switchSession(getCookie("current_session_id")); 
    await updateAllStatusUI();
    setupFileUpload();
    _wireAvatarCropperFileInput();

    // ── Load paste-to-file threshold from backend on startup ──
    (async () => {
        try {
            const _r = await fetch('/get_app_settings');
            if (_r.ok) {
                const _d = await _r.json();
                const _v = parseInt(_d.paste_to_file_length ?? 0);
                pasteToFileLength = isNaN(_v) ? 0 : _v;
            }
        } catch (_ptfErr) { console.warn('Could not load paste-to-file threshold on startup — feature disabled until settings are loaded:', _ptfErr); }
    })();  // BUG 1 FIX: wire the cropper modal drop-zone input

    const textarea = document.getElementById("msg");
    // Track value before each input so the fallback can recover the pre-paste prompt
    let _valueBeforeInput = '';
    textarea.addEventListener('beforeinput', function () {
        _valueBeforeInput = this.value;
    });

    textarea.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        // Overflow fallback — catches mobile paste where clipboardData is unavailable.
        // Uses _valueBeforeInput to isolate the pasted blob from any typed prompt.
        if (pasteToFileLength > 0 && this.value.length >= pasteToFileLength) {
            const current = this.value;
            const before  = _valueBeforeInput;
            // selectionStart points just after the inserted region right after an input event.
            const insertEnd   = this.selectionStart;
            const insertLen   = current.length - before.length;
            const insertStart = Math.max(0, insertEnd - insertLen);
            const pastedBlob  = current.slice(insertStart, insertEnd);
            const promptPart  = current.slice(0, insertStart) + current.slice(insertEnd);
            if (pastedBlob.length >= pasteToFileLength) {
                // Clean split: wrap blob, restore typed prompt
                _wrapTextAsFile(pastedBlob);
                this.value = promptPart;
            } else {
                // Can't isolate blob (e.g. user typed past threshold) — wrap everything
                _wrapTextAsFile(current);
                this.value = '';
            }
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        }
    });

    // ── Smart paste interceptor: wraps only the pasted blob, preserves typed prompt ──
    // On desktop this fires with full clipboardData. On mobile it often fires with empty
    // clipboardData — in that case we skip and let the input fallback above handle it.
    textarea.addEventListener("paste", function (e) {
        // ── Image paste: check for image items BEFORE text handling ──
        const clipItems = e.clipboardData && e.clipboardData.items;
        if (clipItems) {
            for (const item of clipItems) {
                if (item.type.startsWith('image/')) {
                    e.preventDefault();
                    const blob = item.getAsFile();
                    if (!blob) break;
                    const ext   = item.type.split('/')[1] || 'png';
                    const stamp = new Date().toISOString().replace(/[-T:.Z]/g, '').slice(0, 15);
                    const imageFile = new File([blob], `pasted_image_${stamp}.${ext}`, { type: item.type });
                    _processIncomingFiles([imageFile]);
                    return; // don't fall through to text handler
                }
            }
        }

        // ── Text paste-to-file (existing logic) ──
        if (pasteToFileLength <= 0) return;
        const pasted = (e.clipboardData || window.clipboardData).getData('text');
        if (!pasted || pasted.length < pasteToFileLength) return;
        // Desktop path: clipboard data available — handle here, suppress input fallback
        e.preventDefault();
        const before = this.value;
        _wrapTextAsFile(pasted);
        this.value = before;
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });

    // Enter = send, Shift+Enter = new line
    textarea.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        // On mobile/touch — Enter creates a new line, send via button
        if (window.matchMedia("(pointer: coarse)").matches) return;
        e.preventDefault();
        send();
    }
    });

    // --- ADDED THIS FOR AUTO-EXPANDING MEMORY INPUT ---
    const memoryInput = document.getElementById("memoryInput");
    if (memoryInput) {
        memoryInput.addEventListener("input", function () {
            this.style.height = "auto";
            this.style.height = (this.scrollHeight) + "px";
            const words = this.value.trim() === '' ? 0 : this.value.trim().split(/\s+/).length;
            document.getElementById('memoryWordCount').textContent = words.toLocaleString() + ' words';
        });
    }
    // ----------------------------------------------------
    
    setupSlider('summaryLengthSlider', 'summaryLengthValue');
    
    setupSlider('chatStreamTemp', 'chatStreamTempValue', true);
    setupSlider('summarizationTemp', 'summarizationTempValue', true);
    
    setupSlider('scrapingWordLimit', 'scrapingWordLimitValue');
    setupSlider('searchMaxResults', 'searchMaxResultsValue');
    
    setupSlider('charDelay', 'charDelayValue');
    setupSlider('punctuationDelay', 'punctuationDelayValue');
    setupSlider('commaDelay', 'commaDelayValue');
    
    setupSlider('topP', 'topPValue', true);
    setupSlider('topK', 'topKValue');
    setupSlider('minP', 'minPValue', true);
    setupSlider('repetitionPenalty', 'repetitionPenaltyValue', true);
    setupSlider('dryMultiplier', 'dryMultiplierValue', true);
    setupSlider('dryBase', 'dryBaseValue', true);
    setupSlider('dryAllowedLength', 'dryAllowedLengthValue');

    setupSlider('recalledMessageCharLimit', 'recalledMessageCharLimitValue');

    // Bidirectional slider+number pairs
    setupSyncedSlider('maxChatMessages',    'maxChatMessagesNum');
    setupSyncedSlider('llmMaxTokens',       'llmMaxTokensNum');
    
    document.getElementById("editMemoryTextarea").addEventListener("input", updateMemoryStats);
});

// =============================================================================
// --- CHUNKING DLC ---
// =============================================================================

let _chunkUnit = 'tokens'; // local state, synced with backend on open

function openChunkingModal() {
    document.getElementById('chunkingModal').style.display = 'flex';
    fetch('/get_chunking_settings')
        .then(r => r.json())
        .then(s => {
            const enabled = s.chunking_enabled || false;
            document.getElementById('chunkingEnabledToggle').checked = enabled;

            document.getElementById('chunkIndexSystemToggle').checked =
                s.chunk_index_system_immediately !== false;

            // Chunk unit pill
            _chunkUnit = s.chunk_unit || 'tokens';
            _applyChunkUnitUI(_chunkUnit);

            // Chunk limit input
            const limit = s.chunk_token_limit || 400;
            document.getElementById('chunkTokenLimit').value = limit;
            _updateWordEstimate(limit);

            // Overlap input
            const overlap = s.chunk_overlap_tokens ?? 100;
            document.getElementById('chunkOverlapTokens').value = overlap;

            // Retrieval mode
            document.getElementById('chunkRetrievalMode').value =
                s.chunk_retrieval_mode || 'precision';
            updateLongContextVisibility();

            // Long context sub-options
            document.getElementById('chunkPinnedReassembleToggle').checked =
                s.chunk_pinned_always_reassemble !== false;

            document.getElementById('chunkCodeAwareToggle').checked =
                s.chunk_code_aware === true;

            // Apply DLC body enabled/disabled state
            updateChunkingDLCState();
        })
        .catch(err => {
            console.error('Error loading chunking settings:', err);
            showToast('Could not load chunking settings from server.', 'error');
        });
}

function closeChunkingModal() {
    document.getElementById('chunkingModal').style.display = 'none';
}

function updateChunkingDLCState() {
    const enabled = document.getElementById('chunkingEnabledToggle').checked;
    const body = document.getElementById('chunkingDLCBody');
    body.style.opacity = enabled ? '1' : '0.4';
    body.style.pointerEvents = enabled ? 'auto' : 'none';
}

function updateLongContextVisibility() {
    const mode = document.getElementById('chunkRetrievalMode').value;
    document.getElementById('longContextOptions').style.display =
        mode === 'long_context' ? 'block' : 'none';
}

function setChunkUnit(unit) {
    _chunkUnit = unit;
    _applyChunkUnitUI(unit);
    // Re-run estimate with current slider value
    _updateWordEstimate(parseInt(document.getElementById('chunkTokenLimit').value));
}

function _applyChunkUnitUI(unit) {
    const tokBtn  = document.getElementById('chunkUnitTokensBtn');
    const wrdBtn  = document.getElementById('chunkUnitWordsBtn');
    const unitLbl = document.getElementById('chunkUnitLabel');
    if (!tokBtn || !wrdBtn) return;
    tokBtn.classList.toggle('active-unit', unit === 'tokens');
    wrdBtn.classList.toggle('active-unit', unit === 'words');
    if (unitLbl) unitLbl.textContent = unit;
}

function onChunkLimitChange(val) {
    _updateWordEstimate(parseInt(val) || 0);
}

function nudgeInput(id, delta, min, max, onchangeFn) {
    const el = document.getElementById(id);
    if (!el) return;
    let val = (parseInt(el.value) || 0) + delta;
    val = Math.max(min, Math.min(max, val));
    el.value = val;
    // Fire the oninput callback if specified
    if (onchangeFn && window[onchangeFn]) window[onchangeFn](val);
}

function _updateWordEstimate(tokenCount) {
    const est = document.getElementById('chunkWordEstimate');
    if (!est) return;
    // Rough conversion: 1 token ≈ 0.77 words (tiktoken o200k_base average)
    const words = Math.round(tokenCount * 0.77);
    if (_chunkUnit === 'tokens') {
        est.textContent = `≈ ${words} words per chunk`;
    } else {
        // words mode: show reverse — token estimate
        const toks = Math.round(tokenCount / 0.77);
        est.textContent = `≈ ${toks} tokens per chunk`;
    }
}

async function saveChunkingSettings() {
    const saveBtn = document.getElementById('chunkingSaveBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    const payload = {
        chunking_enabled:               document.getElementById('chunkingEnabledToggle').checked,
        chunk_index_system_immediately: document.getElementById('chunkIndexSystemToggle').checked,
        chunk_token_limit:              parseInt(document.getElementById('chunkTokenLimit').value),
        chunk_overlap_tokens:           parseInt(document.getElementById('chunkOverlapTokens').value),
        chunk_unit:                     _chunkUnit,
        chunk_retrieval_mode:           document.getElementById('chunkRetrievalMode').value,
        chunk_pinned_always_reassemble: document.getElementById('chunkPinnedReassembleToggle').checked,
        chunk_code_aware:               document.getElementById('chunkCodeAwareToggle').checked,
    };

    try {
        const res    = await fetch('/set_chunking_settings', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload)
        });
        const result = await res.json();
        if (res.ok) {
            saveBtn.textContent = 'Saved!';
            setTimeout(() => {
                saveBtn.disabled    = false;
                saveBtn.textContent = 'Save Settings';
                closeChunkingModal();
            showToast('Chunking settings saved', 'success');
            }, 1200);
        } else {
            throw new Error(result.error || 'Unknown error');
        }
    } catch (err) {
        console.error('Error saving chunking settings:', err);
        showToast('Error saving chunking settings: ' + err.message, 'error');
        saveBtn.disabled    = false;
        saveBtn.textContent = 'Save Settings';
    }
}
// =============================================================================
// --- END CHUNKING DLC ---
// =============================================================================
// =============================================================================
// --- AVATAR SHAPE & FOCUS ---
// =============================================================================

// AVATAR_SHAPE_CLASSES is defined at the top of the file (near global vars).

/** Apply shape class to every matching avatar in the chatlog */
function _applyAvatarShapeToDOM(who, shape) {
    // who = 'assistant' | 'user'
    const cls = who === 'assistant' ? 'grok' : 'user';
    const bubbles = document.querySelectorAll(`.chat-bubble.${cls} .avatar`);
    bubbles.forEach(el => {
        AVATAR_SHAPE_CLASSES.forEach(c => el.classList.remove(c));
        el.classList.add(`avatar-${shape}`);
    });
}

/** Immediately save shape to server and apply live */
function _persistAvatarStyle(who, shape) {
    const payload = who === 'assistant'
        ? { assistantAvatarShape: shape }
        : { userAvatarShape: shape };
    fetch('/set_appearance_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).catch(e => console.warn('Could not save avatar shape:', e));
    _applyAvatarShapeToDOM(who, shape);
}

/** Called by shape pill buttons */
function setAvatarShape(who, shape) {
    const pillsId = who === 'assistant' ? 'assistantShapePills' : 'userShapePills';
    const pills = document.querySelectorAll(`#${pillsId} .shape-pill`);
    pills.forEach(p => p.classList.toggle('active-shape', p.dataset.shape === shape));
    _persistAvatarStyle(who, shape);
}

/** Populate the shape pills from saved settings when modal opens */
function _loadAvatarStyleUI(who, saved) {
    const shape = (who === 'assistant' ? saved.assistantAvatarShape : saved.userAvatarShape) || 'circle';
    const pillsId = who === 'assistant' ? 'assistantShapePills' : 'userShapePills';
    document.querySelectorAll(`#${pillsId} .shape-pill`).forEach(p => {
        p.classList.toggle('active-shape', p.dataset.shape === shape);
    });
}

// =============================================================================
// --- END AVATAR SHAPE ---
// =============================================================================

// =============================================================================
// --- INITIALIZATION ON PAGE LOAD ---
// =============================================================================
(function initializeSearchTokens() {
    // Fetch search tool tokens from backend on page load
    fetch('/get_search_settings')
        .then(response => response.json())
        .then(settings => {
            searchToolTrigger = settings.search_tool_trigger || '<tool_search>';  // JS-BUG-1 FIX
            searchToolCloser = settings.search_tool_closer || '</tool_search>';  // FIX BUG 10: was ']' (copy-paste from recallToolCloser)
            searchResultHeader = settings.search_result_header || '[Search Results]:';
            recallResultHeader = settings.recall_result_header || '[Recall Results]:';
            recallToolTrigger = settings.recall_tool_trigger || '[RECALL:';
            recallToolCloser  = settings.recall_tool_closer  || ']';
            console.log('Search tool tokens loaded:', {searchToolTrigger, searchToolCloser, searchResultHeader, recallResultHeader});
        })
        .catch(error => {
            console.warn('Could not load search tool tokens, using defaults:', error);
        });
})();


// ============================================================================
// RERANKER SETTINGS FUNCTIONS
// ============================================================================

// High-context reranker models with their max token limits
const RERANKER_HIGH_CTX_MODELS = {
    "bge-reranker-v2-m3": 8192,
    "jina-reranker-v2": 32768
};

function updateRerankerCtxSliderVisibility(modelChoice) {
    /**
     * Shows/hides the context length slider based on model choice.
     * Only high-context models (8K/32K) get the slider.
     */
    const ctxControl = document.getElementById('rerankerCtxControl');
    const slider = document.getElementById('rerankerCtxSlider');
    const label = document.getElementById('rerankerCtxValue');

    const maxCtx = RERANKER_HIGH_CTX_MODELS[modelChoice];

    if (maxCtx) {
        // High-context model - show slider
        ctxControl.style.display = 'block';
        slider.max = maxCtx;
        
        // Set slider to 2K default if current value exceeds new max
        const currentVal = parseInt(slider.value);
        if (currentVal > maxCtx) {
            slider.value = Math.min(2048, maxCtx);
        }
        label.textContent = slider.value;
    } else {
        // Fixed 512 context - hide slider
        ctxControl.style.display = 'none';
    }
}

function updateRerankerUIState() {
    /**
     * Grays out/enables reranker settings based on master toggle.
     * Mirrors the chunking DLC pattern.
     */
    const enabled = document.getElementById('rerankerEnabledToggle').checked;
    const body = document.getElementById('rerankerSettingsBody');
    
    if (enabled) {
        body.style.opacity = '1';
        body.style.pointerEvents = 'auto';
    } else {
        body.style.opacity = '0.4';
        body.style.pointerEvents = 'none';
    }
    
    // Update pipeline status display (affects Stage 3 buffer visibility)
    if (typeof updatePipelineStatusDisplay === 'function') {
        updatePipelineStatusDisplay();
    }
}

function loadRerankerSettings() {
    /**
     * Loads reranker settings from backend when modal opens.
     * Returns the fetch promise so callers can await both loaders before
     * calling updatePipelineStatusDisplay() — avoids a race with loadSanityCheckSettings().
     */
    return fetch('/get_reranker_settings')
        .then(response => response.json())
        .then(settings => {
            // Master toggle
            document.getElementById('rerankerEnabledToggle').checked = settings.reranker_enabled || false;
            
            // Model selection
            document.getElementById('rerankerModelSelect').value = settings.reranker_model || 'ms-marco-mini-v2';
            
            // Context length slider (conditional)
            const ctxSlider = document.getElementById('rerankerCtxSlider');
            const ctxValue = document.getElementById('rerankerCtxValue');
            if (settings.reranker_ctx_length) {
                ctxSlider.value = settings.reranker_ctx_length;
                ctxValue.textContent = settings.reranker_ctx_length;
            }
            
            // Update slider visibility based on model
            updateRerankerCtxSliderVisibility(settings.reranker_model || 'ms-marco-mini-v2');
            
            // Expansion factor
            document.getElementById('rerankerExpansionFactor').value = settings.reranker_expansion_factor || 3;
            
            // Score threshold
            document.getElementById('rerankerScoreThreshold').value = settings.reranker_score_threshold || 0.0;
            document.getElementById('rerankerScoreThresholdValue').textContent = 
                parseFloat(settings.reranker_score_threshold || 0.0).toFixed(1);
            
            // Batch size
            document.getElementById('rerankerBatchSize').value = settings.reranker_batch_size || 32;
            
            // Update UI state (gray out if disabled)
            updateRerankerUIState();
        })
        .catch(error => {
            console.error("Error loading reranker settings:", error);
            // Don't alert - this is called automatically and shouldn't interrupt
            console.warn("Could not load reranker settings, using defaults.");
        });
}

async function saveRerankerSettings() {
    /**
     * Saves reranker settings to backend.
     * Returns true on success, false on failure.
     * FIX: null-guard all getElementById calls so a missing/hidden element
     * never throws a TypeError and silently aborts the save.
     */
    const expansionRaw = parseInt(document.getElementById('rerankerExpansionFactor')?.value);
    const ctxRaw       = parseInt(document.getElementById('rerankerCtxSlider')?.value);
    const threshRaw    = parseFloat(document.getElementById('rerankerScoreThreshold')?.value);
    const batchRaw     = parseInt(document.getElementById('rerankerBatchSize')?.value);

    const settings = {
        reranker_enabled:          document.getElementById('rerankerEnabledToggle')?.checked ?? false,
        reranker_model:            document.getElementById('rerankerModelSelect')?.value || 'ms-marco-mini-v2',
        reranker_expansion_factor: isNaN(expansionRaw) ? 3    : expansionRaw,
        reranker_ctx_length:       isNaN(ctxRaw)       ? null : ctxRaw,
        reranker_score_threshold:  isNaN(threshRaw)    ? 0.0  : threshRaw,
        reranker_batch_size:       isNaN(batchRaw)     ? 32   : batchRaw,
    };

    try {
        const response = await fetch('/set_reranker_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log("Reranker settings saved:", result.message);
            
            // Show user feedback for model reloads
            if (result.model_reloaded) {
                console.log(`🔄 ${result.message}`);
            }
            
            return true;
        } else {
            console.error("Failed to save reranker settings:", result);
            showToast("Failed to save reranker settings: " + (result.error || "Unknown error"), 'error');
            return false;
        }
    } catch (error) {
        console.error("Error saving reranker settings:", error);
        showToast("Error saving reranker settings: " + error.message, 'error');
        return false;
    }
}

// ============================================================================
// STAGE 3: SANITY CHECK FUNCTIONS
// ============================================================================

function updateZeroshotThresholdState() {
    const enabled = document.getElementById('zeroshotThresholdEnabled')?.checked ?? true;
    const row = document.getElementById('zeroshotThresholdSliderRow');
    if (row) {
        row.style.opacity = enabled ? '1' : '0.4';
        row.style.pointerEvents = enabled ? 'auto' : 'none';
    }
}

function updateZeroshotUIState() {
    const enabled = document.getElementById('zeroshotIntentToggle')?.checked || false;
    const dlcBody     = document.getElementById('zeroshotDlcBody');
    const vanillaBody = document.getElementById('sanityVanillaBody');

    if (dlcBody) {
        dlcBody.style.opacity = enabled ? '1' : '0.4';
        dlcBody.style.pointerEvents = enabled ? 'auto' : 'none';
    }
    if (vanillaBody) {
        vanillaBody.style.opacity = enabled ? '0.4' : '1';
        vanillaBody.style.pointerEvents = enabled ? 'none' : 'auto';
    }
    updatePipelineStatusDisplay();
}

function updateSanityCheckUIState() {
    const enabled = document.getElementById('sanityCheckEnabledToggle').checked;
    const body = document.getElementById('sanityCheckSettingsBody');
    const contentBody = document.getElementById('sanityContentFilterBody');

    [body, contentBody].forEach(el => {
        if (!el) return;
        el.style.opacity = enabled ? '1' : '0.4';
        el.style.pointerEvents = enabled ? 'auto' : 'none';
    });

    updatePipelineStatusDisplay();
}

function updatePipelineStatusDisplay() {
    const rerankerEnabled = document.getElementById('rerankerEnabledToggle')?.checked || false;
    const sanityEnabled   = document.getElementById('sanityCheckEnabledToggle')?.checked || false;
    const zeroshotEnabled = document.getElementById('zeroshotIntentToggle')?.checked || false;
    const intentEnabled   = document.getElementById('sanityIntentEnabledToggle')?.checked ?? true;
    const statusDiv = document.getElementById('sanityPipelineStatus');

    if (!statusDiv) return;

    // Correct execution order: Pre-Gate → FAISS → Content Filter → Reranker
    let gateLabel = '❌';
    if (sanityEnabled && intentEnabled) {
        gateLabel = zeroshotEnabled ? '✅ (NLI 🧪)' : '✅ (Embedding)';
    }

    const faissLabel    = '✅';
    const contentLabel  = sanityEnabled ? '✅' : '❌';
    const rerankerLabel = rerankerEnabled ? '✅' : '❌';

    statusDiv.textContent =
        `Pre-Gate: Intent ${gateLabel} → FAISS ${faissLabel} → Content Filter ${contentLabel} → Reranker ${rerankerLabel}`;
}

function loadSanityCheckSettings() {
    // Returns the fetch promise so openResonanceSettingsModal can await both
    // loaders via Promise.all before rendering the final pipeline status display.
    return fetch('/get_sanity_check_settings')
        .then(response => response.json())
        .then(settings => {
            document.getElementById('sanityCheckEnabledToggle').checked = settings.sanity_check_enabled || false;
            document.getElementById('sanityIntentEnabledToggle').checked = settings.sanity_check_intent_enabled !== undefined ? settings.sanity_check_intent_enabled : true;
            
            document.getElementById('sanityIntentThreshold').value = settings.sanity_check_intent_threshold || 0.40;
            document.getElementById('sanityIntentThresholdValue').textContent = parseFloat(settings.sanity_check_intent_threshold || 0.40).toFixed(2);
            
            document.getElementById('sanityContentThreshold').value = settings.sanity_check_content_threshold || 0.35;
            document.getElementById('sanityContentThresholdValue').textContent = parseFloat(settings.sanity_check_content_threshold || 0.35).toFixed(2);
            
            document.getElementById('sanityRecallPhrases').value = settings.sanity_check_recall_phrases || "do you remember, what did we talk about, recall our conversation, tell me about, what did I say, when did we discuss, yesterday we, last time, that time when";

            document.getElementById('sanityNegativePhrases').value = settings.sanity_check_negative_phrases || "good morning, good evening, good night, good afternoon, hello there, hey there, hi there, how are you, what's up, how's it going, hey how are you, good to see you, nice to meet you, how have you been";
            
            document.getElementById('sanityBufferMultiplier').value = settings.sanity_check_buffer_multiplier || 2.0;
            document.getElementById('sanityShowScoresToggle').checked = settings.sanity_check_show_scores !== undefined ? settings.sanity_check_show_scores : true;

            // --- Zeroshot DLC ---
            const zsEnabled = settings.zeroshot_intent_enabled || false;
            if (document.getElementById('zeroshotIntentToggle'))
                document.getElementById('zeroshotIntentToggle').checked = zsEnabled;
            if (document.getElementById('zeroshotModelSelect'))
                document.getElementById('zeroshotModelSelect').value = settings.zeroshot_model || 'nli-minilm-l6';
            if (document.getElementById('zeroshotCtxLength'))
                document.getElementById('zeroshotCtxLength').value = settings.zeroshot_ctx_length || 512;

            // Threshold gating toggle
            const zsThreshEnabled = settings.zeroshot_threshold_enabled !== undefined ? settings.zeroshot_threshold_enabled : true;
            if (document.getElementById('zeroshotThresholdEnabled'))
                document.getElementById('zeroshotThresholdEnabled').checked = zsThreshEnabled;

            if (document.getElementById('zeroshotThreshold')) {
                document.getElementById('zeroshotThreshold').value = settings.zeroshot_entailment_threshold || 0.70;
                document.getElementById('zeroshotThresholdValue').textContent = parseFloat(settings.zeroshot_entailment_threshold || 0.70).toFixed(2);
            }

            // Aggregation radio
            const agg = settings.zeroshot_aggregation || 'max';
            const aggRadio = document.querySelector(`input[name="zeroshotAggregation"][value="${agg}"]`);
            if (aggRadio) aggRadio.checked = true;

            if (document.getElementById('zeroshotPremise'))
                document.getElementById('zeroshotPremise').value = settings.zeroshot_premise_example || 'do you remember when we talked about pizza?';

            // Hypotheses — stored as array, display as one-per-line
            if (document.getElementById('zeroshotHypotheses')) {
                const hyps = settings.zeroshot_hypotheses;
                if (Array.isArray(hyps)) {
                    document.getElementById('zeroshotHypotheses').value = hyps.join('\n');
                } else if (typeof hyps === 'string') {
                    document.getElementById('zeroshotHypotheses').value = hyps;
                } else {
                    document.getElementById('zeroshotHypotheses').value =
                        "the user wants to recall a past conversation\nthe user is asking about something discussed before\nthe user wants to remember something from a previous chat";
                }
            }

            updateSanityCheckUIState();
            updateZeroshotUIState();
            updateZeroshotThresholdState();

            // --- Advanced NLI fields ---
            if (document.getElementById('zsScoreFloor')) {
                const floor = settings.zeroshot_score_floor ?? 0.10;
                document.getElementById('zsScoreFloor').value = floor;
                document.getElementById('zsScoreFloorValue').textContent = parseFloat(floor).toFixed(2);
            }
            if (document.getElementById('zsTopN')) {
                document.getElementById('zsTopN').value = settings.zeroshot_top_n_hypotheses ?? 0;
            }
        })
        .catch(error => {
            console.error("Error loading sanity check settings:", error);
        });
}

async function saveSanityCheckSettings() {
    // FIX: Guard all parseFloat/parseInt calls against NaN.
    // NaN serialises as JSON null, and the backend coerces null to wrong types
    // (e.g. float(None) raises TypeError), causing the POST to return 400.
    const intentThresh  = parseFloat(document.getElementById('sanityIntentThreshold')?.value);
    const contentThresh = parseFloat(document.getElementById('sanityContentThreshold')?.value);
    const bufferMult    = parseFloat(document.getElementById('sanityBufferMultiplier')?.value);
    const zsCtx         = parseInt(document.getElementById('zeroshotCtxLength')?.value);
    const zsThreshold   = parseFloat(document.getElementById('zeroshotThreshold')?.value);
    const zsScoreFloor  = parseFloat(document.getElementById('zsScoreFloor')?.value);
    const zsTopN        = parseInt(document.getElementById('zsTopN')?.value);

    const settings = {
        sanity_check_enabled:           document.getElementById('sanityCheckEnabledToggle')?.checked ?? false,
        sanity_check_intent_enabled:    document.getElementById('sanityIntentEnabledToggle')?.checked ?? true,
        sanity_check_intent_threshold:  isNaN(intentThresh)  ? 0.40 : intentThresh,
        sanity_check_content_threshold: isNaN(contentThresh) ? 0.35 : contentThresh,
        sanity_check_recall_phrases:    document.getElementById('sanityRecallPhrases')?.value.trim()   || '',
        sanity_check_negative_phrases:  document.getElementById('sanityNegativePhrases')?.value.trim() || '',
        sanity_check_buffer_multiplier: isNaN(bufferMult)    ? 2.0  : bufferMult,
        sanity_check_show_scores:       document.getElementById('sanityShowScoresToggle')?.checked ?? true,
        // --- Zeroshot DLC ---
        zeroshot_intent_enabled:        document.getElementById('zeroshotIntentToggle')?.checked || false,
        zeroshot_model:                 document.getElementById('zeroshotModelSelect')?.value   || 'nli-minilm-l6',
        zeroshot_ctx_length:            isNaN(zsCtx)        ? 512  : zsCtx,
        zeroshot_threshold_enabled:     document.getElementById('zeroshotThresholdEnabled')?.checked ?? true,
        zeroshot_entailment_threshold:  isNaN(zsThreshold)  ? 0.70 : zsThreshold,
        zeroshot_aggregation:           document.querySelector('input[name="zeroshotAggregation"]:checked')?.value || 'max',
        zeroshot_premise_example:       document.getElementById('zeroshotPremise')?.value.trim()      || '',
        zeroshot_hypotheses:            document.getElementById('zeroshotHypotheses')?.value.trim()   || 'the user wants to recall a past conversation',
        // --- Advanced NLI ---
        zeroshot_score_floor:           isNaN(zsScoreFloor) ? 0.10 : zsScoreFloor,
        zeroshot_top_n_hypotheses:      isNaN(zsTopN)       ? 0    : zsTopN,
    };

    try {
        const response = await fetch('/set_sanity_check_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log("Sanity check settings saved:", result.message);
            if (result.phrases_reloaded) {
                console.log("🔄 Intent phrases re-embedded.");
            }
            return true;
        } else {
            console.error("Failed to save sanity check settings:", result);
            showToast("Failed to save sanity check settings: " + (result.error || "Unknown error"), 'error');
            return false;
        }
    } catch (error) {
        console.error("Error saving sanity check settings:", error);
        showToast("Error saving sanity check settings: " + error.message, 'error');
        return false;
    }
}

// ============================================================================
// AVATAR CROPPER SYSTEM
// ============================================================================

let avatarCropperState = {
    image: null,
    canvas: null,
    ctx: null,
    isDragging: false,
    startX: 0,
    startY: 0,
    offsetX: 0,
    offsetY: 0,
    zoom: 100,
    shape: 'circle',
    avatarType: null, // 'persona' or 'user'
    persona_name: null,
    loadout_name: null
};

function openAvatarCropper(avatarType, personaName = null, loadoutName = null) {
    const modal = document.getElementById('avatarCropperModal');
    if (!modal) {
        console.error('Avatar cropper modal not found in DOM');
        return;
    }

    avatarCropperState.avatarType = avatarType;
    avatarCropperState.persona_name = personaName;
    avatarCropperState.loadout_name = loadoutName;

    // Reset to defaults each open
    avatarCropperState.zoom = 100;
    avatarCropperState.offsetX = 0;
    avatarCropperState.offsetY = 0;
    avatarCropperState.shape = 'circle';

    modal.style.display = 'flex';

    // Reset shape button states
    document.querySelectorAll('.avatar-cropper-shape-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.shape === 'circle');
    });

    // Reset zoom slider
    const zoomSlider = document.getElementById('avatarCropperZoomSlider');
    if (zoomSlider) {
        zoomSlider.value = 100;
        const zoomVal = document.getElementById('avatarCropperZoomValue');
        if (zoomVal) zoomVal.textContent = '100%';
    }
}

async function reopenPersonaAvatarCropper() {
    try {
        const res = await fetch(`/get_persona_avatar_info/${encodeURIComponent(currentActivePersonaName)}`);
        if (!res.ok) throw new Error('Failed to fetch persona avatar info');
        const data = await res.json();
        if (!data.avatar_url) { showToast('No image avatar to reposition.', 'info'); return; }
        const imgRes = await fetch(data.avatar_url + '?t=' + Date.now());
        const blob = await imgRes.blob();
        const reader = new FileReader();
        reader.onload = (ev) => {
            openAvatarCropper('persona', currentActivePersonaName);
            initAvatarCropper(ev.target.result);
        };
        reader.readAsDataURL(blob);
    } catch(e) { showToast('Could not open cropper: ' + e.message, 'error'); }
}

async function reopenUserAvatarCropper() {
    const select = document.getElementById('userLoadoutSelect');
    const loadoutName = select ? select.value : null;
    if (!loadoutName) { showToast('No loadout selected.', 'info'); return; }
    try {
        const imgRes = await fetch(`/get_user_loadout_avatar/${encodeURIComponent(loadoutName)}?t=${Date.now()}`);
        if (!imgRes.ok) { showToast('No image avatar to reposition.', 'info'); return; }
        const blob = await imgRes.blob();
        const reader = new FileReader();
        reader.onload = (ev) => {
            openAvatarCropper('user', null, loadoutName);
            initAvatarCropper(ev.target.result);
        };
        reader.readAsDataURL(blob);
    } catch(e) { showToast('Could not open cropper: ' + e.message, 'error'); }
}

// ============================================================
// --- GREETING PICKER (Character Cards) ---
// ============================================================

function showGreetingPicker(greetings, personaName) {
    const modal = document.getElementById('greetingPickerModal');
    const list  = document.getElementById('greetingPickerList');
    const input = document.getElementById('greetingPickerInput');

    list.innerHTML = '';
    greetings.forEach((preview, i) => {
        const item = document.createElement('div');
        item.className = 'greeting-picker-item';
        item.innerHTML = `<span class="greeting-picker-num">${i})</span> ${preview}`;
        item.addEventListener('click', () => {
            input.value = i;
            document.querySelectorAll('.greeting-picker-item')
                    .forEach(el => el.classList.remove('selected'));
            item.classList.add('selected');
        });
        list.appendChild(item);
    });

    input.value = '0';
    input.max   = greetings.length - 1;
    modal.dataset.personaName = personaName;
    modal.style.display = 'flex';
}

function closeGreetingPickerModal() {
    document.getElementById('greetingPickerModal').style.display = 'none';
}

async function confirmGreetingPick() {
    const modal       = document.getElementById('greetingPickerModal');
    const personaName = modal.dataset.personaName;
    const idx         = parseInt(document.getElementById('greetingPickerInput').value) || 0;

    closeGreetingPickerModal();
    showBusyOverlay('Injecting greeting...', false);

    try {
        const res  = await fetch('/inject_greeting', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ persona_name: personaName, greeting_index: idx })
        });
        const data = await res.json();
        hideBusyOverlay();

        if (!res.ok) {
            showToast('Could not inject greeting: ' + (data.error || 'Unknown error'), 'error');
            return;
        }

        showToast(`💬 Greeting ${idx} loaded!`, 'success', 3000);
        await reloadChat();
        await updateAllStatusUI();
    } catch (err) {
        hideBusyOverlay();
        showToast('Network error injecting greeting.', 'error');
        console.error('Greeting inject error:', err);
    }
}

function closeAvatarCropper() {
    const modal = document.getElementById('avatarCropperModal');
    if (modal) modal.style.display = 'none';
    const container = document.getElementById('avatarCropperContainer');
    if (container) container.style.display = 'none';
    const uploadPrompt = document.querySelector('#avatarCropperModal [onclick*="avatarCropperFileInput"]');
    if (uploadPrompt) uploadPrompt.style.display = '';
    avatarCropperState = {
        image: null, canvas: null, ctx: null, isDragging: false,
        startX: 0, startY: 0, offsetX: 0, offsetY: 0,
        zoom: 100, shape: 'circle', avatarType: null,
        persona_name: null, loadout_name: null
    };
}

function initAvatarCropper(imageData) {
    const container = document.getElementById('avatarCropperContainer');
    const canvas = document.getElementById('avatarCropperCanvas');

    if (!container || !canvas) {
        console.error('Cropper elements not found in DOM');
        return;
    }

    container.style.display = 'block';
    const uploadPrompt = document.querySelector('#avatarCropperModal [onclick*="avatarCropperFileInput"]');
    if (uploadPrompt) uploadPrompt.style.display = 'none';

    avatarCropperState.canvas = canvas;
    avatarCropperState.ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = function() {
        avatarCropperState.image = img;
        canvas.width = 350;
        canvas.height = 350;
        avatarCropperState.offsetX = 0;
        avatarCropperState.offsetY = 0;
        updateAvatarCropperShapeOverlay();
        redrawAvatarCanvas();
        setupAvatarCropperEventListeners();
    };
    img.src = imageData;
}

function setupAvatarCropperEventListeners() {
    const canvas = avatarCropperState.canvas;
    const zoomSlider = document.getElementById('avatarCropperZoomSlider');

    if (!canvas) return;

    // Drag
    canvas.addEventListener('mousedown', (e) => {
        avatarCropperState.isDragging = true;
        const rect = canvas.getBoundingClientRect();
        avatarCropperState.startX = e.clientX - rect.left - avatarCropperState.offsetX;
        avatarCropperState.startY = e.clientY - rect.top - avatarCropperState.offsetY;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!avatarCropperState.isDragging) return;
        const rect = canvas.getBoundingClientRect();
        avatarCropperState.offsetX = e.clientX - rect.left - avatarCropperState.startX;
        avatarCropperState.offsetY = e.clientY - rect.top - avatarCropperState.startY;
        redrawAvatarCanvas();
    });

    canvas.addEventListener('mouseup', () => { avatarCropperState.isDragging = false; });
    canvas.addEventListener('mouseleave', () => { avatarCropperState.isDragging = false; });

    // Touch support
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        avatarCropperState.isDragging = true;
        const rect = canvas.getBoundingClientRect();
        const t = e.touches[0];
        avatarCropperState.startX = t.clientX - rect.left - avatarCropperState.offsetX;
        avatarCropperState.startY = t.clientY - rect.top - avatarCropperState.offsetY;
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!avatarCropperState.isDragging) return;
        const rect = canvas.getBoundingClientRect();
        const t = e.touches[0];
        avatarCropperState.offsetX = t.clientX - rect.left - avatarCropperState.startX;
        avatarCropperState.offsetY = t.clientY - rect.top - avatarCropperState.startY;
        redrawAvatarCanvas();
    }, { passive: false });

    canvas.addEventListener('touchend', () => { avatarCropperState.isDragging = false; });

    // Wheel zoom
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -10 : 10;
        avatarCropperState.zoom = Math.max(100, Math.min(300, avatarCropperState.zoom + delta));
        if (zoomSlider) {
            zoomSlider.value = avatarCropperState.zoom;
            const zoomVal = document.getElementById('avatarCropperZoomValue');
            if (zoomVal) zoomVal.textContent = avatarCropperState.zoom + '%';
        }
        redrawAvatarCanvas();
    });

    // Zoom slider
    if (zoomSlider) {
        zoomSlider.addEventListener('input', () => {
            avatarCropperState.zoom = parseInt(zoomSlider.value);
            const zoomVal = document.getElementById('avatarCropperZoomValue');
            if (zoomVal) zoomVal.textContent = avatarCropperState.zoom + '%';
            redrawAvatarCanvas();
        });
    }
}

function redrawAvatarCanvas() {
    const { canvas, ctx, image, zoom, offsetX, offsetY } = avatarCropperState;
    if (!image) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scale = zoom / 100;
    const canvasRatio = canvas.width / canvas.height;
    const imgRatio = image.width / image.height;

    let drawWidth, drawHeight;
    if (imgRatio > canvasRatio) {
        drawHeight = canvas.height * scale;
        drawWidth = drawHeight * imgRatio;
    } else {
        drawWidth = canvas.width * scale;
        drawHeight = drawWidth / imgRatio;
    }

    const centerX = (canvas.width - drawWidth) / 2;
    const centerY = (canvas.height - drawHeight) / 2;

    ctx.drawImage(image, centerX + offsetX, centerY + offsetY, drawWidth, drawHeight);
}

function updateAvatarCropperShapeOverlay() {
    const overlay = document.getElementById('avatarCropperShapeOverlay');
    if (overlay) {
        overlay.className = 'avatar-crop-shape-overlay avatar-crop-overlay-' + avatarCropperState.shape;
    }
}

function setAvatarCropperShape(shape) {
    avatarCropperState.shape = shape;
    updateAvatarCropperShapeOverlay();
    document.querySelectorAll('.avatar-cropper-shape-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.shape === shape);
    });
    // Persist shape to server based on which avatar type is being cropped
    if (avatarCropperState.avatarType) {
        const key = avatarCropperState.avatarType === 'persona' ? 'assistantAvatarShape' : 'userAvatarShape';
        fetch('/set_appearance_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: shape })
        }).catch(e => console.warn('Could not save cropper shape:', e));
    }
}

async function saveAvatarCrop() {
    const { avatarType, persona_name, loadout_name, canvas, shape } = avatarCropperState;

    if (!avatarType) { showToast('Error: No avatar type set.', 'error'); return; }
    if (!canvas) { showToast('Error: No image loaded into cropper.', 'error'); return; }

    try {
        const croppedImageData = canvas.toDataURL('image/png');

        let uploadEndpoint = '';
        let uploadPayload = { image: croppedImageData };

        if (avatarType === 'persona') {
            uploadEndpoint = '/upload_persona_avatar';
            uploadPayload.persona_name = persona_name;
        } else if (avatarType === 'user') {
            uploadEndpoint = '/upload_user_loadout_avatar';
            uploadPayload.loadout_name = loadout_name;
        } else {
            showToast('Error: Invalid avatar type.', 'error'); return;
        }

        const uploadRes = await fetch(uploadEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(uploadPayload)
        });

        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            showToast('Failed to save cropped image: ' + (err.error || 'Unknown error'), 'error');
            return;
        }

        console.log('Cropped avatar saved successfully.');
        closeAvatarCropper();
        showToast('Avatar saved ✓', 'success');  // BUG 3 FIX: was silently closing with no feedback

        if (avatarType === 'persona') {
            await _updatePersonaAvatarPreview();
            await fetchAndApplyPersonaAvatar(persona_name);
        } else if (avatarType === 'user') {
            const loRes = await fetch('/get_user_loadouts');
            if (!loRes.ok) throw new Error(`/get_user_loadouts returned ${loRes.status}`);
            const loData = await loRes.json();
            await _updateUserLoadoutAvatarPreview(loadout_name, loData.loadouts[loadout_name] || {});  // BUG 5 FIX: was missing await — reloadChat raced ahead before preview refreshed
            await applyActiveUserLoadoutAvatar();
        }
        await reloadChat();

    } catch (err) {
        console.error('Error saving crop:', err);
        showToast('Error saving crop: ' + err.message, 'error');
    }
}

// ============================================================================
// END AVATAR CROPPER SYSTEM
// ============================================================================

// ============================================================================
// ASSISTANT NAME DISPLAY  (read-only — driven by active persona)
// The assistantNameInput field is purely a display field. The assistant name
// is always derived from the active persona on the backend via get_system_prompt().
// Changing it requires editing the persona itself in the Persona Settings panel.
// saveAssistantName() was removed — it was dead code (input is readonly, function
// was never called, and NAME_SETTINGS["assistant_name"] is always overridden by
// the persona system anyway).
// ============================================================================

// ============================================================================
// MEMORY SEARCH TEST  (/api/search_memory)
// ============================================================================
async function runMemorySearch() {
    const query = document.getElementById('memorySearchInput')?.value.trim();
    const topK  = parseInt(document.getElementById('memorySearchTopK')?.value || 5);
    const resultsDiv = document.getElementById('memorySearchResults');
    if (!resultsDiv) return;

    if (!query) {
        resultsDiv.style.display = 'block';
        resultsDiv.innerHTML = '<p style="color:#888; font-size:0.85em;">Enter a query first.</p>';
        return;
    }

    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<p style="color:#888; font-size:0.85em;">🔎 Searching...</p>';

    try {
        const res = await fetch('/api/search_memory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK })
        });
        const data = await res.json();

        if (!res.ok || data.error) {
            // FIX XSS: data.error comes from the server and may contain HTML chars.
            // Use _escHtml() before injecting — same pattern as the toast fix (FIX #11).
            resultsDiv.innerHTML = `<p style="color:#f55; font-size:0.85em;">Error: ${_escHtml(data.error || 'Request failed')}</p>`;
            return;
        }

        const results = data.results || [];
        if (results.length === 0) {
            resultsDiv.innerHTML = '<p style="color:#888; font-size:0.85em;">No results found.</p>';
            return;
        }

        resultsDiv.innerHTML = results.map((r, i) => {
            const score      = (r.score        ?? r.faiss_score ?? 0).toFixed(4);
            const reranker   = r.reranker_score != null ? ` · reranker <strong>${r.reranker_score.toFixed(4)}</strong>` : '';
            const sanity     = r.sanity_score   != null ? ` · sanity <strong>${r.sanity_score.toFixed(4)}</strong>`   : '';
            const roleColor  = r.role === 'user' ? '#7eb8f7' : '#a8e6a3';
            // JS-BUG-7 FIX: escape memory content before injecting into innerHTML
            const _rawContent = r.content.length > 200 ? r.content.slice(0, 200) + '\u2026' : r.content;
            const content = _rawContent.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            return `
                <div style="background:#111827; border:1px solid #2a2a3e; border-radius:6px; padding:8px 10px; margin-bottom:6px; font-size:0.82em;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:${roleColor}; font-weight:bold; text-transform:uppercase; font-size:0.8em;">${r.role}</span>
                        <span style="color:#666; font-size:0.78em;">score <strong style="color:#ccc;">${score}</strong>${reranker}${sanity}</span>
                    </div>
                    <div style="color:#ccc; line-height:1.4;">${content}</div>
                </div>`;
        }).join('');
    } catch (e) {
        resultsDiv.innerHTML = `<p style="color:#f55; font-size:0.85em;">Error: ${e.message}</p>`;
    }
}


// =============================================================================
// --- ZEROSHOT ADVANCED MODAL ---
// =============================================================================

function openZeroshotAdvancedModal() {
    // Load presets from backend first, then open
    fetch('/api/zeroshot_preset')
        .then(r => r.json())
        .then(data => {
            _renderZsPresetButtons(data.presets, data.current_model);
            _updateZsPreview();
            document.getElementById('zeroshotAdvancedModal').style.display = 'flex';
        })
        .catch(err => {
            console.error('Failed to load zeroshot presets:', err);
            // Still open modal — presets section will just be empty
            document.getElementById('zeroshotAdvancedModal').style.display = 'flex';
        });
}

function closeZeroshotAdvancedModal() {
    document.getElementById('zeroshotAdvancedModal').style.display = 'none';
}

function _renderZsPresetButtons(presets, currentModel) {
    const container = document.getElementById('zsPresetButtons');
    if (!container) return;
    container.innerHTML = '';
    for (const [modelName, preset] of Object.entries(presets)) {
        const isActive = modelName === currentModel;
        const btn = document.createElement('button');
        btn.textContent = modelName + (isActive ? ' ✓' : '');
        btn.title = `threshold: ${preset.threshold} | agg: ${preset.aggregation} | floor: ${preset.score_floor}`;
        btn.style.cssText = `
            padding: 5px 12px; border-radius: 20px; font-size: 0.80em; cursor: pointer;
            border: 1px solid ${isActive ? '#7ec87e' : '#2a3a2a'};
            color: ${isActive ? '#7ec87e' : '#4a7a4a'};
            background: ${isActive ? '#0d1f0d' : '#0a0f0a'};
            transition: all 0.15s;
        `;
        btn.onmouseover = () => { btn.style.borderColor = '#7ec87e'; btn.style.color = '#7ec87e'; };
        btn.onmouseout  = () => { btn.style.borderColor = isActive ? '#7ec87e' : '#2a3a2a'; btn.style.color = isActive ? '#7ec87e' : '#4a7a4a'; };
        btn.onclick = () => _applyZsPreset(modelName, preset);
        container.appendChild(btn);
    }
}

function _applyZsPreset(modelName, preset) {
    // Fill the advanced fields with preset values
    if (document.getElementById('zsScoreFloor')) {
        document.getElementById('zsScoreFloor').value = preset.score_floor;
        document.getElementById('zsScoreFloorValue').textContent = parseFloat(preset.score_floor).toFixed(2);
    }
    // Also update the main threshold + aggregation in the parent modal
    if (document.getElementById('zeroshotThreshold')) {
        document.getElementById('zeroshotThreshold').value = preset.threshold;
        document.getElementById('zeroshotThresholdValue').textContent = parseFloat(preset.threshold).toFixed(2);
    }
    const aggRadio = document.querySelector(`input[name="zeroshotAggregation"][value="${preset.aggregation}"]`);
    if (aggRadio) aggRadio.checked = true;

    _updateZsPreview();
    showToast(
        `Preset '${modelName}' loaded — threshold: ${preset.threshold}, agg: ${preset.aggregation}, floor: ${preset.score_floor}. Hit Save to apply 🎯`,
        'success', 4000
    );
}

function _updateZsPreview() {
    const preview = document.getElementById('zsAdvancedPreview');
    if (!preview) return;
    const floor  = parseFloat(document.getElementById('zsScoreFloor')?.value ?? 0.10).toFixed(2);
    const topN   = document.getElementById('zsTopN')?.value ?? 0;
    const thresh = parseFloat(document.getElementById('zeroshotThreshold')?.value ?? 0.70).toFixed(2);
    const agg    = document.querySelector('input[name="zeroshotAggregation"]:checked')?.value ?? 'max';

    preview.textContent = '';

    const line1 = document.createElement('span');
    const t1 = document.createTextNode('threshold: ');
    const s1 = document.createElement('span');
    s1.style.color = '#7ec87e';
    s1.textContent = thresh;
    const sep1 = document.createTextNode(' \u00a0|\u00a0 aggregation: ');
    const s2 = document.createElement('span');
    s2.style.color = '#7ec87e';
    s2.textContent = agg;
    line1.appendChild(t1); line1.appendChild(s1); line1.appendChild(sep1); line1.appendChild(s2);

    const br = document.createElement('br');

    const line2 = document.createElement('span');
    const t2 = document.createTextNode('score_floor: ');
    const s3 = document.createElement('span');
    s3.style.color = '#7ec87e';
    s3.textContent = floor;
    const sep2 = document.createTextNode(' \u00a0|\u00a0 top_n: ');
    const s4 = document.createElement('span');
    s4.style.color = '#7ec87e';
    s4.textContent = (topN === '0' || topN === 0) ? 'all' : topN;
    line2.appendChild(t2); line2.appendChild(s3); line2.appendChild(sep2); line2.appendChild(s4);

    preview.appendChild(line1);
    preview.appendChild(br);
    preview.appendChild(line2);
}

async function saveZeroshotAdvanced() {
    // Collect advanced values and merge into a full sanity check settings save.
    // FIX 2+3: Fetch live backend settings first so parent-modal fields (threshold,
    // aggregation, enabled flags) are never stale or missing when the parent modal
    // is closed.  DOM values take priority when present; live backend values are
    // the fallback — ensuring we never overwrite user-set values with hardcoded
    // defaults just because those elements aren't currently in the DOM.
    let live = {};
    try {
        const _lr = await fetch('/get_sanity_check_settings');
        if (_lr.ok) live = await _lr.json();
    } catch (_) { /* non-critical — DOM values / hardcoded defaults used below */ }

    const floor = parseFloat(document.getElementById('zsScoreFloor')?.value ?? live.zeroshot_score_floor ?? 0.10);
    const topN  = parseInt(document.getElementById('zsTopN')?.value ?? live.zeroshot_top_n_hypotheses ?? 0);

    // For parent-modal fields, read from DOM only when the element is actually
    // present and populated; otherwise fall back to the freshly-fetched live value.
    const _threshEl        = document.getElementById('zeroshotThreshold');
    const threshold        = _threshEl ? parseFloat(_threshEl.value) : (live.zeroshot_entailment_threshold ?? 0.70);

    const _aggEl           = document.querySelector('input[name="zeroshotAggregation"]:checked');
    const aggregation      = _aggEl ? _aggEl.value : (live.zeroshot_aggregation ?? 'max');

    const _threshEnEl      = document.getElementById('zeroshotThresholdEnabled');
    const thresholdEnabled = _threshEnEl ? _threshEnEl.checked : (live.zeroshot_threshold_enabled ?? true);

    const _zsEnEl          = document.getElementById('zeroshotIntentToggle');
    const zsEnabled        = _zsEnEl ? _zsEnEl.checked : (live.zeroshot_intent_enabled ?? false);

    const _sanEnEl         = document.getElementById('sanityCheckEnabledToggle');
    const sanityEnabled    = _sanEnEl ? _sanEnEl.checked : (live.sanity_check_enabled ?? false);

    try {
        const response = await fetch('/set_sanity_check_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                // Advanced-only fields
                zeroshot_score_floor: floor,
                zeroshot_top_n_hypotheses: topN,
                // Parent-modal fields included for safety — prevents silent wipe
                zeroshot_entailment_threshold: threshold,
                zeroshot_aggregation: aggregation,
                zeroshot_threshold_enabled: thresholdEnabled,
                zeroshot_intent_enabled: zsEnabled,
                sanity_check_enabled: sanityEnabled,
            })
        });
        const result = await response.json();
        if (response.ok) {
            _updateZsPreview();
            showToast('Advanced NLI settings saved ✓', 'success');
            closeZeroshotAdvancedModal();
        } else {
            showToast('Error: ' + (result.error || 'Save failed'), 'error');
        }
    } catch (err) {
        console.error('saveZeroshotAdvanced error:', err);
        showToast('Save failed — check console', 'error');
    }
}

async function resetZeroshotAdvanced() {
    try {
        const response = await fetch('/api/zeroshot_preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reset: true })
        });
        const result = await response.json();
        if (response.ok) {
            // Apply the reset values back to DOM
            const applied = result.applied;
            if (document.getElementById('zsScoreFloor')) {
                document.getElementById('zsScoreFloor').value = applied.score_floor;
                document.getElementById('zsScoreFloorValue').textContent = parseFloat(applied.score_floor).toFixed(2);
            }
            if (document.getElementById('zsTopN'))
                document.getElementById('zsTopN').value = applied.top_n_hypotheses;
            if (document.getElementById('zeroshotThreshold')) {
                document.getElementById('zeroshotThreshold').value = applied.threshold;
                document.getElementById('zeroshotThresholdValue').textContent = parseFloat(applied.threshold).toFixed(2);
            }
            const aggRadio = document.querySelector(`input[name="zeroshotAggregation"][value="${applied.aggregation}"]`);
            if (aggRadio) aggRadio.checked = true;
            _updateZsPreview();
            showToast('Advanced NLI settings reset to defaults ↺', 'success');
        } else {
            showToast('Reset failed: ' + (result.error || 'unknown'), 'error');
            // Re-fetch live server state so sliders don't lie if the reset was rejected
            openZeroshotAdvancedModal();
        }
    } catch (err) {
        // Network failure — sliders may be out of sync with whatever the server
        // actually has saved. Re-open the modal to re-fetch and resync everything
        // rather than leaving the user staring at sliders that don't reflect reality.
        console.error('resetZeroshotAdvanced error:', err);
        showToast('Reset failed — resyncing from server…', 'error');
        openZeroshotAdvancedModal();
    }
}

// =============================================================================
// --- END ZEROSHOT ADVANCED MODAL ---
// =============================================================================


// =============================================================================
// --- IDLE MODE MODULE v2 (with custom prompt input) ---
// =============================================================================

let _idleEnabled    = false;
let _idleTimer      = null;
let _idleIntervalMs = 8000;
let _idleFiring     = false;
let _msgQueue       = [];
let _idleInputFocused = false;  // true while user is editing the interval or prompt inputs

// Returns true while an idle input owns the caret.
// Flag-only — no document.activeElement here. activeElement is stale during
// pointerdown (fires before focus shifts), so reading it here causes a
// false-positive that permanently blocks the scheduler. The flag is the
// single source of truth; it is cleared by blur, pointerdown, and visualViewport.
function _isIdleInputBusy() {
    return _idleInputFocused;
}

function _getIdlePrompt() {
    const input = document.getElementById('idlePromptInput');
    const val   = (input && input.value.trim()) || '';
    return val || '*continue';
}

// ── Persist to server ─────────────────────────────────────────────────────────
async function _saveIdleSettings() {
    try {
        const secInput    = document.getElementById('idleSecondsInput');
        const promptInput = document.getElementById('idlePromptInput');
        const res = await fetch('/set_idle_settings', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                idleEnabled:         _idleEnabled,
                idleIntervalSeconds: Math.max(1, parseInt(secInput && secInput.value) || 8),
                idlePrompt:          (promptInput && promptInput.value.trim()) || '',
            }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            console.warn('Idle save failed:', err.error || res.status);
            showToast('\u26a0\ufe0f Could not save IDLE settings \u2014 ' + (err.error || 'server error ' + res.status), 'warn', 4000);
        }
    } catch (e) {
        console.warn('Could not save idle settings:', e);
        showToast('\u26a0\ufe0f Could not save IDLE settings (network error)', 'warn', 4000);
    }
}

// ── Load from server — called once on page load ───────────────────────────────
async function _loadIdleSettings() {
    try {
        const res  = await fetch('/get_idle_settings');
        if (!res.ok) return;
        const data = await res.json();

        const secInput    = document.getElementById('idleSecondsInput');
        const promptInput = document.getElementById('idlePromptInput');

        if (secInput && data.idleIntervalSeconds != null) {
            secInput.value = data.idleIntervalSeconds;
        }
        if (promptInput && data.idlePrompt != null) {
            promptInput.value = data.idlePrompt;
        }
        if (data.idleEnabled) {
            // Re-arm silently — update internal state + UI without firing _scheduleIdle yet.
            // We do NOT auto-schedule on load to avoid surprise AI messages. IDLE resumes
            // naturally after the user's first manual send via _onSendComplete().
            // The toast below makes this behaviour explicit so it's never a surprise.
            _idleEnabled = true;
            _idleIntervalMs = Math.max(1, (data.idleIntervalSeconds || 8)) * 1000;
            const btn = document.getElementById('idleToggleBtn');
            if (btn) {
                btn.style.opacity = '1';
                btn.style.color   = '#3b8bd4';
                btn.title         = 'IDLE active — will resume after next send. Click to stop.';
            }
            if (secInput)    secInput.style.display    = 'inline-block';
            if (promptInput) promptInput.style.display = 'inline-block';
            showToast(
                '💤 IDLE restored · every ' + (data.idleIntervalSeconds || 8) + 's · resumes after your next send',
                'info', 5000
            );
        }
    } catch (e) {
        console.warn('Could not load idle settings:', e);
    }
}

// ── Toggle ───────────────────────────────────────────────────────────────────
function toggleIdleMode() {
    _idleEnabled = !_idleEnabled;
    const btn         = document.getElementById('idleToggleBtn');
    const secInput    = document.getElementById('idleSecondsInput');
    const promptInput = document.getElementById('idlePromptInput');

    if (_idleEnabled) {
        _idleIntervalMs = Math.max(1, parseInt(secInput.value) || 8) * 1000;
        btn.style.opacity   = '1';
        btn.style.color     = '#3b8bd4';
        btn.title           = 'IDLE active — click to stop';
        secInput.style.display    = 'inline-block';
        promptInput.style.display = 'inline-block';
        showToast(`💤 IDLE ON · every ${_idleIntervalMs/1000}s · prompt: "${_getIdlePrompt()}"`, 'info', 3500);
        _scheduleIdle();
    } else {
        _cancelIdle();
        btn.style.opacity         = '0.45';
        btn.style.color           = '';
        btn.title                 = 'IDLE mode — let AI keep thinking';
        secInput.style.display    = 'none';
        promptInput.style.display = 'none';
        showToast('💤 IDLE OFF', 'info', 2000);
    }
    _saveIdleSettings(); // persist enabled state + current interval + prompt to server
}

// ── Scheduler ────────────────────────────────────────────────────────────────
function _scheduleIdle() {
    _cancelIdle();
    if (!_idleEnabled) return;
    if (_isIdleInputBusy()) return;  // caret is in an idle input — hold fire
    _idleTimer = setTimeout(_idleTick, _idleIntervalMs);
}

function _cancelIdle() {
    if (_idleTimer) { clearTimeout(_idleTimer); _idleTimer = null; }
}

async function _idleTick() {
    if (!_idleEnabled) return;
    if (_isIdleInputBusy()) { _scheduleIdle(); return; }  // caret still in input — defer
    if (isSending) { _scheduleIdle(); return; }
    await _idleSend();
}

// ── _idleSend — sends idle prompt, no visible user bubble ────────────────────
async function _idleSend() {
    if (isSending) return;

    const idlePrompt = _getIdlePrompt();
    const sendBtn    = document.getElementById('sendBtn');

    _idleFiring = true;
    isSending   = true;
    const myGenId = ++_generationId;

    sendBtn.disabled  = true;
    sendBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"
        viewBox="0 0 24 24" fill="currentColor">
        <rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;
    sendBtn.onclick   = abortGeneration;
    sendBtn.title     = 'Stop generating';
    sendBtn.disabled  = false;

    // Assistant bubble
    const chatlog         = document.getElementById('chatlog');
    const assistantBubble = document.createElement('div');
    assistantBubble.classList.add('chat-bubble', 'grok');
    assistantBubble.innerHTML = `${buildBubbleHeader('grok', assistantName)}
        <div class="message-body">
            <div class="message-content"></div>
            <div class="thinking-indicator"></div>
        </div>`;
    chatlog.appendChild(assistantBubble);

    try {
        const shape    = (_appearanceCache && _appearanceCache.assistantAvatarShape) || 'circle';
        const avatarEl = assistantBubble.querySelector('.avatar');
        if (avatarEl) {
            AVATAR_SHAPE_CLASSES.forEach(c => avatarEl.classList.remove(c));
            avatarEl.classList.add(`avatar-${shape}`);
        }
    } catch(e) {}

    const thinkingIndicator = assistantBubble.querySelector('.thinking-indicator');
    startThinkingAnimation(thinkingIndicator);

    _streamAbortController = new AbortController();
    const signal = _streamAbortController.signal;

    // Send the actual idle prompt text — backend never sees blank anymore
    const fetchPromise = fetch('/chat_stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: idlePrompt }),
        signal
    });

    try {
        await streamAssistantResponse(fetchPromise, assistantBubble, signal);
    } finally {
        _idleFiring = false;
        isSending   = false;
        if (_generationId === myGenId) {
            sendBtn.disabled  = false;
            sendBtn.onclick   = send;
            sendBtn.title     = 'Send';
            sendBtn.innerHTML = `<span id="sendBtnIcon"><svg xmlns="http://www.w3.org/2000/svg"
                width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                class="feather feather-send">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg></span>`;
        }
        _streamAbortController = null;
        _onSendComplete();
    }

    if (_generationId === myGenId) {
        await reloadChat();
    }
}

// ── Queue interceptor ─────────────────────────────────────────────────────────
const _originalSend = send;

send = async function() {
    if (isSending) {
        const msgInput = document.getElementById('msg');
        const msg      = (msgInput && msgInput.value.trim()) || '';
        if (!msg && !attachedFiles.length) return;
        _msgQueue.push({ msg, files: attachedFiles.length ? attachedFiles.map(f => Object.assign({}, f)) : [] });
        if (msgInput) { msgInput.value = ''; msgInput.style.height = 'auto'; }
        showToast(`⏳ Queued (#${_msgQueue.length}) — fires when AI finishes`, 'info', 2800);
        return;
    }
    return _originalSend();
};

// Rebind after override
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('sendBtn');
    if (btn) btn.onclick = send;

    // Load persisted idle settings from server on every page load
    _loadIdleSettings();

    // ── IDLE input focus/blur/pointerdown guards ──────────────────────────
    // focus  → cancel countdown immediately (reliable on all platforms)
    // blur   → save + resume (reliable on desktop + most mobile)
    // pointerdown on document → "tapped outside" universal fallback for iOS/Android
    //   where blur can fire late, fire twice, or not fire at all when the
    //   virtual keyboard dismisses without another element taking focus.
    const secInput    = document.getElementById('idleSecondsInput');
    const promptInput = document.getElementById('idlePromptInput');

    function _onIdleInputFocus() {
        _idleInputFocused = true;
        _cancelIdle();  // kill any pending countdown immediately
    }

    function _onIdleInputBlur() {
        // Guard: ignore spurious blur events (iOS fires blur+focus rapidly on scroll)
        if (!_idleInputFocused) return;
        _idleInputFocused = false;
        // Re-read interval in case the user changed it
        if (secInput) {
            const parsed = Math.max(1, parseInt(secInput.value) || 8);
            secInput.value  = parsed;           // normalise display
            _idleIntervalMs = parsed * 1000;
        }
        _saveIdleSettings();                    // silent persist
        if (_idleEnabled) _scheduleIdle();      // resume with fresh interval
    }

    if (secInput) {
        secInput.addEventListener('focus', _onIdleInputFocus);
        secInput.addEventListener('blur',  _onIdleInputBlur);
    }
    if (promptInput) {
        promptInput.addEventListener('focus', _onIdleInputFocus);
        promptInput.addEventListener('blur',  _onIdleInputBlur);
    }

    // pointerdown fires before focus shifts on every platform — use it as a
    // universal "tapped outside" signal to commit immediately without waiting
    // for blur. passive:true keeps scroll performance intact on mobile.
    document.addEventListener('pointerdown', function(e) {
        if (!_idleInputFocused) return;         // nothing to do
        if (e.target === secInput || e.target === promptInput) return;  // still inside
        _onIdleInputBlur();                     // treat tap-outside as blur
    }, { passive: true });

    // visualViewport resize = virtual keyboard appeared or dismissed.
    // Covers Done button / back button / swipe-down keyboard dismissal on
    // iOS and Android — none of those fire pointerdown on the document,
    // and blur is unreliable for keyboard-dismiss-without-focus-change.
    // Only acts when the viewport grows (keyboard going away) and our flag
    // is still set, meaning no other path already handled it.
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', function() {
            if (!_idleInputFocused) return;
            // Confirm neither input still has focus before committing
            const a = document.activeElement;
            if (a === secInput || a === promptInput) return;
            _onIdleInputBlur();  // keyboard gone + caret not in our inputs = safe to resume
        });
    }
});

// ── Post-send hook ────────────────────────────────────────────────────────────
function _onSendComplete() {
    if (_msgQueue.length > 0) {
        const next    = _msgQueue.shift();
        const msgInput = document.getElementById('msg');
        if (msgInput) msgInput.value = next.msg || '';
        if (next.files) attachedFiles = next.files;
        if (_msgQueue.length > 0)
            showToast(`⏳ ${_msgQueue.length} still queued`, 'info', 1800);
        setTimeout(() => _originalSend(), 100);
        return;
    }
    if (_idleEnabled) _scheduleIdle();
}

// =============================================================================
// --- END IDLE MODE MODULE ---
// =============================================================================

// =============================================================================
// --- VIDEO SETTINGS MODULE ---
// =============================================================================

const VIDEO_DEFAULTS = {
    frame_divisor:   3,
    frame_floor:     4,
    frame_cap:       60,
    jpeg_quality:    0.82,
    resolution:      512,
};

function _updateVideoPreview() {
    const divisor = document.getElementById('videoFrameDivisor')?.value ?? VIDEO_DEFAULTS.frame_divisor;
    const floor   = document.getElementById('videoFrameFloor')?.value   ?? VIDEO_DEFAULTS.frame_floor;
    const cap     = document.getElementById('videoFrameCap')?.value     ?? VIDEO_DEFAULTS.frame_cap;
    const quality = document.getElementById('videoJpegQuality')?.value  ?? VIDEO_DEFAULTS.jpeg_quality;
    const res     = document.getElementById('videoResolution')?.value   ?? VIDEO_DEFAULTS.resolution;

    const preview = document.getElementById('videoSettingsPreview');
    if (!preview) return;

    // Build preview using DOM, never innerHTML — avoids XSS with user-typed values
    preview.textContent = '';

    const lines = [
        `frames = clamp( duration / ${divisor}s,  floor=${floor},  cap=${cap} )`,
        `jpeg_quality = ${parseFloat(quality).toFixed(2)}   |   max_dim = ${res}px`,
    ];

    lines.forEach((txt, i) => {
        const span = document.createElement('span');
        span.style.color = i === 0 ? '#4a90d9' : '#888';
        span.textContent = txt;
        preview.appendChild(span);
        if (i < lines.length - 1) preview.appendChild(document.createElement('br'));
    });
}

function selectVideoRes(btn) {
    document.querySelectorAll('.video-res-pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('videoResolution').value = btn.dataset.res;
    _updateVideoPreview();
}

async function openVideoSettingsModal() {
    document.getElementById('videoSettingsModal').style.display = 'flex';
    await loadVideoSettings();
}

function closeVideoSettingsModal() {
    document.getElementById('videoSettingsModal').style.display = 'none';
}

async function loadVideoSettings() {
    try {
        const res = await fetch('/get_video_settings');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const s = await res.json();

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.value = val;
        };
        const setText = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };

        const divisor = s.frame_divisor  ?? VIDEO_DEFAULTS.frame_divisor;
        const floor   = s.frame_floor    ?? VIDEO_DEFAULTS.frame_floor;
        const cap     = s.frame_cap      ?? VIDEO_DEFAULTS.frame_cap;
        const quality = s.jpeg_quality   ?? VIDEO_DEFAULTS.jpeg_quality;
        const resPx   = s.resolution     ?? VIDEO_DEFAULTS.resolution;

        set('videoFrameDivisor',  divisor);
        setText('videoFrameDivisorValue', divisor + 's');
        setText('frameDivisorLabel',      divisor);

        set('videoFrameFloor',  floor);
        setText('videoFrameFloorValue', floor);

        set('videoFrameCap',  cap);
        setText('videoFrameCapValue', cap);

        set('videoJpegQuality',  quality);
        setText('videoJpegQualityValue', parseFloat(quality).toFixed(2));

        set('videoResolution', resPx);
        document.querySelectorAll('.video-res-pill').forEach(p => {
            p.classList.toggle('active', parseInt(p.dataset.res) === parseInt(resPx));
        });

    } catch (err) {
        console.warn('Could not load video settings, using defaults:', err);
        // Sliders already carry defaults from HTML — just update the preview
    }
    _updateVideoPreview();
}

async function saveVideoSettings() {
    const payload = {
        frame_divisor: parseInt(document.getElementById('videoFrameDivisor').value),
        frame_floor:   parseInt(document.getElementById('videoFrameFloor').value),
        frame_cap:     parseInt(document.getElementById('videoFrameCap').value),
        jpeg_quality:  parseFloat(document.getElementById('videoJpegQuality').value),
        resolution:    parseInt(document.getElementById('videoResolution').value),
    };

    try {
        const res = await fetch('/set_video_settings', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });
        const result = await res.json();
        if (res.ok) {
            showToast('Video settings saved ✓', 'success');
            closeVideoSettingsModal();
        } else {
            showToast('Error: ' + (result.error || 'Save failed'), 'error');
        }
    } catch (err) {
        console.error('saveVideoSettings error:', err);
        showToast('Save failed — check console', 'error');
    }
}

function resetVideoSettings() {
    const d = VIDEO_DEFAULTS;

    document.getElementById('videoFrameDivisor').value  = d.frame_divisor;
    document.getElementById('videoFrameDivisorValue').textContent = d.frame_divisor + 's';
    document.getElementById('frameDivisorLabel').textContent      = d.frame_divisor;

    document.getElementById('videoFrameFloor').value   = d.frame_floor;
    document.getElementById('videoFrameFloorValue').textContent = d.frame_floor;

    document.getElementById('videoFrameCap').value     = d.frame_cap;
    document.getElementById('videoFrameCapValue').textContent = d.frame_cap;

    document.getElementById('videoJpegQuality').value  = d.jpeg_quality;
    document.getElementById('videoJpegQualityValue').textContent = d.jpeg_quality.toFixed(2);

    document.getElementById('videoResolution').value = d.resolution;
    document.querySelectorAll('.video-res-pill').forEach(p => {
        p.classList.toggle('active', parseInt(p.dataset.res) === d.resolution);
    });

    _updateVideoPreview();
    showToast('Reset to defaults ↺', 'info', 1800);
}

// =============================================================================
// --- END VIDEO SETTINGS MODULE ---
// =============================================================================

