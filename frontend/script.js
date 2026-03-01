/* ================================================================
   J.A.R.V.I.S Frontend — Main Application Logic
   ================================================================

   ARCHITECTURE OVERVIEW
   ---------------------
   This file powers the entire frontend of the J.A.R.V.I.S AI assistant.
   It handles:

   1. CHAT MESSAGING — The user types (or speaks) a message, which is
      sent to the backend via a POST request. The backend responds using
      Server-Sent Events (SSE), allowing the reply to stream in
      token-by-token (like ChatGPT's typing effect).

   2. TEXT-TO-SPEECH (TTS) — When TTS is enabled, the backend also
      sends base64-encoded audio chunks inside the SSE stream. These
      are queued up and played sequentially through a single <audio>
      element. This queue-based approach prevents overlapping audio
      and supports mobile browsers (especially iOS/Safari).

   3. SPEECH RECOGNITION — The Web Speech API captures the user's
      voice, transcribes it in real time, and auto-sends the final
      transcript as a chat message.

   4. ANIMATED ORB — A WebGL-powered visual orb (rendered by a
      separate OrbRenderer class) acts as a visual indicator. It
      "activates" when J.A.R.V.I.S is speaking and goes idle otherwise.

   5. MODE SWITCHING — The UI supports two modes:
      - "General" mode  → uses the /chat/stream endpoint
      - "Realtime" mode → uses the /chat/realtime/stream endpoint
      The mode determines which backend pipeline processes the message.

   6. SESSION MANAGEMENT — A session ID is returned by the server on
      the first message. Subsequent messages include that ID so the
      backend can maintain conversation context. Starting a "New Chat"
      clears the session.

   DATA FLOW (simplified):
   User input → sendMessage() → POST to backend → SSE stream opens →
   tokens arrive as JSON chunks → rendered into the DOM in real time →
   optional audio chunks are enqueued in TTSPlayer → played sequentially.

   ================================================================ */

/*
 * API — The base URL for all backend requests.
 *
 * In production, this resolves to the same origin the page was loaded from
 * (e.g., "https://jarvis.example.com"). During local development, it falls
 * back to "http://localhost:8000" (the default FastAPI dev server port).
 *
 * `window.location.origin` gives us the protocol + host + port of the
 * current page, making the frontend deployment-agnostic (no hardcoded URLs).
 */
const API = (typeof window !== 'undefined' && window.location.origin)
    ? window.location.origin
    : 'http://localhost:8000';

/* ================================================================
   APPLICATION STATE
   ================================================================
   These variables track the global state of the application. They are
   intentionally kept as simple top-level variables rather than in a
   class or store, since this is a single-page app with one chat view.
   ================================================================ */

/*
 * sessionId — Unique conversation identifier returned by the server.
 * Starts as null (no conversation yet). Once the first server response
 * arrives, it contains a UUID string that we send back with every
 * subsequent message so the backend knows which conversation we're in.
 */
let sessionId = null;

/*
 * currentMode — Which AI pipeline to use: 'general' or 'realtime'.
 * This determines which backend endpoint we POST to (/chat/stream
 * vs /chat/realtime/stream). The mode can be toggled via the UI buttons.
 */
let currentMode = 'general';

/*
 * isStreaming — Guard flag that is true while an SSE response is being
 * received. Prevents the user from sending another message while the
 * assistant is still replying (avoids race conditions and garbled output).
 */
let isStreaming = false;

/*
 * isListening — True while the speech recognition engine is actively
 * capturing audio from the microphone. Used to toggle the mic button
 * styling and to decide whether to start or stop listening on click.
 */
let isListening = false;

/*
 * orb — Reference to the OrbRenderer instance (the animated WebGL orb).
 * Null if OrbRenderer is unavailable or failed to initialize.
 * We call orb.setActive(true/false) to animate it during TTS playback.
 */
let orb = null;

/*
 * recognition — The SpeechRecognition instance from the Web Speech API.
 * Null if the browser doesn't support speech recognition.
 */
let recognition = null;

/*
 * ttsPlayer — Instance of the TTSPlayer class (defined below) that
 * manages queuing and playing audio segments received from the server.
 */
let ttsPlayer = null;

/* ================================================================
   DOM REFERENCES
   ================================================================
   We grab references to frequently-used DOM elements once at startup
   rather than querying for them every time we need them. This is both
   a performance optimization and a readability convenience.
   ================================================================ */

/*
 * $ — Shorthand helper for document.getElementById. Writing $('foo')
 * is more concise than document.getElementById('foo').
 */
const $ = id => document.getElementById(id);

const chatMessages = $('chat-messages');   // The scrollable container that holds all chat messages
const messageInput = $('message-input');   // The <textarea> where the user types their message
const sendBtn      = $('send-btn');        // The send button (arrow icon)
const micBtn       = $('mic-btn');         // The microphone button for speech-to-text
const ttsBtn       = $('tts-btn');         // The speaker button to toggle text-to-speech
const newChatBtn   = $('new-chat-btn');    // The "New Chat" button that resets the conversation
const modeLabel    = $('mode-label');      // Displays the current mode name ("General Mode" / "Realtime Mode")
const charCount    = $('char-count');      // Shows character count when the message gets long
const welcomeTitle = $('welcome-title');   // The greeting text on the welcome screen ("Good morning.", etc.)
const modeSlider   = $('mode-slider');     // The sliding pill indicator behind the mode toggle buttons
const btnGeneral   = $('btn-general');     // The "General" mode button
const btnRealtime  = $('btn-realtime');    // The "Realtime" mode button
const statusDot    = document.querySelector('.status-dot');  // Green/red dot showing backend status
const statusText   = document.querySelector('.status-text'); // Text next to the dot ("Online" / "Offline")
const orbContainer = $('orb-container');   // The container <div> that holds the WebGL orb canvas
const searchResultsToggle = $('search-results-toggle');   // Header button to open search results panel
const searchResultsWidget = $('search-results-widget');   // Right-side panel for Tavily search data
const searchResultsClose  = $('search-results-close');    // Close button inside the panel
const searchResultsQuery  = $('search-results-query');    // Displays the search query
const searchResultsAnswer = $('search-results-answer');   // Displays the AI answer from search
const searchResultsList   = $('search-results-list');     // Container for source result cards

/* ================================================================
   TTS AUDIO PLAYER (Text-to-Speech Queue System)
   ================================================================

   HOW THE TTS QUEUE WORKS — EXPLAINED FOR LEARNERS
   -------------------------------------------------
   When TTS is enabled, the backend doesn't send one giant audio file.
   Instead, it sends many small base64-encoded MP3 *chunks* as part of
   the SSE stream (one chunk per sentence or phrase). This approach has
   two advantages:
     1. Audio starts playing before the full response is generated
        (lower latency — the user hears the first sentence immediately).
     2. Each chunk is small, so there's no long download wait.

   The TTSPlayer works like a conveyor belt:
     - enqueue() adds a new audio chunk to the end of the queue.
     - _playLoop() picks up chunks one by one and plays them.
     - When a chunk finishes playing (audio.onended), the loop moves
       to the next chunk.
     - When the queue is empty and no more chunks are arriving, playback
       stops and the orb goes back to idle.

   WHY A SINGLE <audio> ELEMENT?
   iOS Safari has strict autoplay policies — it only allows audio
   playback from a user-initiated event. By reusing one <audio> element
   that was "unlocked" during a user gesture, all subsequent plays
   through that same element are allowed. Creating new Audio() objects
   each time would trigger autoplay blocks on iOS.

   ================================================================ */
class TTSPlayer {
    /**
     * Creates a new TTSPlayer instance.
     *
     * Properties:
     *   queue    — Array of base64 audio strings waiting to be played.
     *   playing  — True if the play loop is currently running.
     *   enabled  — True if the user has toggled TTS on (via the speaker button).
     *   stopped  — True if playback was forcibly stopped (e.g., new chat).
     *              This prevents queued audio from playing after a stop.
     *   audio    — A single persistent <audio> element reused for all playback.
     */
    constructor() {
        this.queue = [];
        this.playing = false;
        this.enabled = true;   // TTS on by default
        this.stopped = false;
        this.audio = document.createElement('audio');
        this.audio.preload = 'auto';
    }

    /**
     * unlock() — "Warms up" the audio element so browsers (especially iOS
     * Safari) allow subsequent programmatic playback.
     *
     * This should be called during a user gesture (e.g., clicking "Send").
     *
     * It does two things:
     *   1. Plays a tiny silent WAV file on the <audio> element, which
     *      tells the browser "the user initiated audio playback."
     *   2. Creates a brief AudioContext oscillator at zero volume — this
     *      unlocks the Web Audio API context on iOS (a separate lock from
     *      the <audio> element).
     *
     * After this, the browser treats subsequent .play() calls on the same
     * <audio> element as user-initiated, even if they happen in an async
     * callback (like our SSE stream handler).
     */
    unlock() {
        // A minimal valid WAV file (44-byte header + 2 bytes of silence)
        const silentWav = 'data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA';
        this.audio.src = silentWav;
        const p = this.audio.play();
        if (p) p.catch(() => {});
        try {
            // Create a Web Audio context and play a zero-volume oscillator for <1ms
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const g = ctx.createGain();
            g.gain.value = 0;
            const o = ctx.createOscillator();
            o.connect(g);
            g.connect(ctx.destination);
            o.start(0);
            o.stop(ctx.currentTime + 0.001);
            setTimeout(() => ctx.close(), 200);
        } catch (_) {}
    }

    /**
     * enqueue(base64Audio) — Adds a base64-encoded MP3 chunk to the
     * playback queue.
     *
     * @param {string} base64Audio - The base64 string of the MP3 audio data.
     *
     * If TTS is disabled or playback has been force-stopped, the chunk
     * is silently discarded. Otherwise it's pushed onto the queue.
     * If the play loop isn't already running, we kick it off.
     */
    enqueue(base64Audio) {
        if (!this.enabled || this.stopped) return;
        this.queue.push(base64Audio);
        if (!this.playing) this._playLoop();
    }

    /**
     * stop() — Immediately halts all audio playback and clears the queue.
     *
     * Called when:
     *   - The user starts a "New Chat"
     *   - The user toggles TTS off while audio is playing
     *   - We need to reset before a new streaming response
     *
     * It also removes visual indicators (CSS classes on the TTS button,
     * the orb container, and deactivates the orb animation).
     */
    stop() {
        this.stopped = true;
        this.audio.pause();
        this.audio.removeAttribute('src');
        this.audio.load();                        // Fully resets the audio element
        this.queue = [];                           // Discard any pending audio chunks
        this.playing = false;
        if (ttsBtn) ttsBtn.classList.remove('tts-speaking');
        if (orbContainer) orbContainer.classList.remove('speaking');
        if (orb) orb.setActive(false);
    }

    /**
     * reset() — Stops playback AND clears the "stopped" flag so new
     * audio can be enqueued again.
     *
     * Called at the beginning of each new message send. Without clearing
     * `this.stopped`, enqueue() would keep discarding audio from the
     * previous stop() call.
     */
    reset() {
        this.stop();
        this.stopped = false;
    }

    /**
     * _playLoop() — The internal playback engine. Processes the queue
     * one chunk at a time in a while-loop.
     *
     * WHY THE LOOP ID (_loopId)?
     * If stop() is called and then a new stream starts, there could be
     * two concurrent _playLoop() calls — the old one (still awaiting a
     * Promise) and the new one. The loop ID lets us detect when a loop
     * has been superseded: each invocation gets a unique ID, and if the
     * ID changes mid-loop (because a new loop started), the old loop
     * exits gracefully. This prevents double-playback or stale loops.
     *
     * VISUAL INDICATORS:
     * While playing, we add CSS classes 'tts-speaking' (to the button)
     * and 'speaking' (to the orb container) for visual feedback. These
     * are removed when the queue is drained or playback is stopped.
     */
    async _playLoop() {
        if (this.playing) return;
        this.playing = true;
        this._loopId = (this._loopId || 0) + 1;
        const myId = this._loopId;

        // Activate visual indicators: button glow + orb animation
        if (ttsBtn) ttsBtn.classList.add('tts-speaking');
        if (orbContainer) orbContainer.classList.add('speaking');
        if (orb) orb.setActive(true);

        // Process queued audio chunks one at a time
        while (this.queue.length > 0) {
            if (this.stopped || myId !== this._loopId) break;  // Exit if stopped or superseded
            const b64 = this.queue.shift();                     // Take the next chunk from the front
            try {
                await this._playB64(b64);                       // Wait for it to finish playing
            } catch (e) {
                console.warn('TTS segment error:', e);
            }
        }

        // If another loop took over, don't touch the shared state
        if (myId !== this._loopId) return;
        this.playing = false;
        // Deactivate visual indicators
        if (ttsBtn) ttsBtn.classList.remove('tts-speaking');
        if (orbContainer) orbContainer.classList.remove('speaking');
        if (orb) orb.setActive(false);
    }

    /**
     * _playB64(b64) — Plays a single base64-encoded MP3 chunk.
     *
     * @param {string} b64 - Base64-encoded MP3 audio data.
     * @returns {Promise<void>} Resolves when the audio finishes playing
     *                          (or errors out).
     *
     * Sets the <audio> element's src to a data URL and calls .play().
     * Returns a Promise that resolves on 'ended' or 'error', so the
     * _playLoop() can await it and move to the next chunk.
     */
    _playB64(b64) {
        return new Promise(resolve => {
            this.audio.src = 'data:audio/mp3;base64,' + b64;
            const done = () => { resolve(); };
            this.audio.onended = done;   // Normal completion
            this.audio.onerror = done;   // Error — resolve anyway so the loop continues
            const p = this.audio.play();
            if (p) p.catch(done);        // Handle play() rejection (e.g., autoplay block)
        });
    }
}

/* ================================================================
   INITIALIZATION
   ================================================================
   init() is the entry point for the entire application. It is called
   once when the DOM is fully loaded (see the DOMContentLoaded listener
   at the bottom of this file).

   It sets up every subsystem in the correct order:
     1. TTSPlayer — so audio is ready before any messages
     2. Greeting  — display a time-appropriate welcome message
     3. Orb       — initialize the WebGL visual
     4. Speech    — set up the microphone / speech recognition
     5. Health    — ping the backend to check if it's online
     6. Events    — wire up all button clicks and keyboard shortcuts
     7. Input     — auto-resize the textarea to fit content
   ================================================================ */
function init() {
    ttsPlayer = new TTSPlayer();
    if (ttsBtn) ttsBtn.classList.add('tts-active');   // Show TTS as on by default
    setGreeting();
    initOrb();
    initSpeech();
    checkHealth();
    bindEvents();
    autoResizeInput();
}

/* ================================================================
   GREETING
   ================================================================ */

/**
 * setGreeting() — Sets the welcome screen title based on the current
 * time of day.
 *
 * Time ranges:
 *   00:00–11:59 → "Good morning."
 *   12:00–16:59 → "Good afternoon."
 *   17:00–21:59 → "Good evening."
 *   22:00–23:59 → "Burning the midnight oil?" (a fun late-night touch)
 *
 * This is called on page load and when starting a new chat.
 */
function setGreeting() {
    const h = new Date().getHours();
    let g = 'Good evening.';
    if (h < 12) g = 'Good morning.';
    else if (h < 17) g = 'Good afternoon.';
    else if (h >= 22) g = 'Burning the midnight oil?';
    welcomeTitle.textContent = g;
}

/* ================================================================
   WEBGL ORB INITIALIZATION
   ================================================================ */

/**
 * initOrb() — Creates the animated WebGL orb inside the orbContainer.
 *
 * OrbRenderer is defined in a separate JS file (orb.js). If that file
 * hasn't loaded (e.g., network error), OrbRenderer will be undefined
 * and we skip initialization gracefully.
 *
 * Configuration:
 *   hue: 0                           — The base hue of the orb color
 *   hoverIntensity: 0.3              — How much the orb reacts to mouse hover
 *   backgroundColor: [0.02,0.02,0.06] — Near-black dark blue background (RGB, 0–1 range)
 *
 * The orb's "active" state (pulsing animation) is toggled via
 * orb.setActive(true/false), which we call when TTS starts/stops.
 */
function initOrb() {
    if (typeof OrbRenderer === 'undefined') return;
    try {
        orb = new OrbRenderer(orbContainer, {
            hue: 0,
            hoverIntensity: 0.3,
            backgroundColor: [0.02, 0.02, 0.06]
        });
    } catch (e) { console.warn('Orb init failed:', e); }
}

/* ================================================================
   SPEECH RECOGNITION (Speech-to-Text)
   ================================================================

   HOW SPEECH RECOGNITION WORKS — EXPLAINED FOR LEARNERS
   ------------------------------------------------------
   The Web Speech API (SpeechRecognition) is a browser-native feature
   that converts spoken audio from the microphone into text. Here's
   the lifecycle:

   1. User clicks the mic button → startListening() is called.
   2. recognition.start() begins capturing audio from the mic.
   3. As the user speaks, the browser fires 'result' events with
      partial (interim) transcripts. We display these in the input
      field in real time so the user sees what's being recognized.
   4. When the user pauses, the browser finalizes the transcript
      (result.isFinal becomes true).
   5. On finalization, we stop listening and automatically send the
      recognized text as a chat message.

   IMPORTANT PROPERTIES:
   - continuous: false → Stops after one utterance (sentence). If true,
     it would keep listening for multiple sentences.
   - interimResults: true → We get partial results as the user speaks
     (not just the final result). This gives real-time feedback.
   - lang: 'en-US' → Optimize recognition for American English.

   BROWSER SUPPORT: Chrome has the best support. Firefox and Safari
   have limited or no support for this API. We gracefully degrade by
   checking if the API exists before using it.
   ================================================================ */

/**
 * initSpeech() — Sets up the SpeechRecognition instance and its
 * event handlers.
 *
 * If the browser doesn't support the API, we update the mic button's
 * tooltip to inform the user and return early.
 */
function initSpeech() {
    // SpeechRecognition is prefixed in some browsers (webkit for Chrome/Safari)
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { micBtn.title = 'Speech not supported in this browser'; return; }

    recognition = new SR();
    recognition.continuous = false;       // Stop after one complete utterance
    recognition.interimResults = true;    // Emit partial results for real-time feedback
    recognition.lang = 'en-US';           // Recognition language

    // Fired every time the recognizer has a new or updated result
    recognition.onresult = e => {
        const result = e.results[e.results.length - 1];  // Get the latest result
        const text = result[0].transcript;                // The recognized text string
        messageInput.value = text;                        // Show it in the input field
        autoResizeInput();                                // Resize textarea to fit
        if (result.isFinal) {
            // The browser has finalized this utterance — send it
            stopListening();
            if (text.trim()) sendMessage(text.trim());
        }
    };
    recognition.onerror = () => stopListening();                       // Stop on any recognition error
    recognition.onend = () => { if (isListening) stopListening(); };   // Clean up if recognition ends unexpectedly
}

/**
 * startListening() — Activates the microphone and begins speech recognition.
 *
 * Guards:
 *   - Does nothing if recognition isn't available (unsupported browser).
 *   - Does nothing if we're currently streaming a response (to avoid
 *     accidentally sending a voice message mid-stream).
 */
function startListening() {
    if (!recognition || isStreaming) return;
    isListening = true;
    micBtn.classList.add('listening');     // Visual feedback: highlight the mic button
    try { recognition.start(); } catch (_) {}
}

/**
 * stopListening() — Deactivates the microphone and stops recognition.
 *
 * Called when:
 *   - A final transcript is received (auto-send).
 *   - The user clicks the mic button again (manual toggle off).
 *   - An error occurs.
 *   - The recognition engine stops unexpectedly.
 */
function stopListening() {
    isListening = false;
    micBtn.classList.remove('listening');  // Remove visual highlight
    try { recognition.stop(); } catch (_) {}
}

/* ================================================================
   BACKEND HEALTH CHECK
   ================================================================ */

/**
 * checkHealth() — Pings the backend's /health endpoint to determine
 * if the server is running and healthy.
 *
 * Updates the status indicator in the UI:
 *   - Green dot + "Online"  if the server responds with { status: "healthy" }
 *   - Red dot   + "Offline" if the request fails or returns unhealthy
 *
 * Uses AbortSignal.timeout(5000) to avoid waiting forever if the
 * server is down — the request will automatically abort after 5 seconds.
 */
async function checkHealth() {
    try {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
        const d = await r.json();
        const ok = d.status === 'healthy';
        statusDot.classList.toggle('offline', !ok);   // Add 'offline' class if NOT healthy
        statusText.textContent = ok ? 'Online' : 'Offline';
    } catch {
        statusDot.classList.add('offline');
        statusText.textContent = 'Offline';
    }
}

/* ================================================================
   EVENT BINDING
   ================================================================
   All user-interaction event listeners are centralized here for
   clarity. This function is called once during init().
   ================================================================ */

/**
 * bindEvents() — Wires up all click, keydown, and input event
 * listeners for the UI.
 */
function bindEvents() {
    // SEND BUTTON — Send the message when clicked (if not already streaming)
    sendBtn.addEventListener('click', () => { if (!isStreaming) sendMessage(); });

    // ENTER KEY — Send on Enter (but allow Shift+Enter for new lines)
    messageInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!isStreaming) sendMessage(); }
    });

    // INPUT CHANGE — Auto-resize the textarea and show character count for long messages
    messageInput.addEventListener('input', () => {
        autoResizeInput();
        const len = messageInput.value.length;
        // Only show the counter once the message exceeds 100 characters (avoids clutter)
        charCount.textContent = len > 100 ? `${len.toLocaleString()} / 32,000` : '';
    });

    // MIC BUTTON — Toggle speech recognition on/off
    micBtn.addEventListener('click', () => { isListening ? stopListening() : startListening(); });

    // TTS BUTTON — Toggle text-to-speech on/off
    ttsBtn.addEventListener('click', () => {
        ttsPlayer.enabled = !ttsPlayer.enabled;
        ttsBtn.classList.toggle('tts-active', ttsPlayer.enabled);  // Visual indicator
        if (!ttsPlayer.enabled) ttsPlayer.stop();                  // Stop any playing audio immediately
    });

    // NEW CHAT BUTTON — Reset the conversation
    newChatBtn.addEventListener('click', newChat);

    // MODE TOGGLE BUTTONS — Switch between General and Realtime modes
    btnGeneral.addEventListener('click',  () => setMode('general'));
    btnRealtime.addEventListener('click', () => setMode('realtime'));

    // QUICK-ACTION CHIPS — Predefined messages on the welcome screen
    // Each chip has a data-msg attribute containing the message to send
    document.querySelectorAll('.chip').forEach(c => {
        c.addEventListener('click', () => { if (!isStreaming) sendMessage(c.dataset.msg); });
    });

    // SEARCH RESULTS WIDGET — Toggle panel open from header button; close from panel button
    if (searchResultsToggle) {
        searchResultsToggle.addEventListener('click', () => {
            if (searchResultsWidget) searchResultsWidget.classList.add('open');
        });
    }
    if (searchResultsClose && searchResultsWidget) {
        searchResultsClose.addEventListener('click', () => searchResultsWidget.classList.remove('open'));
    }
}

/**
 * autoResizeInput() — Dynamically adjusts the textarea height to fit
 * its content, up to a maximum of 120px.
 *
 * How it works:
 *   1. Reset height to 'auto' so scrollHeight reflects actual content height.
 *   2. Set height to the smaller of scrollHeight or 120px.
 *   This creates a textarea that grows as the user types but doesn't
 *   take over the whole screen for very long messages.
 */
function autoResizeInput() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

/* ================================================================
   MODE SWITCH (General ↔ Realtime)
   ================================================================
   The app supports two AI modes, each hitting a different backend
   endpoint:
     - "General"  → /chat/stream         (standard LLM pipeline)
     - "Realtime" → /chat/realtime/stream (realtime/low-latency pipeline)

   The mode is purely a UI + routing concern — the frontend logic for
   streaming and rendering is identical for both modes.
   ================================================================ */

/**
 * setMode(mode) — Switches the active mode and updates the UI.
 *
 * @param {string} mode - Either 'general' or 'realtime'.
 *
 * Updates:
 *   - currentMode variable (used when sending messages)
 *   - Button active states (highlights the selected button)
 *   - Slider position (slides the pill indicator left or right)
 *   - Mode label text (displayed in the header area)
 */
function setMode(mode) {
    currentMode = mode;
    btnGeneral.classList.toggle('active', mode === 'general');
    btnRealtime.classList.toggle('active', mode === 'realtime');
    modeSlider.classList.toggle('right', mode === 'realtime');    // CSS slides the pill to the right
    modeLabel.textContent = mode === 'general' ? 'General Mode' : 'Realtime Mode';
}

/* ================================================================
   NEW CHAT
   ================================================================ */

/**
 * newChat() — Resets the entire conversation to a fresh state.
 *
 * Steps:
 *   1. Stop any playing TTS audio.
 *   2. Clear the session ID (server will create a new one on next message).
 *   3. Clear all messages from the chat container.
 *   4. Re-create and display the welcome screen.
 *   5. Clear the input field and reset its size.
 *   6. Update the greeting text (in case time-of-day changed).
 */
function newChat() {
    if (ttsPlayer) ttsPlayer.stop();
    sessionId = null;
    chatMessages.innerHTML = '';
    chatMessages.appendChild(createWelcome());
    messageInput.value = '';
    autoResizeInput();
    setGreeting();
    if (searchResultsWidget) searchResultsWidget.classList.remove('open');
    if (searchResultsToggle) searchResultsToggle.style.display = 'none';
}

/**
 * createWelcome() — Builds and returns the welcome screen DOM element.
 *
 * @returns {HTMLDivElement} The welcome screen element, ready to be
 *                           appended to the chat container.
 *
 * The welcome screen includes:
 *   - A decorative SVG icon
 *   - A time-based greeting (same logic as setGreeting)
 *   - A subtitle prompt ("How may I assist you today?")
 *   - Quick-action chip buttons with predefined messages
 *
 * The chip buttons get their own click listeners here because they
 * are dynamically created (not present in the original HTML).
 */
function createWelcome() {
    const h = new Date().getHours();
    let g = 'Good evening.';
    if (h < 12) g = 'Good morning.';
    else if (h < 17) g = 'Good afternoon.';
    else if (h >= 22) g = 'Burning the midnight oil?';

    const div = document.createElement('div');
    div.className = 'welcome-screen';
    div.id = 'welcome-screen';
    div.innerHTML = `
        <div class="welcome-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        </div>
        <h2 class="welcome-title">${g}</h2>
        <p class="welcome-sub">How may I assist you today?</p>
        <div class="welcome-chips">
            <button class="chip" data-msg="What can you do?">What can you do?</button>
            <button class="chip" data-msg="Open YouTube for me">Open YouTube</button>
            <button class="chip" data-msg="Tell me a fun fact">Fun fact</button>
            <button class="chip" data-msg="Play some music">Play music</button>
        </div>`;

    // Attach click handlers to the dynamically created chip buttons
    div.querySelectorAll('.chip').forEach(c => {
        c.addEventListener('click', () => { if (!isStreaming) sendMessage(c.dataset.msg); });
    });
    return div;
}

/* ================================================================
   MESSAGE RENDERING
   ================================================================
   These functions build the chat message DOM elements. Each message
   consists of:
     - An avatar circle ("J" for Jarvis, "U" for user)
     - A body containing a label (name + mode) and the content text

   The structure mirrors common chat UIs (Slack, Discord, ChatGPT).
   ================================================================ */

/**
 * isUrlLike(str) — True if the string looks like a URL or encoded path (not a readable title/snippet).
 */
function isUrlLike(str) {
    if (!str || typeof str !== 'string') return false;
    const s = str.trim();
    return s.length > 40 && (/^https?:\/\//i.test(s) || /\%2f|\%3a|\.com\/|\.org\//i.test(s));
}

/**
 * friendlyUrlLabel(url) — Short, readable label for a URL (domain + path hint) for display.
 */
function friendlyUrlLabel(url) {
    if (!url || typeof url !== 'string') return 'View source';
    try {
        const u = new URL(url.startsWith('http') ? url : 'https://' + url);
        const host = u.hostname.replace(/^www\./, '');
        const path = u.pathname !== '/' ? u.pathname.slice(0, 20) + (u.pathname.length > 20 ? '…' : '') : '';
        return path ? host + path : host;
    } catch (_) {
        return url.length > 40 ? url.slice(0, 37) + '…' : url;
    }
}

/**
 * truncateSnippet(text, maxLen) — Truncate to maxLen with ellipsis, one line for card content.
 */
function truncateSnippet(text, maxLen) {
    if (!text || typeof text !== 'string') return '';
    const t = text.trim();
    if (t.length <= maxLen) return t;
    return t.slice(0, maxLen).trim() + '…';
}

/**
 * renderSearchResults(payload) — Fills the right-side search results widget
 * with Tavily data (query, AI answer, and source cards). Filters junk, truncates
 * content, and shows friendly URL labels so layout stays clean and responsive.
 */
function renderSearchResults(payload) {
    if (!payload) return;
    if (searchResultsQuery) searchResultsQuery.textContent = (payload.query || '').trim() || 'Search';
    if (searchResultsAnswer) searchResultsAnswer.textContent = (payload.answer || '').trim() || '';
    if (!searchResultsList) return;
    searchResultsList.innerHTML = '';
    const results = payload.results || [];
    const maxContentLen = 220;
    for (const r of results) {
        let title = (r.title || '').trim();
        let content = (r.content || '').trim();
        const url = (r.url || '').trim();
        if (isUrlLike(title)) title = friendlyUrlLabel(url) || 'Source';
        if (!title) title = friendlyUrlLabel(url) || 'Source';
        if (isUrlLike(content)) content = '';
        content = truncateSnippet(content, maxContentLen);
        const score = r.score != null ? Math.round((r.score || 0) * 100) : null;
        const card = document.createElement('div');
        card.className = 'search-result-card';
        const urlDisplay = url ? escapeHtml(friendlyUrlLabel(url)) : '';
        const urlSafe = url ? url.replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;') : '';
        card.innerHTML = `
            <div class="card-title">${escapeHtml(title)}</div>
            ${content ? `<div class="card-content">${escapeHtml(content)}</div>` : ''}
            ${url ? `<a href="${urlSafe}" target="_blank" rel="noopener" class="card-url" title="${escapeAttr(url)}">${urlDisplay}</a>` : ''}
            ${score != null ? `<div class="card-score">Relevance: ${escapeHtml(String(score))}%</div>` : ''}`;
        searchResultsList.appendChild(card);
    }
}

/**
 * escapeAttr(str) — Escape for HTML attribute (e.g. href, title).
 */
function escapeAttr(str) {
    if (typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML.replace(/"/g, '&quot;');
}

/**
 * escapeHtml(str) — Escapes & < > " ' for safe insertion into HTML.
 */
function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * hideWelcome() — Removes the welcome screen from the DOM.
 *
 * Called before adding the first message, since the welcome screen
 * should disappear once a conversation begins.
 */
function hideWelcome() {
    const w = document.getElementById('welcome-screen');
    if (w) w.remove();
}

/**
 * addMessage(role, text) — Creates and appends a chat message bubble.
 *
 * @param {string} role - Either 'user' or 'assistant'. Determines
 *                         styling, avatar letter, and label text.
 * @param {string} text - The message content to display.
 * @returns {HTMLDivElement} The inner content element — returned so
 *                           the caller (sendMessage) can update it
 *                           later during streaming.
 *
 * DOM structure created:
 *   <div class="message user|assistant">
 *     <div class="msg-avatar"><svg>...</svg></div>
 *     <div class="msg-body">
 *       <div class="msg-label">Jarvis (General) | You</div>
 *       <div class="msg-content">...text...</div>
 *     </div>
 *   </div>
 */
/* Inline SVG icons for chat avatars (user = person, assistant = bot). */
const AVATAR_ICON_USER = '<svg class="msg-avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
const AVATAR_ICON_ASSISTANT = '<svg class="msg-avatar-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><circle cx="9" cy="16" r="1" fill="currentColor"/><circle cx="15" cy="16" r="1" fill="currentColor"/></svg>';

function addMessage(role, text) {
    hideWelcome();
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = role === 'assistant' ? AVATAR_ICON_ASSISTANT : AVATAR_ICON_USER;

    const body = document.createElement('div');
    body.className = 'msg-body';

    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = role === 'assistant'
        ? `Jarvis  (${currentMode === 'realtime' ? 'Realtime' : 'General'})`
        : 'You';

    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;

    body.appendChild(label);
    body.appendChild(content);
    msg.appendChild(avatar);
    msg.appendChild(body);
    chatMessages.appendChild(msg);
    scrollToBottom();
    return content;  // Returned so the streaming logic can update it in real time
}

/**
 * addTypingIndicator() — Shows an animated "..." typing indicator
 * while waiting for the assistant's response to begin streaming.
 *
 * @returns {HTMLDivElement} The content element (containing the dots).
 *
 * This creates a message bubble that looks like the assistant is
 * typing. It's removed once actual content starts arriving.
 * The three <span> elements inside .typing-dots are animated via CSS
 * to create the bouncing dots effect.
 */
function addTypingIndicator() {
    hideWelcome();
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.id = 'typing-msg';               // ID so we can find and remove it later

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = AVATAR_ICON_ASSISTANT;

    const body = document.createElement('div');
    body.className = 'msg-body';

    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = `Jarvis  (${currentMode === 'realtime' ? 'Realtime' : 'General'})`;

    const content = document.createElement('div');
    content.className = 'msg-content';
    content.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

    body.appendChild(label);
    body.appendChild(content);
    msg.appendChild(avatar);
    msg.appendChild(body);
    chatMessages.appendChild(msg);
    scrollToBottom();
    return content;
}

/**
 * removeTypingIndicator() — Removes the typing indicator from the DOM.
 *
 * Called when:
 *   - The first token of the response arrives (replaced by real content).
 *   - An error occurs (replaced by an error message).
 */
function removeTypingIndicator() {
    const t = document.getElementById('typing-msg');
    if (t) t.remove();
}

/**
 * scrollToBottom() — Scrolls the chat container to show the latest message.
 *
 * Uses requestAnimationFrame so the scroll runs after the browser has
 * laid out newly added content (typing indicator, "Thinking...", or
 * streamed chunks). Without this, scroll can happen before layout and
 * the user would have to scroll manually to see new content.
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

/* ================================================================
   SEND MESSAGE + SSE STREAMING
   ================================================================

   HOW SSE (Server-Sent Events) STREAMING WORKS — EXPLAINED FOR LEARNERS
   ----------------------------------------------------------------------
   Instead of waiting for the entire AI response to generate (which
   could take seconds), we use SSE streaming to receive the response
   token-by-token as it's generated. This creates the "typing" effect.

   STANDARD SSE FORMAT:
   The server sends a stream of lines like:
     data: {"chunk": "Hello"}
     data: {"chunk": " there"}
     data: {"chunk": "!"}
     data: {"done": true}

   Each line starts with "data: " followed by a JSON payload. Lines
   are separated by newlines ("\n"). An empty line separates events.

   HOW WE READ THE STREAM:
   1. We POST the user's message to the backend.
   2. The server responds with Content-Type: text/event-stream.
   3. We use res.body.getReader() to read the response body as a
      stream of raw bytes (Uint8Array chunks).
   4. We decode each chunk to text and append it to an SSE buffer.
   5. We split the buffer by newlines and process each complete line.
   6. Lines starting with "data: " are parsed as JSON.
   7. Each JSON payload may contain:
      - chunk: a piece of the text response (appended to the UI)
      - audio: a base64 MP3 segment (enqueued for TTS playback)
      - session_id: the conversation ID (saved for future messages)
      - error: an error message from the server
      - done: true when the response is complete

   WHY NOT USE EventSource?
   The native EventSource API only supports GET requests. We need POST
   (to send the message body), so we use fetch() + manual SSE parsing.

   THE SSE BUFFER:
   Network chunks don't align with SSE line boundaries — one chunk
   might contain half a line, or multiple lines. The sseBuffer variable
   accumulates raw text. We split by '\n', process all complete lines,
   and keep the last (potentially incomplete) line in the buffer for
   the next iteration.

   ================================================================ */

/**
 * sendMessage(textOverride) — The main function that sends a user
 * message and streams the AI's response.
 *
 * @param {string} [textOverride] - Optional text to send instead of
 *                                   the input field's value. Used by
 *                                   chip buttons and voice input.
 *
 * This is an async function because it awaits the streaming fetch
 * response. The full flow:
 *
 *   1. Get the message text (from parameter or input field).
 *   2. Clear the input field and show the user's message in the chat.
 *   3. Show a typing indicator while waiting for the server.
 *   4. Lock the UI (isStreaming = true, disable send button).
 *   5. Reset the TTS player and unlock audio for iOS.
 *   6. POST to the appropriate endpoint based on currentMode.
 *   7. Read the SSE stream chunk by chunk.
 *   8. For each data line: parse JSON, append text to the DOM,
 *      enqueue audio, save session ID.
 *   9. When done, clean up the streaming cursor and unlock the UI.
 *  10. On error, show an error message in the chat.
 */
async function sendMessage(textOverride) {
    // Step 1: Get the message text, trimming whitespace
    const text = (textOverride || messageInput.value).trim();
    if (!text || isStreaming) return;  // Ignore empty messages or if already streaming

    // Step 2: Clear the input field immediately (responsive UX)
    messageInput.value = '';
    autoResizeInput();
    charCount.textContent = '';

    // Step 3: Display the user's message and show typing indicator
    addMessage('user', text);
    addTypingIndicator();

    // Step 4: Lock the UI to prevent double-sending
    isStreaming = true;
    sendBtn.disabled = true;

    // Step 5: Reset TTS for this new response and unlock audio (iOS)
    if (ttsPlayer) { ttsPlayer.reset(); ttsPlayer.unlock(); }

    // Step 6: Choose the endpoint based on the current mode
    const endpoint = currentMode === 'realtime' ? '/chat/realtime/stream' : '/chat/stream';

    try {
        // Step 7: Send the POST request to the backend
        const res = await fetch(`${API}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,                                 // The user's message
                session_id: sessionId,                         // null on first message; UUID after that
                tts: !!(ttsPlayer && ttsPlayer.enabled)        // Tell the backend whether to generate audio
            }),
        });

        // Handle HTTP errors (4xx, 5xx)
        if (!res.ok) {
            const err = await res.json().catch(() => null);
            throw new Error(err?.detail || `HTTP ${res.status}`);
        }

        // Step 8: Replace the typing indicator with an empty assistant message
        removeTypingIndicator();
        const contentEl = addMessage('assistant', '');
        const placeholder = currentMode === 'realtime' ? 'Searching...' : 'Thinking...';
        contentEl.innerHTML = `<span class="msg-stream-text">${placeholder}</span>`;
        scrollToBottom();   // Scroll so placeholder is visible without manual scroll

        // Set up the stream reader and SSE parser
        const reader = res.body.getReader();       // ReadableStream reader for the response body
        const decoder = new TextDecoder();          // Converts raw bytes (Uint8Array) to strings
        let sseBuffer = '';                         // Accumulates partial SSE lines between chunks
        let fullResponse = '';                      // The complete assistant response text so far
        let cursorEl = null;                        // The blinking "|" cursor shown during streaming

        // Step 9: Read the stream in a loop until it's done
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;  // Stream has ended

            // Decode the bytes and add to our SSE buffer
            sseBuffer += decoder.decode(value, { stream: true });

            // Split by newlines to get individual SSE lines
            const lines = sseBuffer.split('\n');

            // The last element might be an incomplete line — keep it in the buffer
            sseBuffer = lines.pop();

            // Process each complete line
            for (const line of lines) {
                // SSE lines that don't start with "data: " are empty lines or comments — skip them
                if (!line.startsWith('data: ')) continue;
                try {
                    // Parse the JSON payload (everything after "data: ")
                    const data = JSON.parse(line.slice(6));

                    // Save the session ID if the server sends one
                    if (data.session_id) sessionId = data.session_id;

                    // SEARCH RESULTS — Tavily data (realtime): show in right-side widget and reveal toggle
                    if (data.search_results) {
                        renderSearchResults(data.search_results);
                        if (searchResultsToggle) searchResultsToggle.style.display = '';
                        if (searchResultsWidget) searchResultsWidget.classList.add('open');
                    }

                    // TEXT CHUNK — Append to the displayed response
                    if (data.chunk) {
                        fullResponse += data.chunk;
                        const textSpan = contentEl.querySelector('.msg-stream-text');
                        if (textSpan) textSpan.textContent = fullResponse;

                        // Add a blinking cursor at the end (created once, on the first chunk)
                        if (!cursorEl) {
                            cursorEl = document.createElement('span');
                            cursorEl.className = 'stream-cursor';
                            cursorEl.textContent = '|';
                            contentEl.appendChild(cursorEl);
                        }
                        scrollToBottom();
                    }

                    // AUDIO CHUNK — Enqueue for TTS playback
                    if (data.audio && ttsPlayer) {
                        ttsPlayer.enqueue(data.audio);
                    }

                    // ERROR — The server reported an error in the stream
                    if (data.error) throw new Error(data.error);

                    // DONE — The server signals that the response is complete
                    if (data.done) break;
                } catch (parseErr) {
                    // Ignore JSON parse errors (e.g., partial lines) but re-throw real errors
                    if (parseErr.message && !parseErr.message.includes('JSON'))
                        throw parseErr;
                }
            }
        }

        // Step 10: Clean up — remove the blinking cursor
        if (cursorEl) cursorEl.remove();

        // If the server sent nothing, show a placeholder
        const textSpan = contentEl.querySelector('.msg-stream-text');
        if (textSpan && !fullResponse) textSpan.textContent = '(No response)';

    } catch (err) {
        // On any error, remove the typing indicator and show the error
        removeTypingIndicator();
        addMessage('assistant', `Something went wrong: ${err.message}`);
    } finally {
        // Always unlock the UI, whether the request succeeded or failed
        isStreaming = false;
        sendBtn.disabled = false;
    }
}

/* ================================================================
   BOOT — Application Entry Point
   ================================================================
   DOMContentLoaded fires when the HTML document has been fully parsed
   (but before images/stylesheets finish loading). This is the ideal
   time to initialize our app because all DOM elements are available.
   ================================================================ */
document.addEventListener('DOMContentLoaded', init);
