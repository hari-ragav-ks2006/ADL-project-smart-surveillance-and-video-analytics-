/**
 * VISION AURA — NEXT-GENERATION SURVEILLANCE & VIDEO ANALYTICS
 * Minimalist Architectural Client Controller
 * Complete 7-Stage Debug Logging Pipeline & Live Bidirectional Inference Engine
 */

(function () {
    "use strict";

    // =========================================================================
    // 1. STATE & SYSTEM CONFIGURATION
    // =========================================================================
    const state = {
        activeView: "overview", // "overview", "intrusion", "atm", "registry"
        audioEnabled: true,
        lastAlarmPlayedAt: 0,
        wsClientFeed: null,
        wsConnected: false,
        wsReconnectTimer: null,
        frameProcessingBusy: false,
        lastProcessedTimestamp: 0,
        lastRttMs: 0,
        wsFrameCount: 0,
        wsFps: 30,
        wsLastFpsCheck: Date.now(),

        // Biometric / ATM State
        webcamStream: null,
        webcamActive: false,
        atmTelemetry: {
            person_count: 0,
            face_detected: false,
            face_landmarks_visible: true,
            visibility_score: 100,
            rolling_visibility_score: 100,
            blackout_active: false,
            is_safe: true,
            status_label: "ATM Ready",
            status_level: "safe",
            landmarks: []
        },
        atmDetections: [],
        blackoutStartTime: null,

        // Intrusion State
        intrusionTelemetry: {
            is_breached: false,
            intruder_count: 0,
            status_label: "Restricted Area Secure — Perimeter Clear",
            intruder_ids: []
        },
        intrusionDetections: [],
        zonePolygon: [
            [350, 140],
            [750, 140],
            [820, 540],
            [280, 540]
        ],

        // Incident Log
        cachedAlerts: [],
        currentAlertFilter: { severity: "ALL", module: "ALL", search: "" }
    };

    // =========================================================================
    // 2. DOM REFERENCES
    // =========================================================================
    const navTabs = document.querySelectorAll(".nav-link");
    const appViews = document.querySelectorAll(".app-view");
    const navBrandLogo = document.getElementById("nav-brand-logo");

    // Top Bar
    const clockHms = document.getElementById("clock-hms");
    const clockMs = document.getElementById("clock-ms");
    const wsStatusDot = document.getElementById("ws-status-dot");
    const wsStatusLabel = document.getElementById("ws-status-label");
    const btnToggleAudio = document.getElementById("btn-toggle-audio");
    const iconSoundOn = document.getElementById("icon-sound-on");
    const iconSoundOff = document.getElementById("icon-sound-off");
    const btnExportExcelTop = document.getElementById("btn-export-excel-top");
    const btnOpenConfigModal = document.getElementById("btn-open-config-modal");
    const navThreatCount = document.getElementById("nav-threat-count");

    // Overview View
    const overviewVideo = document.getElementById("overview-video");
    const overviewCanvas = document.getElementById("overview-canvas");
    const overviewFpsVal = document.getElementById("overview-fps-val");
    const overviewRttVal = document.getElementById("overview-rtt-val");
    const overviewHudBanner = document.getElementById("overview-hud-banner");
    const statActivePersons = document.getElementById("stat-active-persons");
    const statActiveBreaches = document.getElementById("stat-active-breaches");
    const overviewStateBadge = document.getElementById("overview-state-badge");
    const overviewEventsContainer = document.getElementById("overview-events-container");
    const btnQuickSwitchAtm = document.getElementById("btn-quick-switch-atm");
    const btnGotoRegistryLink = document.getElementById("btn-goto-registry-link");

    // Intrusion View
    const intrusionVideo = document.getElementById("intrusion-video");
    const intrusionCanvas = document.getElementById("intrusion-canvas");
    const intrusionFpsVal = document.getElementById("intrusion-fps-val");
    const intrusionRttVal = document.getElementById("intrusion-rtt-val");
    const intrusionHudBanner = document.getElementById("intrusion-hud-banner");
    const intrusionPedestrianCount = document.getElementById("intrusion-pedestrian-count");
    const intrusionIntruderCount = document.getElementById("intrusion-intruder-count");
    const btnRestartIntrusionFeed = document.getElementById("btn-restart-intrusion-feed");

    // ATM View
    const atmWebcamVideo = document.getElementById("atm-webcam-video");
    const atmCanvas = document.getElementById("atm-canvas");
    const atmWebcamGate = document.getElementById("atm-webcam-gate");
    const btnActivateWebcam = document.getElementById("btn-activate-webcam");
    const btnReconnectWebcam = document.getElementById("btn-reconnect-webcam");
    const atmFpsVal = document.getElementById("atm-fps-val");
    const atmRttVal = document.getElementById("atm-rtt-val");
    const atmHudBanner = document.getElementById("atm-hud-banner");
    const atmBannerText = document.getElementById("atm-banner-text");
    const atmVisStatusBadge = document.getElementById("atm-vis-status-badge");
    const atmVisibilityPctLabel = document.getElementById("atm-visibility-pct-label");
    const atmVisibilityMeterBar = document.getElementById("atm-visibility-meter-bar");
    const atmUsersCount = document.getElementById("atm-users-count");
    const atmFaceStatusText = document.getElementById("atm-face-status-text");

    // Registry View
    const registrySearchInput = document.getElementById("registry-search-input");
    const registryFilterSeverity = document.getElementById("registry-filter-severity");
    const registryFilterModule = document.getElementById("registry-filter-module");
    const btnRefreshRegistry = document.getElementById("btn-refresh-registry");
    const btnExportExcelTable = document.getElementById("btn-export-excel-table");
    const registryTableBody = document.getElementById("registry-table-body");

    // Modals & Toast
    const evidenceModal = document.getElementById("evidence-modal");
    const btnCloseEvidenceModal = document.getElementById("btn-close-evidence-modal");
    const btnModalCloseAction = document.getElementById("btn-modal-close-action");
    const modalEvidenceImg = document.getElementById("modal-evidence-img");
    const modalEvidenceDetails = document.getElementById("modal-evidence-details");

    const configModal = document.getElementById("config-modal");
    const btnCloseConfigModal = document.getElementById("btn-close-config-modal");
    const btnCancelConfig = document.getElementById("btn-cancel-config");
    const btnSaveRuntimeConfig = document.getElementById("btn-save-runtime-config");
    const cfgIntrusionDebounce = document.getElementById("cfg-intrusion-debounce");
    const cfgFaceDebounce = document.getElementById("cfg-face-debounce");
    const cfgMultiDebounce = document.getElementById("cfg-multi-debounce");

    const globalToast = document.getElementById("global-toast");
    const toastTitle = document.getElementById("toast-title");
    const toastSubtitle = document.getElementById("toast-subtitle");
    let toastTimer = null;

    // Canvas Contexts
    const overviewCtx = overviewCanvas ? overviewCanvas.getContext("2d") : null;
    const intrusionCtx = intrusionCanvas ? intrusionCanvas.getContext("2d") : null;
    const atmCtx = atmCanvas ? atmCanvas.getContext("2d") : null;

    // Offscreen Canvas for Frame Downscaling (640x360 for high-speed sub-20ms inference)
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = 640;
    offscreenCanvas.height = 360;
    const offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

    // =========================================================================
    // 3. SOUND / AUDIO ALERT ENGINE (PRIORITY 1 FIX)
    // =========================================================================
    const audioClips = {
        siren: new Audio("/static/audio/security_siren.wav"),
        warning: new Audio("/static/audio/security_warning.wav"),
        safe: new Audio("/static/audio/security_safe.wav")
    };

    Object.values(audioClips).forEach(clip => {
        clip.load();
        clip.volume = 0.55;
    });

    function playAudioAlert(type = "warning") {
        if (!state.audioEnabled) return;
        const now = Date.now();
        if (type === "warning" && now - state.lastAlarmPlayedAt < 2500) return;
        if (type === "critical" && now - state.lastAlarmPlayedAt < 1800) return;

        state.lastAlarmPlayedAt = now;
        try {
            const sound = audioClips[type] || audioClips.warning;
            sound.currentTime = 0;
            const playPromise = sound.play();
            if (playPromise !== undefined) {
                playPromise.catch(() => {});
            }
        } catch (e) {
            console.debug("[AUDIO] Audio alert notice:", e);
        }
    }

    function stopAllAudioAlerts() {
        Object.values(audioClips).forEach(clip => {
            clip.pause();
            clip.currentTime = 0;
        });
    }

    // =========================================================================
    // 4. WEBSOCKET PIPELINE & 7-STAGE DEBUG LOGGING (PRIORITY 2 FIX)
    // =========================================================================
    function initWebSocketEngine() {
        if (state.wsClientFeed && (state.wsClientFeed.readyState === WebSocket.OPEN || state.wsClientFeed.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/client-feed`;

        try {
            const ws = new WebSocket(wsUrl);
            state.wsClientFeed = ws;

            ws.onopen = () => {
                console.log("[WS] Connected to Vision Aura real-time backend engine.");
                updateWsStatus(true, "CONNECTED");
                if (state.wsReconnectTimer) {
                    clearTimeout(state.wsReconnectTimer);
                    state.wsReconnectTimer = null;
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    state.frameProcessingBusy = false;

                    // Calculate round-trip latency
                    if (data.client_t) {
                        state.lastRttMs = Math.max(1, Math.round(Date.now() - data.client_t));
                    }

                    // FPS calculation
                    state.wsFrameCount++;
                    const now = Date.now();
                    if (now - state.wsLastFpsCheck >= 1000) {
                        state.wsFps = Math.round((state.wsFrameCount * 1000) / (now - state.wsLastFpsCheck));
                        state.wsFrameCount = 0;
                        state.wsLastFpsCheck = now;
                        updateWsStatus(true, `${state.wsFps} FPS`);
                    }

                    if (data && data.success) {
                        if (data.mode === "atm") {
                            // [Stage G]: Frontend receives and renders ATM telemetry
                            console.debug("[ATM-DEBUG][Stage G] Frontend received & rendering telemetry:", {
                                rttMs: state.lastRttMs,
                                is_safe: data.is_safe,
                                person_count: data.person_count,
                                landmarks_count: data.atm_telemetry?.landmarks?.length || 0
                            });
                            handleAtmDetectionResult(data);
                        } else if (data.mode === "restricted") {
                            handleIntrusionDetectionResult(data);
                        }
                    }
                } catch (err) {
                    state.frameProcessingBusy = false;
                    console.warn("[WS] Payload parsing error:", err);
                }
            };

            ws.onerror = (err) => {
                console.warn("[WS] Connection error:", err);
                updateWsStatus(false, "ERROR");
            };

            ws.onclose = () => {
                updateWsStatus(false, "OFFLINE");
                state.wsClientFeed = null;
                if (!state.wsReconnectTimer) {
                    state.wsReconnectTimer = setTimeout(initWebSocketEngine, 2000);
                }
            };
        } catch (e) {
            updateWsStatus(false, "DISCONNECTED");
        }
    }

    function updateWsStatus(online, label) {
        state.wsConnected = online;
        if (wsStatusDot) {
            wsStatusDot.className = online ? "status-dot online" : "status-dot offline";
        }
        if (wsStatusLabel) {
            wsStatusLabel.textContent = label;
            wsStatusLabel.style.color = online ? "#10b981" : "#ef4444";
        }
    }

    // =========================================================================
    // 5. CLOCK & CHRONOMETER
    // =========================================================================
    function startChronometer() {
        function update() {
            const now = new Date();
            const h = String(now.getUTCHours()).padStart(2, "0");
            const m = String(now.getUTCMinutes()).padStart(2, "0");
            const s = String(now.getUTCSeconds()).padStart(2, "0");
            const ms = String(now.getUTCMilliseconds()).padStart(3, "0");

            if (clockHms) clockHms.textContent = `${h}:${m}:${s}`;
            if (clockMs) clockMs.textContent = `.${ms}`;
        }
        setInterval(update, 50);
        update();
    }

    // =========================================================================
    // 6. VIEW SWITCHING & HUD NAVIGATION
    // =========================================================================
    function switchView(targetViewId) {
        const viewKey = targetViewId.replace("view-", "");
        state.activeView = viewKey;

        navTabs.forEach(tab => {
            const isMatch = tab.getAttribute("data-target") === targetViewId;
            tab.classList.toggle("active", isMatch);
        });

        appViews.forEach(view => {
            const isMatch = view.id === targetViewId;
            view.classList.toggle("active", isMatch);
        });

        if (viewKey === "overview") {
            startOverviewFeed();
        } else if (viewKey === "intrusion") {
            startIntrusionFeed();
        } else if (viewKey === "atm") {
            startAtmFeed();
        } else if (viewKey === "registry") {
            fetchAlertsRegistry();
        }
    }

    // =========================================================================
    // 7. OVERVIEW MODULE FEED
    // =========================================================================
    function ensureVideoPlaying(videoEl) {
        if (!videoEl) return;
        videoEl.muted = true;
        videoEl.defaultMuted = true;
        videoEl.playsInline = true;
        videoEl.loop = true;
        if (videoEl.paused) {
            const p = videoEl.play();
            if (p !== undefined) {
                p.catch(err => {
                    console.debug("[VIDEO] Play deferred:", err);
                });
            }
        }
    }

    function startOverviewFeed() {
        if (overviewVideo) {
            ensureVideoPlaying(overviewVideo);
        }
        requestAnimationFrame(renderOverviewCanvas);
        startOverviewStreamingLoop();
    }

    function startOverviewStreamingLoop() {
        if (state.activeView !== "overview") return;

        const now = Date.now();
        if (now - state.lastProcessedTimestamp > 45 && !state.frameProcessingBusy) {
            state.lastProcessedTimestamp = now;

            if (overviewVideo && overviewVideo.videoWidth > 0 && !overviewVideo.paused) {
                try {
                    offscreenCtx.drawImage(overviewVideo, 0, 0, 640, 360);
                    const base64Jpeg = offscreenCanvas.toDataURL("image/jpeg", 0.55);

                    if (state.wsClientFeed && state.wsClientFeed.readyState === WebSocket.OPEN) {
                        state.frameProcessingBusy = true;
                        state.wsClientFeed.send(JSON.stringify({
                            frame: base64Jpeg,
                            mode: "restricted",
                            t: now
                        }));
                    }
                } catch (e) {
                    state.frameProcessingBusy = false;
                }
            }
        }

        requestAnimationFrame(startOverviewStreamingLoop);
    }

    function renderOverviewCanvas() {
        if (!overviewCanvas || !overviewCtx) return;
        if (state.activeView !== "overview") return;

        const cw = overviewCanvas.width;
        const ch = overviewCanvas.height;
        overviewCtx.clearRect(0, 0, cw, ch);

        const scaleX = cw / 640.0;
        const scaleY = ch / 360.0;

        // Draw zone polygon (defined in 960x540 space)
        if (state.zonePolygon && state.zonePolygon.length >= 3) {
            overviewCtx.beginPath();
            state.zonePolygon.forEach((pt, i) => {
                const px = pt[0] * (cw / 960.0);
                const py = pt[1] * (ch / 540.0);
                if (i === 0) overviewCtx.moveTo(px, py);
                else overviewCtx.lineTo(px, py);
            });
            overviewCtx.closePath();
            overviewCtx.fillStyle = state.intrusionTelemetry.is_breached ? "rgba(239, 68, 68, 0.18)" : "rgba(245, 158, 11, 0.12)";
            overviewCtx.fill();
            overviewCtx.strokeStyle = state.intrusionTelemetry.is_breached ? "#ef4444" : "#f59e0b";
            overviewCtx.lineWidth = 2.5;
            overviewCtx.stroke();
        }

        // Draw tracked boxes
        state.intrusionDetections.forEach(obj => {
            const bbox = obj.bbox || [0, 0, 0, 0];
            const x1 = bbox[0] * scaleX;
            const y1 = bbox[1] * scaleY;
            const bw = (bbox[2] - bbox[0]) * scaleX;
            const bh = (bbox[3] - bbox[1]) * scaleY;

            const isIntruder = obj.is_intruder;
            const color = isIntruder ? "#ef4444" : "#10b981";

            overviewCtx.strokeStyle = color;
            overviewCtx.lineWidth = 2.5;
            overviewCtx.strokeRect(x1, y1, bw, bh);

            overviewCtx.fillStyle = isIntruder ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.10)";
            overviewCtx.fillRect(x1, y1, bw, bh);

            // Label
            const labelText = isIntruder ? `#${obj.track_id} INTRUDER` : `#${obj.track_id} PERSON`;
            overviewCtx.fillStyle = color;
            overviewCtx.font = "bold 11px Plus Jakarta Sans, sans-serif";
            overviewCtx.fillText(labelText, x1 + 4, Math.max(14, y1 - 6));
        });

        // Update HUD counters
        if (overviewFpsVal) overviewFpsVal.textContent = `${state.wsFps}.0`;
        if (overviewRttVal) overviewRttVal.textContent = `${state.lastRttMs}ms`;

        requestAnimationFrame(renderOverviewCanvas);
    }

    // =========================================================================
    // 8. PERIMETER INTRUSION MODULE FEED
    // =========================================================================
    function startIntrusionFeed() {
        if (intrusionVideo) {
            ensureVideoPlaying(intrusionVideo);
        }
        requestAnimationFrame(renderIntrusionCanvas);
        startIntrusionStreamingLoop();
    }

    function startIntrusionStreamingLoop() {
        if (state.activeView !== "intrusion") return;

        const now = Date.now();
        if (now - state.lastProcessedTimestamp > 45 && !state.frameProcessingBusy) {
            state.lastProcessedTimestamp = now;

            if (intrusionVideo && intrusionVideo.videoWidth > 0 && !intrusionVideo.paused) {
                try {
                    offscreenCtx.drawImage(intrusionVideo, 0, 0, 640, 360);
                    const base64Jpeg = offscreenCanvas.toDataURL("image/jpeg", 0.55);

                    if (state.wsClientFeed && state.wsClientFeed.readyState === WebSocket.OPEN) {
                        state.frameProcessingBusy = true;
                        state.wsClientFeed.send(JSON.stringify({
                            frame: base64Jpeg,
                            mode: "restricted",
                            t: now
                        }));
                    }
                } catch (e) {
                    state.frameProcessingBusy = false;
                }
            }
        }

        requestAnimationFrame(startIntrusionStreamingLoop);
    }

    function handleIntrusionDetectionResult(result) {
        state.intrusionDetections = result.tracked_objects || [];
        state.intrusionTelemetry = {
            is_breached: result.is_breached ?? false,
            intruder_count: result.intruder_ids?.length || (result.is_breached ? 1 : 0),
            status_label: result.status_label || "Restricted Area Secure",
            intruder_ids: result.intruder_ids || []
        };

        // Update Overview & Intrusion DOM Stats
        if (statActivePersons) statActivePersons.textContent = state.intrusionDetections.length;
        if (statActiveBreaches) statActiveBreaches.textContent = state.intrusionTelemetry.intruder_count;
        if (intrusionPedestrianCount) intrusionPedestrianCount.textContent = state.intrusionDetections.length;
        if (intrusionIntruderCount) intrusionIntruderCount.textContent = state.intrusionTelemetry.intruder_count;

        if (overviewStateBadge) {
            overviewStateBadge.className = state.intrusionTelemetry.is_breached ? "badge-severity CRITICAL" : "badge-severity INFO";
            overviewStateBadge.textContent = state.intrusionTelemetry.is_breached ? "BREACH ALERT" : "NORMAL";
        }

        if (overviewHudBanner) {
            overviewHudBanner.className = state.intrusionTelemetry.is_breached ? "hud-status-banner critical" : "hud-status-banner safe";
            overviewHudBanner.innerHTML = `<span>${state.intrusionTelemetry.status_label}</span>`;
        }

        if (intrusionHudBanner) {
            intrusionHudBanner.className = state.intrusionTelemetry.is_breached ? "hud-status-banner critical" : "hud-status-banner safe";
            intrusionHudBanner.innerHTML = `<span>${state.intrusionTelemetry.status_label}</span>`;
        }

        if (state.intrusionTelemetry.is_breached) {
            playAudioAlert("critical");
        }
    }

    function renderIntrusionCanvas() {
        if (!intrusionCanvas || !intrusionCtx) return;
        if (state.activeView !== "intrusion") return;

        const cw = intrusionCanvas.width;
        const ch = intrusionCanvas.height;
        intrusionCtx.clearRect(0, 0, cw, ch);

        const scaleX = cw / 640.0;
        const scaleY = ch / 360.0;

        // Draw Zone Polygon (defined in 960x540 space)
        if (state.zonePolygon && state.zonePolygon.length >= 3) {
            intrusionCtx.beginPath();
            state.zonePolygon.forEach((pt, i) => {
                const px = pt[0] * (cw / 960.0);
                const py = pt[1] * (ch / 540.0);
                if (i === 0) intrusionCtx.moveTo(px, py);
                else intrusionCtx.lineTo(px, py);
            });
            intrusionCtx.closePath();
            intrusionCtx.fillStyle = state.intrusionTelemetry.is_breached ? "rgba(239, 68, 68, 0.18)" : "rgba(245, 158, 11, 0.12)";
            intrusionCtx.fill();
            intrusionCtx.strokeStyle = state.intrusionTelemetry.is_breached ? "#ef4444" : "#f59e0b";
            intrusionCtx.lineWidth = 2.5;
            intrusionCtx.stroke();
        }

        // Draw Bounding Boxes
        state.intrusionDetections.forEach(obj => {
            const bbox = obj.bbox || [0, 0, 0, 0];
            const x1 = bbox[0] * scaleX;
            const y1 = bbox[1] * scaleY;
            const bw = (bbox[2] - bbox[0]) * scaleX;
            const bh = (bbox[3] - bbox[1]) * scaleY;

            const isIntruder = obj.is_intruder;
            const color = isIntruder ? "#ef4444" : "#10b981";

            intrusionCtx.strokeStyle = color;
            intrusionCtx.lineWidth = 2.5;
            intrusionCtx.strokeRect(x1, y1, bw, bh);

            intrusionCtx.fillStyle = isIntruder ? "rgba(239, 68, 68, 0.16)" : "rgba(16, 185, 129, 0.10)";
            intrusionCtx.fillRect(x1, y1, bw, bh);

            const labelText = isIntruder ? `#${obj.track_id} INTRUDER DETECTED` : `#${obj.track_id} Person`;
            intrusionCtx.fillStyle = color;
            intrusionCtx.font = "bold 11.5px Plus Jakarta Sans, sans-serif";
            intrusionCtx.fillText(labelText, x1 + 4, Math.max(16, y1 - 6));
        });

        if (intrusionFpsVal) intrusionFpsVal.textContent = `${state.wsFps}.0`;
        if (intrusionRttVal) intrusionRttVal.textContent = `${state.lastRttMs}ms`;

        requestAnimationFrame(renderIntrusionCanvas);
    }

    // =========================================================================
    // 9. ATM BIOMETRICS MODULE & STRICT LIVE WEBCAM (PRIORITY 2 & 3)
    // =========================================================================
    function startAtmFeed() {
        if (!state.webcamActive) {
            if (atmWebcamGate) atmWebcamGate.classList.remove("hidden");
            if (atmWebcamVideo) atmWebcamVideo.classList.add("hidden");
        } else {
            if (atmWebcamGate) atmWebcamGate.classList.add("hidden");
            if (atmWebcamVideo) atmWebcamVideo.classList.remove("hidden");
        }
        requestAnimationFrame(renderAtmCanvas);
        startAtmStreamingLoop();
    }

    async function activateWebcamStream() {
        try {
            if (state.webcamStream) {
                state.webcamStream.getTracks().forEach(t => t.stop());
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 360 }, facingMode: "user" },
                audio: false
            });

            state.webcamStream = stream;
            state.webcamActive = true;

            if (atmWebcamVideo) {
                atmWebcamVideo.srcObject = stream;
                atmWebcamVideo.classList.remove("hidden");
                atmWebcamVideo.play().catch(() => {});
            }

            if (atmWebcamGate) atmWebcamGate.classList.add("hidden");
            initWebSocketEngine();
            console.log("[ATM-DEBUG] Webcam stream activated successfully.");
        } catch (err) {
            console.warn("[ATM-DEBUG] Webcam access error:", err);
            showGlobalToast("Webcam Access Required", "Please allow camera permissions in your browser to run live ATM biometrics.");
        }
    }

    function startAtmStreamingLoop() {
        if (state.activeView !== "atm") return;

        const now = Date.now();
        if (now - state.lastProcessedTimestamp > 45 && !state.frameProcessingBusy && state.webcamActive) {
            state.lastProcessedTimestamp = now;

            const videoSrc = atmWebcamVideo;
            if (videoSrc && videoSrc.videoWidth > 0 && !videoSrc.paused) {
                try {
                    offscreenCtx.drawImage(videoSrc, 0, 0, 640, 360);
                    const base64Jpeg = offscreenCanvas.toDataURL("image/jpeg", 0.55);

                    if (state.wsClientFeed && state.wsClientFeed.readyState === WebSocket.OPEN) {
                        state.frameProcessingBusy = true;

                        state.wsClientFeed.send(JSON.stringify({
                            frame: base64Jpeg,
                            mode: "atm",
                            t: now
                        }));
                    }
                } catch (e) {
                    state.frameProcessingBusy = false;
                }
            }
        }

        requestAnimationFrame(startAtmStreamingLoop);
    }

    function handleAtmDetectionResult(result) {
        const tel = result.atm_telemetry || {};
        state.atmTelemetry = {
            person_count: result.person_count ?? (tel.person_count || 0),
            face_detected: tel.face_detected ?? false,
            face_bbox: tel.face_bbox || (tel.face_object ? tel.face_object.bbox : null),
            face_object: tel.face_object || null,
            face_landmarks_visible: tel.face_landmarks_visible ?? true,
            visibility_score: tel.visibility_score ?? 100,
            rolling_visibility_score: tel.rolling_visibility_score ?? 100,
            blackout_active: tel.blackout_active ?? false,
            is_safe: result.is_safe ?? true,
            status_label: result.status_label || "ATM Ready",
            status_level: result.status_level || "safe",
            landmarks: tel.landmarks || [],
            debug_info: tel.debug_info || {}
        };

        state.atmDetections = result.tracked_objects || [];
        renderAtmStatusDOM();
    }

    function renderAtmStatusDOM() {
        const tel = state.atmTelemetry;
        const count = tel.person_count;
        const faceVis = tel.face_landmarks_visible;
        const faceDet = tel.face_detected;
        const blackout = tel.blackout_active;
        const rollingScore = tel.rolling_visibility_score;

        if (atmUsersCount) atmUsersCount.textContent = count;
        if (atmVisibilityPctLabel) atmVisibilityPctLabel.textContent = `${Math.round(rollingScore)}%`;
        if (atmVisStatusBadge) atmVisStatusBadge.textContent = `${Math.round(rollingScore)}%`;

        if (atmVisibilityMeterBar) {
            const clamped = Math.max(0, Math.min(100, rollingScore));
            atmVisibilityMeterBar.style.width = `${clamped}%`;
            if (clamped < 35) {
                atmVisibilityMeterBar.className = "meter-fill danger";
            } else if (clamped < 70) {
                atmVisibilityMeterBar.className = "meter-fill warn";
            } else {
                atmVisibilityMeterBar.className = "meter-fill";
            }
        }

        if (blackout) {
            if (atmBannerText) atmBannerText.textContent = "CRITICAL: CAMERA OBSTRUCTED / TAMPER DETECTED";
            if (atmHudBanner) atmHudBanner.className = "hud-status-banner critical";
            if (atmFaceStatusText) {
                atmFaceStatusText.textContent = "TAMPERED";
                atmFaceStatusText.className = "big-stat-number danger";
            }
            playAudioAlert("critical");
        } else if (count > 1) {
            if (atmBannerText) atmBannerText.textContent = "Multiple people detected — please use the ATM one person at a time";
            if (atmHudBanner) atmHudBanner.className = "hud-status-banner warning";
            if (atmFaceStatusText) {
                atmFaceStatusText.textContent = "MULTIPLE";
                atmFaceStatusText.className = "big-stat-number warn";
            }
            playAudioAlert("warning");
        } else if (count >= 1 && (!faceVis || rollingScore < 50)) {
            if (atmBannerText) atmBannerText.textContent = "Face covering detected — please remove it to proceed";
            if (atmHudBanner) atmHudBanner.className = "hud-status-banner warning";
            if (atmFaceStatusText) {
                atmFaceStatusText.textContent = "COVERED";
                atmFaceStatusText.className = "big-stat-number warn";
            }
            playAudioAlert("warning");
        } else if (count >= 1 && faceVis && rollingScore >= 50) {
            if (atmBannerText) atmBannerText.textContent = "Safe to use ATM (Face Verified)";
            if (atmHudBanner) atmHudBanner.className = "hud-status-banner safe";
            if (atmFaceStatusText) {
                atmFaceStatusText.textContent = "VERIFIED";
                atmFaceStatusText.className = "big-stat-number safe";
            }
        } else {
            if (atmBannerText) atmBannerText.textContent = "ATM Ready / Standby";
            if (atmHudBanner) atmHudBanner.className = "hud-status-banner safe";
            if (atmFaceStatusText) {
                atmFaceStatusText.textContent = "READY";
                atmFaceStatusText.className = "big-stat-number safe";
            }
        }
    }

    function renderAtmCanvas() {
        if (!atmCanvas || !atmCtx) return;
        if (state.activeView !== "atm") return;

        const cw = atmCanvas.width;
        const ch = atmCanvas.height;
        atmCtx.clearRect(0, 0, cw, ch);

        const scaleX = cw / 640.0;
        const scaleY = ch / 360.0;
        const tel = state.atmTelemetry;

        // 1. Draw Person Bounding Boxes (YOLOv8 ByteTrack + Multi-Face Fused)
        state.atmDetections.forEach(obj => {
            const bbox = obj.bbox || [0, 0, 0, 0];
            const x1 = bbox[0] * scaleX;
            const y1 = bbox[1] * scaleY;
            const bw = (bbox[2] - bbox[0]) * scaleX;
            const bh = (bbox[3] - bbox[1]) * scaleY;

            let color = "#10b981"; // Emerald green for safe single user
            let label = `#${obj.track_id} PERSON: SAFE`;
            let isMulti = tel.person_count > 1;

            if (isMulti) {
                color = "#f59e0b"; // Vibrant Orange for Multiple People
                label = `#${obj.track_id} PERSON: MULTIPLE PEOPLE`;
            } else if (tel.face_detected && !tel.face_landmarks_visible) {
                color = "#f59e0b"; // Orange for face covered
                label = `#${obj.track_id} PERSON: FACE COVERED`;
            } else {
                color = "#10b981"; // Emerald green
                label = `#${obj.track_id} PERSON: SAFE`;
            }

            // Outer glow & stroke
            atmCtx.strokeStyle = color;
            atmCtx.lineWidth = isMulti ? 3.0 : 2.5;
            atmCtx.strokeRect(x1, y1, bw, bh);

            // Shaded person area
            atmCtx.fillStyle = isMulti ? "rgba(245, 158, 11, 0.16)" : (color === "#10b981" ? "rgba(16, 185, 129, 0.10)" : "rgba(245, 158, 11, 0.14)");
            atmCtx.fillRect(x1, y1, bw, bh);

            // Label tag badge
            atmCtx.font = "bold 11.5px Plus Jakarta Sans, sans-serif";
            const textWidth = atmCtx.measureText(label).width;
            const tagH = 20;
            const tagY = Math.max(0, y1 - tagH);

            atmCtx.fillStyle = color;
            atmCtx.fillRect(x1, tagY, textWidth + 12, tagH);

            atmCtx.fillStyle = isMulti ? "#000000" : (color === "#10b981" ? "#000000" : "#ffffff");
            atmCtx.fillText(label, x1 + 6, tagY + 14);
        });

        // 4. Debug Visibility Overlay (Step 1d - toggleable on-screen text)
        const showDebug = true; // Flag for live debug overlay
        if (showDebug && state.webcamActive) {
            const dbgX = 14;
            const dbgY = 14;
            const dbgW = 280;
            const dbgH = 68;

            atmCtx.fillStyle = "rgba(10, 14, 22, 0.82)";
            atmCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
            atmCtx.lineWidth = 1;
            atmCtx.beginPath();
            atmCtx.roundRect(dbgX, dbgY, dbgW, dbgH, 6);
            atmCtx.fill();
            atmCtx.stroke();

            const pFound = tel.person_count > 0 ? `YES (${tel.person_count})` : "NO (0)";
            const fFound = tel.face_detected ? "YES" : "NO";
            const mnConf = `${Math.round(tel.visibility_score)}%`;
            const rollConf = `${Math.round(tel.rolling_visibility_score)}%`;
            const stateStr = tel.blackout_active ? "TAMPER" : (tel.person_count > 1 ? "MULTI" : (tel.face_detected ? (tel.face_landmarks_visible ? "SAFE" : "COVERED") : "STANDBY"));
            const stateColor = (stateStr === "SAFE" || stateStr === "STANDBY") ? "#10b981" : "#f59e0b";

            atmCtx.fillStyle = "rgba(255, 255, 255, 0.75)";
            atmCtx.font = "600 10.5px JetBrains Mono, monospace";
            atmCtx.fillText(`PERSON: ${pFound}  |  FACE: ${fFound}`, dbgX + 10, dbgY + 20);
            atmCtx.fillText(`CONF: ${mnConf} (ROLLING: ${rollConf})`, dbgX + 10, dbgY + 38);
            atmCtx.fillStyle = stateColor;
            atmCtx.fillText(`STATE: ${stateStr}`, dbgX + 10, dbgY + 56);
        }

        if (atmFpsVal) atmFpsVal.textContent = state.webcamActive ? `${state.wsFps}.0` : "0.0";
        if (atmRttVal) atmRttVal.textContent = state.webcamActive ? `${state.lastRttMs}ms` : "0ms";

        requestAnimationFrame(renderAtmCanvas);
    }

    // =========================================================================
    // 10. INCIDENT LOG REGISTRY & API INTEGRATION
    // =========================================================================
    async function fetchAlertsRegistry() {
        try {
            const res = await fetch("/api/alerts?limit=50");
            if (!res.ok) return;
            const data = await res.json();
            state.cachedAlerts = data.alerts || [];

            if (navThreatCount) navThreatCount.textContent = state.cachedAlerts.length;
            renderRegistryTable();
            renderRecentEventsSidebar();
        } catch (e) {
            console.debug("[REGISTRY] Fetch notice:", e);
        }
    }

    function renderRegistryTable() {
        if (!registryTableBody) return;
        registryTableBody.innerHTML = "";

        const { severity, module, search } = state.currentAlertFilter;

        const filtered = state.cachedAlerts.filter(a => {
            if (severity !== "ALL" && a.severity !== severity) return false;
            if (module !== "ALL" && a.module_name !== module) return false;
            if (search) {
                const s = search.toLowerCase();
                const matchMsg = (a.message || "").toLowerCase().includes(s);
                const matchType = (a.alert_type || "").toLowerCase().includes(s);
                const matchZone = (a.zone_name || "").toLowerCase().includes(s);
                if (!matchMsg && !matchType && !matchZone) return false;
            }
            return true;
        });

        if (filtered.length === 0) {
            registryTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 24px; color: var(--text-dim);">No incidents matching active filters.</td></tr>`;
            return;
        }

        filtered.forEach(a => {
            const tr = document.createElement("tr");
            const timeStr = a.timestamp ? a.timestamp.replace("T", " ").substring(0, 19) : "—";
            const sevClass = `badge-severity ${a.severity || "INFO"}`;

            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-weight: 600;">#${a.id}</td>
                <td style="font-family: var(--font-mono);">${timeStr}</td>
                <td><span class="${sevClass}">${a.severity}</span></td>
                <td style="font-weight: 600;">${a.alert_type}</td>
                <td style="color: var(--text-muted);">${a.zone_name || a.module_name}</td>
                <td>${a.message}</td>
                <td><span style="font-family: var(--font-mono); color: ${a.is_active ? 'var(--accent-crimson)' : 'var(--accent-emerald)'};">${a.is_active ? 'ACTIVE' : 'RESOLVED'}</span></td>
                <td>
                    <button class="btn-pill-action btn-view-evidence" data-id="${a.id}" style="padding: 3px 8px; font-size: 11.5px;">View Photo</button>
                </td>
            `;
            registryTableBody.appendChild(tr);
        });

        // Attach event listeners to view buttons
        document.querySelectorAll(".btn-view-evidence").forEach(btn => {
            btn.addEventListener("click", () => {
                const aid = parseInt(btn.getAttribute("data-id"), 10);
                const alertItem = state.cachedAlerts.find(x => x.id === aid);
                if (alertItem) openEvidenceModal(alertItem);
            });
        });
    }

    function renderRecentEventsSidebar() {
        if (!overviewEventsContainer) return;
        const recent = state.cachedAlerts.slice(0, 5);

        if (recent.length === 0) {
            overviewEventsContainer.innerHTML = `<div class="empty-state-notice">System secure. No recent threats logged.</div>`;
            return;
        }

        overviewEventsContainer.innerHTML = "";
        recent.forEach(a => {
            const div = document.createElement("div");
            div.className = "event-feed-item";
            const timeStr = a.timestamp ? a.timestamp.replace("T", " ").substring(11, 19) : "—";

            div.innerHTML = `
                <div class="event-severity-dot ${a.severity}"></div>
                <div class="event-info">
                    <div class="event-title-line">
                        <span>${a.alert_type}</span>
                        <span class="event-time">${timeStr}</span>
                    </div>
                    <div class="event-desc">${a.message}</div>
                </div>
            `;
            overviewEventsContainer.appendChild(div);
        });
    }

    function openEvidenceModal(alertItem) {
        if (!evidenceModal) return;
        if (modalEvidenceImg) {
            modalEvidenceImg.src = alertItem.screenshot_path ? `/${alertItem.screenshot_path}` : "/sample_videos/000000001000.jpg";
            modalEvidenceImg.onerror = () => {
                modalEvidenceImg.src = "/sample_videos/000000001000.jpg";
            };
        }
        if (modalEvidenceDetails) {
            modalEvidenceDetails.innerHTML = `
                <div><strong>Incident ID:</strong> #${alertItem.id}</div>
                <div><strong>Timestamp:</strong> ${alertItem.timestamp}</div>
                <div><strong>Severity:</strong> <span class="badge-severity ${alertItem.severity}">${alertItem.severity}</span></div>
                <div><strong>Type:</strong> ${alertItem.alert_type} (${alertItem.zone_name || alertItem.module_name})</div>
                <div><strong>Message:</strong> ${alertItem.message}</div>
            `;
        }
        evidenceModal.classList.add("open");
    }

    function showGlobalToast(title, subtitle) {
        if (!globalToast) return;
        if (toastTitle) toastTitle.textContent = title;
        if (toastSubtitle) toastSubtitle.textContent = subtitle;

        globalToast.classList.add("show");
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            globalToast.classList.remove("show");
        }, 4000);
    }

    // =========================================================================
    // 11. WIRE ALL INTERACTIVE CONTROLS
    // =========================================================================
    function wireEventHandlers() {
        // Nav Brand
        if (navBrandLogo) {
            navBrandLogo.addEventListener("click", () => switchView("view-overview"));
        }

        // Nav Tabs
        navTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                const target = tab.getAttribute("data-target");
                if (target) switchView(target);
            });
        });

        // Quick Switch Links
        if (btnQuickSwitchAtm) {
            btnQuickSwitchAtm.addEventListener("click", () => switchView("view-atm"));
        }
        if (btnGotoRegistryLink) {
            btnGotoRegistryLink.addEventListener("click", () => switchView("view-registry"));
        }

        // Webcam Activations
        if (btnActivateWebcam) {
            btnActivateWebcam.addEventListener("click", activateWebcamStream);
        }
        if (btnReconnectWebcam) {
            btnReconnectWebcam.addEventListener("click", activateWebcamStream);
        }

        // Video Monitor Direct Click-to-Play
        const overviewMonitor = document.getElementById("overview-stage-monitor");
        if (overviewMonitor && overviewVideo) {
            overviewMonitor.addEventListener("click", () => {
                ensureVideoPlaying(overviewVideo);
            });
        }

        const intrusionMonitor = document.getElementById("intrusion-stage-monitor");
        if (intrusionMonitor && intrusionVideo) {
            intrusionMonitor.addEventListener("click", () => {
                ensureVideoPlaying(intrusionVideo);
            });
        }

        // Video Restart
        if (btnRestartIntrusionFeed) {
            btnRestartIntrusionFeed.addEventListener("click", () => {
                if (intrusionVideo) {
                    intrusionVideo.currentTime = 0;
                    ensureVideoPlaying(intrusionVideo);
                }
            });
        }

        // Audio Toggle
        if (btnToggleAudio) {
            btnToggleAudio.addEventListener("click", () => {
                state.audioEnabled = !state.audioEnabled;
                if (!state.audioEnabled) {
                    stopAllAudioAlerts();
                    if (iconSoundOn) iconSoundOn.classList.add("hidden");
                    if (iconSoundOff) iconSoundOff.classList.remove("hidden");
                } else {
                    if (iconSoundOn) iconSoundOn.classList.remove("hidden");
                    if (iconSoundOff) iconSoundOff.classList.add("hidden");
                    playAudioAlert("safe");
                }
            });
        }

        // Excel Exports
        if (btnExportExcelTop) {
            btnExportExcelTop.addEventListener("click", () => {
                window.location.href = "/api/alerts/export";
            });
        }
        if (btnExportExcelTable) {
            btnExportExcelTable.addEventListener("click", () => {
                window.location.href = "/api/alerts/export";
            });
        }

        // Refresh Registry
        if (btnRefreshRegistry) {
            btnRefreshRegistry.addEventListener("click", fetchAlertsRegistry);
        }

        // Registry Search & Filters
        if (registrySearchInput) {
            registrySearchInput.addEventListener("input", (e) => {
                state.currentAlertFilter.search = e.target.value.trim();
                renderRegistryTable();
            });
        }
        if (registryFilterSeverity) {
            registryFilterSeverity.addEventListener("change", (e) => {
                state.currentAlertFilter.severity = e.target.value;
                renderRegistryTable();
            });
        }
        if (registryFilterModule) {
            registryFilterModule.addEventListener("change", (e) => {
                state.currentAlertFilter.module = e.target.value;
                renderRegistryTable();
            });
        }

        // Evidence Modal Close
        if (btnCloseEvidenceModal) {
            btnCloseEvidenceModal.addEventListener("click", () => evidenceModal.classList.remove("open"));
        }
        if (btnModalCloseAction) {
            btnModalCloseAction.addEventListener("click", () => evidenceModal.classList.remove("open"));
        }
        if (evidenceModal) {
            evidenceModal.addEventListener("click", (e) => {
                if (e.target === evidenceModal) evidenceModal.classList.remove("open");
            });
        }

        // Config Modal
        if (btnOpenConfigModal) {
            btnOpenConfigModal.addEventListener("click", () => configModal.classList.add("open"));
        }
        if (btnCloseConfigModal) {
            btnCloseConfigModal.addEventListener("click", () => configModal.classList.remove("open"));
        }
        if (btnCancelConfig) {
            btnCancelConfig.addEventListener("click", () => configModal.classList.remove("open"));
        }
        if (configModal) {
            configModal.addEventListener("click", (e) => {
                if (e.target === configModal) configModal.classList.remove("open");
            });
        }
        if (btnSaveRuntimeConfig) {
            btnSaveRuntimeConfig.addEventListener("click", async () => {
                const intr = parseFloat(cfgIntrusionDebounce?.value || 0.5);
                const face = parseFloat(cfgFaceDebounce?.value || 1.5);
                const multi = parseFloat(cfgMultiDebounce?.value || 1.5);

                try {
                    await fetch("/api/config/thresholds", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            intrusion_debounce: intr,
                            face_covered_debounce: face,
                            multi_person_debounce: multi
                        })
                    });
                    configModal.classList.remove("open");
                    showGlobalToast("Configuration Saved", "Surveillance debounce thresholds updated successfully.");
                } catch (e) {
                    console.debug("[CONFIG] Save error:", e);
                }
            });
        }
    }

    // =========================================================================
    // 12. INITIALIZATION LIFECYCLE
    // =========================================================================
    window.addEventListener("DOMContentLoaded", () => {
        startChronometer();
        wireEventHandlers();
        initWebSocketEngine();
        startOverviewFeed();
        fetchAlertsRegistry();
        setInterval(fetchAlertsRegistry, 5000);
    });

})();
