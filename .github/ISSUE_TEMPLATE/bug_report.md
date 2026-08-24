---
name: Bug report
about: Report a defect in MeshGPU's rendering, handshake, or tensor pipeline
title: "[Bug] "
labels: bug, needs-triage
assignees: []
---

## Summary

<!-- One or two sentences describing the problem. -->

## Steps to reproduce

<!-- Numbered steps that reliably trigger the bug. -->

1.
2.
3.

## Expected behavior

<!-- What should have happened? -->

## Actual behavior

<!-- What actually happened? Include error text, logs, or a screenshot. -->

## Environment

**Browser & version** (e.g. Chrome 126.0 / Edge 126 / Firefox Nightly 130):

**Operating system** (e.g. macOS 14.5, Windows 11, Ubuntu 24.04):

**Deployment** (local `npm run dev`, built `dist/`, GitHub Pages):

### GPU / WebGPU adapter details

<!-- These are shown in the dashboard's "GPU Adapter" and "Adapter Limits"
     cards. Paste them verbatim where possible. -->

- **Device** (adapter description, e.g. "NVIDIA GeForce RTX 4090"):
- **Vendor / Architecture**:
- **Estimated VRAM**:
- **WebGPU features** (`navigator.gpu` flags shown in the dashboard):
- **Adapter limits** (`maxBufferSize`, `maxStorageBufferBindingSize`, etc.):

```text
<!-- Paste adapter limits JSON/text here, if available. -->
```

### WebRTC local candidate info

<!-- Critical for handshake/connectivity bugs. Run this in the DevTools
     console and paste the output:

     const pc = new RTCPeerConnection({ iceServers: [] });
     pc.createDataChannel('probe');
     pc.onicecandidate = e => { if (!e.candidate) return;
       console.log(e.candidate.candidate); };
     pc.createOffer().then(o => pc.setLocalDescription(o));

     Also note pc.iceGatheringState and pc.iceConnectionState. -->

- **Gathered candidates** (host/srflx/relay + IPs or `.local` names):
- **`iceGatheringState`**:
- **`iceConnectionState`**:
- **Connection mode** (QR scan or Base64 copy-paste):

## Logs

<!-- Browser console output and any error text from the in-app event log. -->

```text

```

## Additional context

<!-- Anything else: network topology (same machine vs two devices), VPNs,
     browser extensions, etc. -->
