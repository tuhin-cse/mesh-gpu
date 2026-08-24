/**
 * TopologyGraph.tsx
 *
 * Canvas-based live map of the P2P pipeline: each peer node shows its assigned
 * layer range, VRAM pool and round-trip latency; arrows show the forward flow
 * of hidden-state tensors through the stages.
 */

import { useEffect, useRef } from 'react';

export interface TopologyNode {
  id: string;
  label: string;
  layerStart: number;
  layerEnd: number;
  isSelf: boolean;
  vramGiB: number | null;
  rttMs?: number | null;
}

export interface TopologyEdge {
  from: string;
  to: string;
}

const NODE_WIDTH = 116;
const NODE_HEIGHT = 62;
const NODE_GAP = 64;
const CANVAS_HEIGHT = 180;

export function TopologyGraph({ nodes, edges }: { nodes: TopologyNode[]; edges: TopologyEdge[] }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const draw = (): void => {
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, container.clientWidth);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(CANVAS_HEIGHT * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${CANVAS_HEIGHT}px`;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, CANVAS_HEIGHT);

      const sorted = [...nodes].sort((a, b) => a.layerStart - b.layerStart);

      if (sorted.length === 0) {
        ctx.fillStyle = '#71717a';
        ctx.font = '12px ui-monospace, SFMono-Regular, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No peers connected — join a room to see the pipeline.', width / 2, CANVAS_HEIGHT / 2);
        return;
      }

      const totalWidth = sorted.length * NODE_WIDTH + (sorted.length - 1) * NODE_GAP;
      const startX = Math.max(8, (width - totalWidth) / 2);
      const nodeY = (CANVAS_HEIGHT - NODE_HEIGHT) / 2;

      const positions = new Map<string, { x: number; y: number }>();
      sorted.forEach((node, index) => {
        positions.set(node.id, { x: startX + index * (NODE_WIDTH + NODE_GAP), y: nodeY });
      });

      // Pipeline flow arrows.
      ctx.strokeStyle = '#52525b';
      ctx.lineWidth = 1.5;
      for (const edge of edges) {
        const from = positions.get(edge.from);
        const to = positions.get(edge.to);
        if (!from || !to) continue;
        drawArrow(
          ctx,
          from.x + NODE_WIDTH,
          from.y + NODE_HEIGHT / 2,
          to.x,
          to.y + NODE_HEIGHT / 2,
        );
      }

      // Nodes.
      sorted.forEach((node) => {
        const pos = positions.get(node.id);
        if (!pos) return;
        const { x, y } = pos;

        roundRect(ctx, x, y, NODE_WIDTH, NODE_HEIGHT, 10);
        ctx.fillStyle = node.isSelf ? '#064e3b' : '#18181b';
        ctx.fill();
        ctx.strokeStyle = node.isSelf ? '#10b981' : '#3f3f46';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';

        ctx.fillStyle = node.isSelf ? '#6ee7b7' : '#e4e4e7';
        ctx.font = '600 12px ui-monospace, SFMono-Regular, monospace';
        ctx.fillText(truncate(node.label, 14), x + NODE_WIDTH / 2, y + 18);

        ctx.fillStyle = '#d4d4d8';
        ctx.font = '600 13px ui-monospace, SFMono-Regular, monospace';
        ctx.fillText(`L${node.layerStart}–${node.layerEnd}`, x + NODE_WIDTH / 2, y + 36);

        const meta: string[] = [];
        if (node.vramGiB !== null) meta.push(`${node.vramGiB.toFixed(1)} GB`);
        if (node.rttMs !== null && node.rttMs !== undefined) meta.push(`${node.rttMs.toFixed(0)} ms`);
        ctx.fillStyle = '#71717a';
        ctx.font = '10px ui-monospace, SFMono-Regular, monospace';
        ctx.fillText(meta.length > 0 ? meta.join(' · ') : '—', x + NODE_WIDTH / 2, y + 52);
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [nodes, edges]);

  return (
    <div ref={containerRef} className="w-full">
      <canvas ref={canvasRef} />
    </div>
  );
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLength = 8;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLength * Math.cos(angle - Math.PI / 6), y2 - headLength * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLength * Math.cos(angle + Math.PI / 6), y2 - headLength * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = '#52525b';
  ctx.fill();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}
