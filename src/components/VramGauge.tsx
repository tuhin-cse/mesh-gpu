/**
 * VramGauge.tsx
 *
 * Semicircular SVG gauge showing how much of this node's VRAM pool is consumed
 * by its assigned pipeline stage.
 */

export interface VramGaugeProps {
  usedBytes: number | null;
  totalBytes: number | null;
  label?: string;
}

export function VramGauge({ usedBytes, totalBytes, label = 'VRAM' }: VramGaugeProps) {
  const pct =
    totalBytes !== null && totalBytes > 0 && usedBytes !== null
      ? Math.min(1, Math.max(0, usedBytes / totalBytes))
      : 0;

  const radius = 80;
  const arcLength = Math.PI * radius; // semicircle
  const viewWidth = 200;
  const viewHeight = 112;
  const centerX = viewWidth / 2;
  const baselineY = 100;
  const startX = centerX - radius;
  const endX = centerX + radius;
  const arcPath = `M ${startX} ${baselineY} A ${radius} ${radius} 0 0 1 ${endX} ${baselineY}`;

  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${viewWidth} ${viewHeight}`} className="w-full max-w-[220px]">
        <path
          d={arcPath}
          fill="none"
          stroke="#27272a"
          strokeWidth={14}
          strokeLinecap="round"
        />
        <path
          d={arcPath}
          fill="none"
          stroke="#34d399"
          strokeWidth={14}
          strokeLinecap="round"
          strokeDasharray={`${(pct * arcLength).toFixed(1)} ${arcLength.toFixed(1)}`}
        />
        <text
          x={centerX}
          y={baselineY - 10}
          textAnchor="middle"
          className="fill-zinc-100 text-2xl font-semibold"
        >
          {(pct * 100).toFixed(0)}%
        </text>
        <text x={centerX} y={baselineY + 14} textAnchor="middle" className="fill-zinc-500 text-xs">
          {label}
        </text>
      </svg>
      <div className="mt-1 text-xs text-zinc-400">
        {formatGiB(usedBytes)} / {formatGiB(totalBytes)}
      </div>
    </div>
  );
}

const GIB = 1024 ** 3;

function formatGiB(bytes: number | null): string {
  if (bytes === null) return '—';
  return `${(bytes / GIB).toFixed(1)} GB`;
}
