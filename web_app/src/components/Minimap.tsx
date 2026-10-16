// src/components/Minimap.tsx
interface Props {
  cameraPos: { x: number; y: number; z: number; rotation: number };
}

export function Minimap({ cameraPos }: Props) {
  // Approximate scale and bounds mapping
  const scale = 24;
  const cx = 80;
  const cy = 80;
  const px = cx + cameraPos.x * scale;
  const py = cy + cameraPos.z * scale;
  const angleDeg = (cameraPos.rotation * 180) / Math.PI;

  return (
    <div className="absolute top-5 right-5 z-20 w-40 h-40 bg-slate-900/85 backdrop-blur-md border border-slate-700/60 rounded-2xl shadow-xl p-3 flex flex-col">
      <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400 mb-2">
        <span>FLOOR PLAN</span>
        <span className="text-blue-400">2D TOP-DOWN</span>
      </div>

      <div className="relative flex-1 bg-slate-950/60 rounded-xl overflow-hidden border border-slate-800">
        {/* Room boundary mockup */}
        <div className="absolute inset-3 border border-slate-700/80 rounded-md" />
        <div className="absolute top-4 left-4 w-12 h-16 border border-slate-800 rounded bg-slate-800/30" />
        <div className="absolute bottom-4 right-4 w-14 h-10 border border-slate-800 rounded bg-slate-800/30" />

        {/* Dynamic camera indicator */}
        <div
          className="absolute w-3.5 h-3.5 -ml-[7px] -mt-[7px] pointer-events-none transition-transform duration-75"
          style={{
            left: `${Math.max(10, Math.min(150, px))}px`,
            top: `${Math.max(10, Math.min(110, py))}px`,
            transform: `rotate(${angleDeg}deg)`,
          }}
        >
          {/* Direction cone */}
          <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[10px] border-b-blue-400 mx-auto" />
          {/* Position dot */}
          <div className="w-2.5 h-2.5 bg-blue-500 rounded-full border border-white shadow-md mx-auto -mt-1" />
        </div>
      </div>
    </div>
  );
}
